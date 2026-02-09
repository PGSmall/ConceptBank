import torch
import numpy as np
from PIL import Image
from contextlib import nullcontext
from typing import Optional, Tuple, Dict

from mmseg.models.segmentors import BaseSegmentor
from mmengine.structures import PixelData
from mmseg.registry import MODELS

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.data_misc import FindStage, interpolate


def to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        x = img
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        if x.dtype != np.uint8:
            if x.max() <= 1.5:
                x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
            else:
                x = np.clip(x, 0, 255).astype(np.uint8)
        return Image.fromarray(x).convert("RGB")
    if torch.is_tensor(img):
        x = img.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0).contiguous()
        x = x.numpy()
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        if x.dtype != np.uint8:
            if x.max() <= 1.5:
                x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
            else:
                x = np.clip(x, 0, 255).astype(np.uint8)
        return Image.fromarray(x).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def slice_along_dim(t: torch.Tensor, start: int, end: int, dim: int) -> torch.Tensor:
    sl = [slice(None)] * t.ndim
    sl[dim] = slice(start, end)
    return t[tuple(sl)]


def _select_user_subbank(bank: dict, concept_user: Optional[str]) -> Tuple[dict, str]:
    if isinstance(bank, dict) and isinstance(bank.get("users", None), dict):
        users = bank["users"]
        if concept_user is None:
            if len(users) == 1:
                concept_user = next(iter(users.keys()))
            else:
                raise ValueError(
                    f"concept_path is a multi-user bank with users={list(users.keys())}, "
                    f"but concept_user is None."
                )
        if concept_user not in users:
            raise KeyError(f"concept_user='{concept_user}' not found in bank users={list(users.keys())}")
        return users[concept_user], concept_user
    return bank, "single_user"


@MODELS.register_module()
class SAM3_OVSS(BaseSegmentor):
    def __init__(
        self,
        concept_path: str,
        concept_user: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        bg_thr: float = 0.0,
        ig_thr: float = 0.0,
        bg_idx: int = 0,
        use_sem_seg: bool = True,
        use_presence_instance: bool = True,
        topk_inst: int = 30,
        prompt_chunk_size: int = 60,
        checkpoint_path: str = "sam3/assets/sam3.pt",
        bpe_path: str = "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        compile: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.device = torch.device(device)

        self.confidence_threshold = float(confidence_threshold)
        self.bg_thr = float(bg_thr)
        self.ig_thr = float(ig_thr)
        self.bg_idx = int(bg_idx)

        self.use_sem_seg = bool(use_sem_seg)
        self.use_presence_instance = bool(use_presence_instance)
        self.topk_inst = int(topk_inst)
        self.prompt_chunk_size = int(prompt_chunk_size)

        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
            device=device,
            eval_mode=True,
            compile=bool(compile),
        )
        self.model.eval()
        self.processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold, device=device)

        bank_all = torch.load(concept_path, map_location="cpu")
        subbank, selected_user = _select_user_subbank(bank_all, concept_user)

        if "query_words" not in subbank or "text_outputs" not in subbank:
            raise RuntimeError(
                "Bad concept bank format. Expected keys ['query_words','text_outputs'] "
                f"in selected subbank. Got keys={list(subbank.keys())}"
            )

        self.concept_user = selected_user
        self.query_words = list(subbank["query_words"])

        qidx = subbank.get("query_idx", None)
        if qidx is None:
            query_idx_list = list(range(len(self.query_words)))
        else:
            if torch.is_tensor(qidx):
                query_idx_list = qidx.detach().cpu().long().tolist()
            else:
                query_idx_list = list(qidx)

        self.text_bank = subbank["text_outputs"]
        self.text_batch_dim = subbank.get("text_batch_dim", {})

        self.num_queries = len(self.query_words)
        self.num_cls = self.num_queries
        self.register_buffer("query_idx", torch.tensor(query_idx_list, dtype=torch.long))

        bg_pc = subbank.get("bg_thr_per_class", None)
        if bg_pc is not None:
            bg_pc = torch.as_tensor(bg_pc, dtype=torch.float32).view(-1)
            if bg_pc.numel() == self.num_queries:
                if not hasattr(self, "bg_thr_per_class"):
                    self.register_buffer("bg_thr_per_class", bg_pc)
                else:
                    self.bg_thr_per_class = bg_pc

        use_pc = False
        if hasattr(self, "bg_thr_per_class") and torch.is_tensor(self.bg_thr_per_class):
                use_pc = bool((self.bg_thr_per_class > 1e-6).any().item())
        self.use_bg_thr_per_class = use_pc

        print(
            f"[SAM3_OVSS] Loaded {concept_path} | user='{self.concept_user}' | num_queries={self.num_queries}"
        )

    def _make_find_stage(self, B: int) -> FindStage:
        return FindStage(
            img_ids=torch.zeros(B, device=self.device, dtype=torch.long),
            text_ids=torch.arange(B, device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def _get_dummy_prompt(self, B: int):
        try:
            return self.model._get_dummy_prompt(num_prompts=B)
        except TypeError:
            try:
                return self.model._get_dummy_prompt(B)
            except TypeError:
                return self.model._get_dummy_prompt()

    def _slice_text_outputs(self, start: int, end: int) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in self.text_bank.items():
            if not torch.is_tensor(v):
                raise TypeError(f"text_bank[{k}] must be a Tensor, got {type(v)}")
            bd = int(self.text_batch_dim.get(k, 0))
            out[k] = slice_along_dim(v, start, end, dim=bd).to(self.device, non_blocking=True)
        return out

    @torch.no_grad()
    def _inference_single_view(self, image) -> torch.Tensor:
        image = to_pil(image)
        W, H = image.size
        seg_logits = torch.zeros((self.num_queries, H, W), device=self.device, dtype=torch.float16)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with amp_ctx:
            st = self.processor.set_image(image)
            backbone_out = st["backbone_out"]

            for start in range(0, self.num_queries, self.prompt_chunk_size):
                end = min(start + self.prompt_chunk_size, self.num_queries)
                B = end - start
                if B <= 0:
                    continue

                backbone_out.update(self._slice_text_outputs(start, end))

                outputs = self.model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=self._make_find_stage(B),
                    geometric_prompt=self._get_dummy_prompt(B),
                    find_target=None,
                )

                pres = outputs["presence_logit_dec"].sigmoid().view(-1)

                pl = outputs["pred_logits"]
                if pl.ndim == 3 and pl.size(-1) == 1:
                    pl = pl.squeeze(-1)
                scores = pl.sigmoid()

                pm = outputs["pred_masks"]
                if pm.ndim == 5 and pm.size(2) == 1:
                    pm = pm.squeeze(2)

                sem_up = None
                if self.use_sem_seg and ("semantic_seg" in outputs):
                    sem = outputs["semantic_seg"]
                    sem_up = interpolate(sem, (H, W), mode="bilinear", align_corners=False).sigmoid().squeeze(1)

                for b in range(B):
                    qidx = start + b
                    cur = torch.zeros((H, W), device=self.device, dtype=torch.float16)

                    s = scores[b]
                    masks_b = pm[b]

                    if self.topk_inst > 0 and s.numel() > self.topk_inst:
                        val, idx = torch.topk(s, k=self.topk_inst, largest=True)
                        s = val
                        masks_b = masks_b[idx]

                    if self.use_presence_instance:
                        s = s * pres[b]

                    keep = s > self.confidence_threshold
                    if keep.any():
                        masks_k = masks_b[keep]
                        scores_k = s[keep]
                        masks_up = interpolate(
                            masks_k.unsqueeze(1), (H, W),
                            mode="bilinear", align_corners=False
                        ).sigmoid().squeeze(1)
                        inst_map = (masks_up * scores_k.view(-1, 1, 1)).amax(dim=0)
                        cur = torch.max(cur, inst_map)

                    if sem_up is not None:
                        cur = torch.max(cur, sem_up[b] * pres[b])

                    seg_logits[qidx] = cur

        return seg_logits

    def predict(self, inputs, data_samples):
        for i, image in enumerate(inputs):
            seg_logits = self._inference_single_view(image)
            max_scores, seg_pred = torch.max(seg_logits, dim=0)

            if self.use_bg_thr_per_class and hasattr(self, "bg_thr_per_class") and torch.is_tensor(self.bg_thr_per_class):
                idx = seg_pred.long().clamp(0, int(self.bg_thr_per_class.numel()) - 1)
                thr_map = self.bg_thr_per_class[idx].to(max_scores.dtype)
                seg_pred[max_scores < thr_map] = int(self.bg_idx)
            elif self.bg_thr > 0.0:
                seg_pred[max_scores < float(self.bg_thr)] = int(self.bg_idx)

            if self.ig_thr > 0.0:
                seg_pred[max_scores < float(self.ig_thr)] = 255

            data_samples[i].set_data({
                "seg_logits": PixelData(**{"data": seg_logits}),
                "pred_sem_seg": PixelData(**{"data": seg_pred.unsqueeze(0)}),
            })

        return data_samples

    def _forward(self, inputs, data_samples=None):
        raise NotImplementedError

    def extract_feat(self, inputs):
        raise NotImplementedError

    def encode_decode(self, inputs, batch_img_metas):
        raise NotImplementedError

    def loss(self, inputs, data_samples):
        raise NotImplementedError
