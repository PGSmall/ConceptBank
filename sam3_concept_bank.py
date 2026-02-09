import os
import re
import json
import copy
import heapq
import gc
import random
import argparse
import time
from collections import OrderedDict
from contextlib import nullcontext
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist

# =========================================================
# 1. DDP Bootstrap
# =========================================================
def setup_distributed_env():
    """Sets CUDA device based on LOCAL_RANK to prevent multi-process conflicts."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)


setup_distributed_env()

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import mkdir_or_exist
from mmseg.registry import DATASETS

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.data_misc import FindStage, interpolate

try:
    import custom_datasets
except Exception:
    pass


# =========================================================
# 2. DDP Utilities
# =========================================================
def dist_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if dist_is_on() else 0

def get_world_size() -> int:
    return dist.get_world_size() if dist_is_on() else 1

def is_main() -> bool:
    return get_rank() == 0

def print0(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)

def ddp_init(backend: str = "nccl"):
    if "WORLD_SIZE" not in os.environ:
        return
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 1:
        return

    if torch.cuda.is_available() and backend == "nccl":
        try:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                device_id=torch.cuda.current_device(),
            )
        except TypeError:
            dist.init_process_group(backend=backend, init_method="env://")
    else:
        dist.init_process_group(backend=backend if backend else "gloo", init_method="env://")

def ddp_barrier():
    if not dist_is_on():
        return
    try:
        if torch.cuda.is_available() and dist.get_backend() == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
    except Exception:
        pass

def ddp_cleanup():
    if dist_is_on():
        ddp_barrier()
        dist.destroy_process_group()

def _dist_reduce_device(prefer: torch.device) -> torch.device:
    """Use CUDA tensors only for NCCL; otherwise reduce on CPU to avoid backend/device mismatch."""
    if not dist_is_on():
        return prefer
    try:
        backend = dist.get_backend()
    except Exception:
        backend = "gloo"
    if backend == "nccl" and torch.cuda.is_available():
        return prefer if prefer.type == "cuda" else torch.device("cuda")
    return torch.device("cpu")

def _ddp_reduce_max_f64(x: float, prefer_device: torch.device) -> float:
    if not dist_is_on():
        return float(x)
    dev = _dist_reduce_device(prefer_device)
    t = torch.tensor([float(x)], device=dev, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())

def _ddp_reduce_sum_i64(x: int, prefer_device: torch.device) -> int:
    if not dist_is_on():
        return int(x)
    dev = _dist_reduce_device(prefer_device)
    t = torch.tensor([int(x)], device=dev, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())

def _sec_to_min(x: float) -> float:
    return float(x) / 60.0


# =========================================================
# 3. String & Parsing Utilities
# =========================================================
def parse_users_arg(s: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not parts:
        raise RuntimeError("--users is empty. Expect 'name=config.py,...'")
    for p in parts:
        if "=" not in p:
            raise RuntimeError(f"Bad --users item: {p}. Expect name=cfg_path")
        name, cfg = p.split("=", 1)
        name = name.strip()
        cfg = cfg.strip()
        if not name or not cfg:
            raise RuntimeError(f"Bad --users item: {p}.")
        out.append((name, cfg))

    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise RuntimeError(f"Duplicate user names in --users: {names}")
    return out

def _norm_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s_\-]+", "", s)
    return s

def dedup_phrases(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x = x.strip()
        if not x:
            continue
        k = " ".join(x.lower().split())
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

def load_name_path(name_path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    classes: List[str] = []
    expanded: Dict[str, List[str]] = {}

    with open(name_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if not line:
                continue

            key = None
            phrases: List[str] = []

            if "\t" in line:
                parts = [p.strip() for p in line.split("\t") if p.strip()]
                key = parts[0]
                rest = parts[1:]
                if len(rest) == 1 and any(c in rest[0] for c in ",|;"):
                    phrases = [x.strip() for x in re.split(r"[,\|\;]", rest[0]) if x.strip()]
                else:
                    phrases = rest
            elif ":" in line or "->" in line:
                sep = ":" if ":" in line else "->"
                key, rest = line.split(sep, 1)
                key, rest = key.strip(), rest.strip()

                if rest.startswith("[") and rest.endswith("]"):
                    try:
                        arr = json.loads(rest)
                        if isinstance(arr, list):
                            phrases = [str(x).strip() for x in arr if str(x).strip()]
                    except Exception:
                        phrases = []

                if not phrases:
                    phrases = [x.strip() for x in re.split(r"[,\|\;]", rest) if x.strip()]
            elif "," in line:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                key = parts[0]
                phrases = parts[1:]
            else:
                key = line.strip()

            if not key:
                continue

            ph = dedup_phrases([key] + phrases)
            classes.append(key)
            expanded[key] = ph

    if not classes:
        raise RuntimeError(f"name_path parsed empty: {name_path}")
    return classes, expanded

def parse_extra_calib(s: Optional[str], name2id: Dict[str, int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = int(v.strip())
        if k in name2id:
            out[name2id[k]] = v
    return out


# =========================================================
# 4. Dataset & Image Utilities
# =========================================================
def _strip_pack_transforms(pipeline: List[dict]) -> List[dict]:
    banned = {"PackSegInputs", "PackPILInputs", "PackInputs"}
    out = []
    for t in pipeline:
        if isinstance(t, dict) and t.get("type", "") not in banned:
            out.append(t)
    return out

def _override_split_in_data_prefix(dataset_cfg: dict, split: str):
    if not isinstance(dataset_cfg, dict):
        return

    coco_map = {"train": "train2017", "val": "val2017", "test": "test2017"}

    if "data_prefix" in dataset_cfg:
        dp = dataset_cfg["data_prefix"]
        if isinstance(dp, dict):

            def repl(p: str) -> str:
                p = p.replace("\\", "/")
                p = re.sub(r"/(train|val|test)(/)?$", lambda m: f"/{split}" + ("/" if m.group(2) else ""), p)
                p = re.sub(r"/(train|val|test)/", f"/{split}/", p)

                if split in coco_map:
                    tgt = coco_map[split]
                    p = re.sub(r"/(train|val|test)2017(?=/|$)", f"/{tgt}", p)

                if split == "train":
                    p = re.sub(r"/validation(/|$)", r"/training\1", p)
                elif split == "val":
                    p = re.sub(r"/training(/|$)", r"/validation\1", p)
                return p

            for k, v in dp.items():
                if isinstance(v, str):
                    dp[k] = repl(v)

    if "ann_file" in dataset_cfg and isinstance(dataset_cfg["ann_file"], str):
        af = dataset_cfg["ann_file"].replace("\\", "/")
        af = re.sub(r"/(train|val|test)\.txt$", f"/{split}.txt", af)
        af = re.sub(r"/trainaug\.txt$", f"/{split}.txt", af)
        dataset_cfg["ann_file"] = af

    if "split" in dataset_cfg and isinstance(dataset_cfg["split"], str):
        dataset_cfg["split"] = split

def _extract_img_and_gt(sample: dict) -> Tuple[Image.Image, np.ndarray]:
    img = sample.get("img", sample.get("inputs", None))
    gt = sample.get("gt_seg_map", sample.get("gt_semantic_seg", None))
    if img is None or gt is None:
        raise RuntimeError(f"Cannot extract img/gt from sample keys={list(sample.keys())}")

    if isinstance(img, Image.Image):
        pil = img.convert("RGB")
    else:
        arr = img
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[2] == 3:
            pil = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        elif arr.ndim == 2:
            pil = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        else:
            raise RuntimeError(f"Unsupported img shape: {arr.shape}")

    if torch.is_tensor(gt):
        gt_np = gt.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt)
    gt_np = gt_np.astype(np.int32)
    return pil, gt_np

def _square_box_from_mask(train_ids: np.ndarray, class_id: int, pad_ratio: float, min_size: int):
    ys, xs = np.where(train_ids == class_id)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    w, h = x2 - x1 + 1, y2 - y1 + 1
    pad = int(max(w, h) * pad_ratio)

    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(train_ids.shape[1] - 1, x2 + pad)
    y2p = min(train_ids.shape[0] - 1, y2 + pad)

    cx, cy = (x1p + x2p) // 2, (y1p + y2p) // 2
    half = max((x2p - x1p + 1), (y2p - y1p + 1), min_size) // 2

    x1p, x2p = max(0, cx - half), min(train_ids.shape[1] - 1, cx + half)
    y1p, y2p = max(0, cy - half), min(train_ids.shape[0] - 1, cy + half)
    return x1p, y1p, x2p, y2p

def make_crop_views_from_np(
    img_np: np.ndarray,
    train_ids: np.ndarray,
    class_id: int,
    pad_ratio: float,
    min_size: int,
    use_context: bool,
    use_masked: bool,
):
    if not (use_context or use_masked):
        use_context = True

    box = _square_box_from_mask(train_ids, class_id, pad_ratio, min_size)
    if box is None:
        return None, None

    x1, y1, x2, y2 = box
    crop_np = img_np[y1 : y2 + 1, x1 : x2 + 1].copy()
    m = (train_ids[y1 : y2 + 1, x1 : x2 + 1] == class_id).astype(np.uint8)

    views: List[Image.Image] = []
    ctx_view = Image.fromarray(crop_np)
    if use_context:
        views.append(ctx_view)
    if use_masked:
        bg = np.full_like(crop_np, 127, dtype=np.uint8)
        masked = np.where(m[..., None].astype(bool), crop_np, bg)
        views.append(Image.fromarray(masked))

    if not views:
        views.append(ctx_view)
    return views, m


# =========================================================
# 5. Model & Tensor Alignment Utilities
# =========================================================
def _find_dim_of_size(shape: torch.Size, size: int) -> Optional[int]:
    for i, s in enumerate(shape):
        if s == size:
            return i
    return None

def _align_3d_to_BTD(x: torch.Tensor, B: int) -> Tuple[torch.Tensor, dict]:
    assert x.ndim == 3
    bd = _find_dim_of_size(x.shape, B)
    if bd is None:
        raise RuntimeError(f"Cannot find batch dim B={B} in {tuple(x.shape)}")
    rem = [0, 1, 2]
    rem.remove(bd)
    d_dim = rem[0] if x.shape[rem[0]] >= x.shape[rem[1]] else rem[1]
    t_dim = rem[1] if d_dim == rem[0] else rem[0]
    x_btd = x.permute(bd, t_dim, d_dim).contiguous()
    layout = {"raw_shape": tuple(x.shape), "batch_dim": bd, "token_dim": t_dim, "d_dim": d_dim}
    return x_btd, layout

def _align_mask_to_BT(x: torch.Tensor, B: int) -> Tuple[torch.Tensor, dict]:
    x = torch.as_tensor(x).squeeze()
    if x.ndim == 1:
        if B != 1:
            raise RuntimeError(f"language_mask is 1D {tuple(x.shape)} but B={B} != 1")
        return x.view(1, -1).contiguous(), {"raw_shape": tuple(x.shape), "batch_dim": 0, "token_dim": 1}

    if x.ndim != 2:
        raise RuntimeError(f"language_mask must be 1D or 2D after squeeze, got {tuple(x.shape)}")

    bd = _find_dim_of_size(x.shape, B)
    if bd is None:
        if B == 1:
            if x.shape[0] == 1:
                return x.contiguous(), {"raw_shape": tuple(x.shape), "batch_dim": 0, "token_dim": 1}
            if x.shape[1] == 1:
                return x.t().contiguous(), {"raw_shape": tuple(x.shape), "batch_dim": 0, "token_dim": 1}
        raise RuntimeError(f"Cannot find batch dim B={B} in mask {tuple(x.shape)}")

    t_dim = 1 - bd
    return x.permute(bd, t_dim).contiguous(), {"raw_shape": tuple(x.shape), "batch_dim": bd, "token_dim": t_dim}

def _align_language_outputs(out: Dict[str, Any], B: int):
    lf_btd, lf_layout = _align_3d_to_BTD(out["language_features"], B)
    pm_bt, pm_layout = _align_mask_to_BT(out["language_mask"], B)
    le_btd, le_layout = _align_3d_to_BTD(out["language_embeds"], B)
    layout = {"language_features": lf_layout, "language_mask": pm_layout, "language_embeds": le_layout}
    return lf_btd, pm_bt, le_btd, layout

def to_raw_from_BTD(x_btd: torch.Tensor, lay: dict) -> torch.Tensor:
    bd, td, dd = lay["batch_dim"], lay["token_dim"], lay["d_dim"]
    src = [0, 0, 0]
    src[bd], src[td], src[dd] = 0, 1, 2
    return x_btd.permute(*src).contiguous()

def mask_to_raw_from_BT(x_bt: torch.Tensor, lay: dict) -> torch.Tensor:
    bd = lay["batch_dim"]
    src = [0, 0]
    src[bd], src[1 - bd] = 0, 1
    return x_bt.permute(*src).contiguous()

def fuse_tokens(selected: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]):
    assert len(selected) >= 1
    T = selected[0][0].shape[0]
    w = torch.tensor([s[3] for s in selected], dtype=torch.float32)
    w = w / w.sum().clamp_min(1e-6)

    lf = torch.stack([s[0].float() for s in selected], dim=0)
    pm = torch.stack([s[1].bool() for s in selected], dim=0)
    le = torch.stack([s[2].float() for s in selected], dim=0)

    valid = (~pm).float()
    denom = (w.view(-1, 1) * valid).sum(dim=0)
    valid_f = denom > 1e-6
    pm_f = ~valid_f
    denom_safe = denom.clamp_min(1e-6).view(1, T, 1)

    lf_num = (w.view(-1, 1, 1) * lf * valid.unsqueeze(-1)).sum(dim=0, keepdim=True)
    le_num = (w.view(-1, 1, 1) * le * valid.unsqueeze(-1)).sum(dim=0, keepdim=True)

    return (lf_num / denom_safe).squeeze(0), pm_f, (le_num / denom_safe).squeeze(0)


# =========================================================
# 6. SAM3 Specific Utilities
# =========================================================
@torch.inference_mode()
def build_prob_map(
    outputs: dict,
    H: int,
    W: int,
    confidence_threshold: float,
    topk_inst: int,
    use_sem: bool = True,
    use_presence_instance: bool = True,
):
    pres = outputs["presence_logit_dec"].sigmoid().view(-1)  # [B]
    pl = outputs["pred_logits"]
    if pl.ndim == 3 and pl.size(-1) == 1:
        pl = pl.squeeze(-1)
    scores = pl.sigmoid()  # [B,N]

    pm = outputs["pred_masks"]
    if pm.ndim == 5 and pm.size(2) == 1:
        pm = pm.squeeze(2)  # [B,N,hf,wf]

    sem_up = None
    if use_sem and ("semantic_seg" in outputs):
        sem = outputs["semantic_seg"]  # [B,1,hf,wf]
        sem_up = interpolate(sem, (H, W), mode="bilinear", align_corners=False).sigmoid().squeeze(1)

    B = scores.shape[0]
    out_maps = []
    for b in range(B):
        s = scores[b]
        masks_b = pm[b]

        if topk_inst > 0 and s.numel() > topk_inst:
            val, idx = torch.topk(s, k=topk_inst, largest=True)
            s = val
            masks_b = masks_b[idx]

        if use_presence_instance:
            s = s * pres[b]

        keep = s > confidence_threshold
        inst_map = torch.zeros((H, W), device=masks_b.device, dtype=torch.float16)

        if keep.any():
            masks_k = masks_b[keep]
            scores_k = s[keep]
            masks_up = interpolate(
                masks_k.unsqueeze(1), (H, W), mode="bilinear", align_corners=False
            ).sigmoid().squeeze(1)
            inst_map = (masks_up * scores_k.view(-1, 1, 1)).amax(dim=0)

        if sem_up is not None:
            sem_val = sem_up[b] * pres[b]
            inst_map = torch.max(inst_map, sem_val)

        out_maps.append(inst_map)
    return torch.stack(out_maps, dim=0)

@torch.inference_mode()
def extract_sam3_image_embedding(backbone_out: Dict[str, Any]) -> torch.Tensor:
    preferred_keys = [
        "image_embedding", "image_embed", "img_embedding", "img_embed",
        "vision_embedding", "vision_embed", "image_features", "vision_features",
        "backbone_features", "features", "feat", "x",
    ]

    def _iter_tensors(d: Any, prefix: str = ""):
        if isinstance(d, dict):
            for k, v in d.items():
                p = f"{prefix}.{k}" if prefix else str(k)
                yield from _iter_tensors(v, p)
        elif torch.is_tensor(d):
            yield prefix, d

    def _pick_tensor():
        for k in preferred_keys:
            if k in backbone_out and torch.is_tensor(backbone_out[k]):
                return backbone_out[k]

        candidates = []
        for name, t in _iter_tensors(backbone_out):
            if t.dtype in (torch.float16, torch.float32, torch.bfloat16):
                lname = name.lower()
                pri = 0
                for i, kk in enumerate(preferred_keys):
                    if kk in lname:
                        pri = max(pri, 100 - i)
                last_dim = int(t.shape[-1]) if t.ndim >= 2 else int(t.numel())
                score = pri * 1_000_000 + last_dim * 1000 + t.ndim * 10
                candidates.append((score, t))

        if not candidates:
            # robust fallback: create a dummy embedding
            return torch.zeros((1, 1), dtype=torch.float32)
        return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]

    t = _pick_tensor()

    if t.ndim == 4:
        if t.size(0) == 1:
            t = t[0]
        if t.ndim == 3:
            t = t.mean(dim=(1, 2))
        else:
            t = t.mean(dim=(0, 2, 3))
    elif t.ndim >= 2:
        t = t.reshape(-1, t.shape[-1]).mean(dim=0)
    else:
        t = t.flatten()

    emb = t.float()
    return emb / emb.norm(p=2).clamp_min(1e-6)


# =========================================================
# 7. Caching & Auto Config
# =========================================================
def auto_cap_from_num_classes(C: int, lo: int = 3, hi: int = 9) -> int:
    cap = int(round(np.sqrt(max(C, 1)) / 5.0 + 2.0))
    return int(np.clip(cap, lo, hi))

def _stage_cleanup_cuda(empty_cache: bool = True):
    gc.collect()
    if torch.cuda.is_available() and empty_cache:
        torch.cuda.empty_cache()

class LRUEmbCache:
    def __init__(self, max_items: int = 8192):
        self.max_items = int(max_items)
        self.od: OrderedDict = OrderedDict()

    def get(self, key):
        v = self.od.get(key, None)
        if v is not None:
            self.od.move_to_end(key)
        return v

    def put(self, key, value):
        if self.max_items <= 0:
            return
        self.od[key] = value
        self.od.move_to_end(key)
        if len(self.od) > self.max_items:
            self.od.popitem(last=False)


# =========================================================
# 8. Main Builder (One User)
# =========================================================
@torch.inference_mode()
def build_one_user_bank(
    user_name: str,
    cfg_path: str,
    args,
    model,
    processor,
    device: torch.device,
    amp_ctx,
    rank: int,
    world: int,
    global_text_layout_cache: dict,
) -> Tuple[Optional[dict], dict, dict]:
    init_default_scope("mmseg")

    prof: Dict[str, Any] = {
        "stage1_sec": 0.0, "stage2_sec": 0.0, "stage3_sec": 0.0,
        "stage1_crops": 0, "stage2_crops": 0, "stage3_crops": 0,
        "total_sec": 0.0,
    }

    # --- Config & Dataset Init ---
    cfg = Config.fromfile(cfg_path)
    if "model" not in cfg or "name_path" not in cfg.model:
        raise RuntimeError(f"[{user_name}] config must contain cfg.model.name_path")
    name_path = cfg.model.name_path

    try:
        bg_idx = int(getattr(cfg.model, "bg_idx", 0))
    except Exception:
        bg_idx = 0

    try:
        bg_thr = float(getattr(cfg.model, "bg_thr", 0))
    except Exception:
        bg_thr = 0.0

    try:
        calib_per_class = int(getattr(cfg.model, "calib_per_class", 10))
    except Exception:
        calib_per_class = 10

    try:
        calib_max_class = int(getattr(cfg.model, "calib_max_class", 10))
    except Exception:
        calib_max_class = 10

    classes, expanded = load_name_path(name_path)
    C = len(classes)
    name2qid = {n: i for i, n in enumerate(classes)}
    name2qid_norm = {_norm_name(n): i for i, n in enumerate(classes)}

    dl_key = args.dataloader_key if args.dataloader_key else ("train_dataloader" if "train_dataloader" in cfg else "test_dataloader")
    if dl_key not in cfg:
        raise RuntimeError(f"[{user_name}] {dl_key} not found.")

    dataset_cfg = copy.deepcopy(cfg[dl_key]["dataset"])
    if "pipeline" in dataset_cfg and isinstance(dataset_cfg["pipeline"], list):
        dataset_cfg["pipeline"] = _strip_pack_transforms(dataset_cfg["pipeline"])

    if args.split:
        _override_split_in_data_prefix(dataset_cfg, args.split)
    if args.img_path:
        dataset_cfg.setdefault("data_prefix", {})["img_path"] = args.img_path
    if args.seg_path:
        dataset_cfg.setdefault("data_prefix", {})["seg_map_path"] = args.seg_path

    dataset = DATASETS.build(dataset_cfg)

    ds_classes = getattr(dataset, "CLASSES", getattr(dataset, "metainfo", {}).get("classes", None))
    if ds_classes is None:
        ds_classes = classes
    ds_classes = list(ds_classes)

    dsid_to_qid: Dict[int, int] = {}
    for i, n in enumerate(ds_classes):
        k = _norm_name(n)
        if k in name2qid_norm:
            dsid_to_qid[i] = name2qid_norm[k]
    for i in range(len(ds_classes)):
        if i not in dsid_to_qid and i < C:
            dsid_to_qid[i] = i

    ignore_index = getattr(dataset, "ignore_index", 255)
    try:
        ignore_index = int(ignore_index)
    except Exception:
        ignore_index = 255

    N = len(dataset)
    indices_shard = list(range(rank, N, world))
    disable_tqdm = not is_main()

    if is_main():
        print0(f"\n========== [USER] {user_name} ==========")
        print0(f"[INFO] cfg={cfg_path}, name_path={name_path}, C={C}, N={N}")

    extra = parse_extra_calib(args.extra_calib, name2qid)
    target = [calib_per_class + extra.get(i, 0) for i in range(C)]

    cap_pass1 = auto_cap_from_num_classes(C, lo=3, hi=9)
    cap_pass2_init = auto_cap_from_num_classes(C, lo=3, hi=9)
    proto_per_class = calib_per_class

    if is_main():
        print0(f"[AUTO] proto={proto_per_class}, cap1={cap_pass1}, cap2={cap_pass2_init}, cache={args.pass2_emb_cache}")

    def _view_weights(views):
        if args.use_context_view and args.use_masked_view and len(views) == 2:
            w = [args.view_weight_context, args.view_weight_masked]
        else:
            w = [1.0] * len(views)
        s = sum(w)
        return [x / max(1e-6, s) for x in w]

    @torch.inference_mode()
    def compute_crop_embedding_device(views):
        ws = _view_weights(views)
        emb_acc = None
        for v, wv in zip(views, ws):
            with amp_ctx:
                st = processor.set_image(v)
                e = extract_sam3_image_embedding(st["backbone_out"])
            e = e.float() * float(wv)
            emb_acc = e if emb_acc is None else (emb_acc + e)
        return emb_acc / emb_acc.norm(p=2).clamp_min(1e-6)

    # =====================================================
    # Stage I: Pass 1 Prototype Collection
    # =====================================================
    _stage_cleanup_cuda()
    t_s1_start = time.perf_counter()

    D_local = 0
    sum_local = None
    cnt_local_cpu = np.zeros((C,), dtype=np.int32)
    cnt_limit_cpu = np.full((C,), calib_max_class, dtype=np.int32)

    it1 = tqdm(indices_shard, desc=f"{user_name}:Pass1(r{rank})", dynamic_ncols=True, disable=disable_tqdm)

    for idx in it1:
        if (cnt_local_cpu >= cnt_limit_cpu).all():
            break
        try:
            sample = dataset[idx]
            img, gt_trainid = _extract_img_and_gt(sample)
        except Exception:
            continue

        img_np = np.asarray(img, dtype=np.uint8)
        uniq = [int(x) for x in np.unique(gt_trainid) if int(x) != ignore_index and int(x) >= 0]
        if not uniq:
            continue

        uniq_q = []
        for dsid in uniq:
            qid = dsid_to_qid.get(dsid, None)
            if qid is not None and 0 <= qid < C:
                uniq_q.append((qid, dsid))

        uniq_q.sort(key=lambda p: (int(cnt_local_cpu[p[0]]), p[0]))

        for (qid, dsid) in uniq_q[:cap_pass1]:
            if cnt_local_cpu[qid] >= cnt_limit_cpu[qid]:
                continue

            views, _ = make_crop_views_from_np(
                img_np, gt_trainid, dsid,
                pad_ratio=args.pad_ratio, min_size=args.min_crop_size,
                use_context=args.use_context_view, use_masked=args.use_masked_view,
            )
            if views is None:
                continue

            try:
                e = compute_crop_embedding_device(views)
            except Exception:
                continue

            if D_local == 0:
                D_local = int(e.numel())
                if D_local <= 0:
                    continue
                sum_local = torch.zeros((C, D_local), device=device, dtype=torch.float32)

            if int(e.numel()) == D_local:
                sum_local[qid] += e.float()
                cnt_local_cpu[qid] += 1

    # Sync/Resolve embedding dim robustly
    if dist_is_on():
        dev_red = _dist_reduce_device(device)
        d_tensor = torch.tensor([int(D_local)], device=dev_red, dtype=torch.int64)
        d_list = [torch.zeros_like(d_tensor) for _ in range(world)]
        dist.all_gather(d_list, d_tensor)
        ds = sorted(set([int(x.item()) for x in d_list if int(x.item()) > 0]))
        D_global = ds[0] if ds else 0
    else:
        D_global = int(D_local)

    if D_global <= 0:
        # robust fallback: no valid protos
        proto_valid_cpu = np.zeros((C,), dtype=bool)
        cnt_local = torch.zeros((C,), device=device, dtype=torch.float32)
        if is_main():
            print0(f"[{user_name}] Pass1: no valid crops for prototype collection; continue with fallback.")
        prof["stage1_crops"] = 0
        prof["stage1_sec"] = _ddp_reduce_max_f64(time.perf_counter() - t_s1_start, device)
        ddp_barrier()
        # create dummy proto tensor for shape safety
        proto = torch.zeros((C, 1), device=device, dtype=torch.float32)
    else:
        if sum_local is None:
            sum_local = torch.zeros((C, D_global), device=device, dtype=torch.float32)

        cnt_local = torch.from_numpy(cnt_local_cpu.astype(np.float32)).to(device)
        if dist_is_on():
            dist.all_reduce(sum_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_local, op=dist.ReduceOp.SUM)

        proto = sum_local / cnt_local.clamp_min(1.0).unsqueeze(1)
        proto = proto / proto.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        proto_valid_cpu = (cnt_local.detach().cpu().numpy() > 0.5)

        if is_main():
            print0(f"[{user_name}] Pass1 Done. Valid protos: {int(proto_valid_cpu.sum())}/{C}")

        prof["stage1_crops"] = int(cnt_local.sum().item())

        ddp_barrier()
        prof["stage1_sec"] = _ddp_reduce_max_f64(time.perf_counter() - t_s1_start, device)

    # =====================================================
    # Stage II: Pass 2 Multi-epoch Fill (Representative Mining)
    # =====================================================
    t_s2_start = time.perf_counter()

    heaps = [[] for _ in range(C)]
    tie = 0
    emb_cache = LRUEmbCache(max_items=int(args.pass2_emb_cache))

    def heap_size(c):
        return len(heaps[c])

    def all_full():
        for c in range(C):
            if target[c] > 0 and heap_size(c) < target[c] and proto_valid_cpu[c]:
                return False
        return True

    if D_global > 0 and bool(proto_valid_cpu.any()):
        for epoch in range(args.pass2_max_epochs):
            if all_full():
                break
            cap_pass2 = int(min(12, cap_pass2_init * (2 ** epoch)))
            it2 = tqdm(indices_shard, desc=f"{user_name}:Pass2(e{epoch},r{rank})", dynamic_ncols=True, disable=disable_tqdm)

            for idx in it2:
                if all_full():
                    break
                try:
                    sample = dataset[idx]
                    img, gt_trainid = _extract_img_and_gt(sample)
                except Exception:
                    continue

                img_np = np.asarray(img, dtype=np.uint8)
                uniq = [int(x) for x in np.unique(gt_trainid) if int(x) != ignore_index and int(x) >= 0]
                if not uniq:
                    continue

                cand = []
                for dsid in uniq:
                    qid = dsid_to_qid.get(dsid, None)
                    if qid is None or not (0 <= qid < C):
                        continue
                    if target[qid] <= 0 or not proto_valid_cpu[qid]:
                        continue
                    if heap_size(qid) < target[qid]:
                        cand.append((target[qid] - heap_size(qid), qid, dsid))

                if not cand:
                    continue
                cand.sort(key=lambda x: (-x[0], x[1], x[2]))

                qids, dsids, embs = [], [], []

                for (_, qid, dsid) in cand[:cap_pass2]:
                    if heap_size(qid) >= target[qid]:
                        continue

                    key = (
                        int(idx),
                        int(dsid),
                        int(args.use_context_view),
                        int(args.use_masked_view),
                        float(args.pad_ratio),
                        int(args.min_crop_size),
                    )

                    cached = emb_cache.get(key)
                    if cached is not None:
                        e = cached.to(device, dtype=torch.float32, non_blocking=True)
                    else:
                        views, _ = make_crop_views_from_np(
                            img_np, gt_trainid, dsid,
                            pad_ratio=args.pad_ratio, min_size=args.min_crop_size,
                            use_context=args.use_context_view, use_masked=args.use_masked_view,
                        )
                        if views is None:
                            continue
                        try:
                            e = compute_crop_embedding_device(views)
                        except Exception:
                            continue
                        emb_cache.put(key, e.detach().cpu().half())

                    if int(e.numel()) == D_global:
                        qids.append(int(qid))
                        dsids.append(int(dsid))
                        embs.append(e)

                if not embs:
                    continue

                E = torch.stack(embs, dim=0).float()
                P = proto[torch.tensor(qids, device=device)].float()
                scores = (E * P).sum(dim=1).detach().cpu().tolist()

                for qid, dsid, sc in zip(qids, dsids, scores):
                    tie += 1
                    item = (float(sc), tie, int(idx), int(dsid))
                    h = heaps[qid]
                    if len(h) < target[qid]:
                        heapq.heappush(h, item)
                    elif float(sc) > h[0][0]:
                        heapq.heapreplace(h, item)

    local_cands = [
        [(float(s), int(i), int(dsid)) for (s, _, i, dsid) in sorted(heaps[c], key=lambda x: x[0], reverse=True)]
        for c in range(C)
    ]

    if dist_is_on():
        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, local_cands)
    else:
        gathered = [local_cands]

    selected_meta = None
    if is_main():
        selected_meta = [[] for _ in range(C)]
        for c in range(C):
            all_items = []
            for r in range(len(gathered)):
                all_items.extend(gathered[r][c])
            all_items.sort(key=lambda x: x[0], reverse=True)
            K = target[c]
            top = all_items[:K] if K > 0 else []
            selected_meta[c] = [(idx, dsid, float(score)) for (score, idx, dsid) in top]

    if dist_is_on():
        obj_list = [selected_meta]
        dist.broadcast_object_list(obj_list, src=0)
        selected_meta = obj_list[0]

    ddp_barrier()

    # Stage II crops = selected representative supports (global)
    try:
        prof["stage2_crops"] = int(sum(len(selected_meta[c]) for c in range(C)))
    except Exception:
        prof["stage2_crops"] = 0

    # Clean up Pass 2 Memory
    try:
        del sum_local, cnt_local
    except Exception:
        pass
    _stage_cleanup_cuda()
    ddp_barrier()

    prof["stage2_sec"] = _ddp_reduce_max_f64(time.perf_counter() - t_s2_start, device)

    # =====================================================
    # Stage III: Text Prompt Encoding + Scoring/Fusion (+ optional bg thr)
    # =====================================================
    t_s3_start = time.perf_counter()
    stage3_crops_local = 0

    phrases_per_class = [dedup_phrases(expanded.get(n, [n])) for n in classes]

    # Cache layout once
    if not global_text_layout_cache:
        out0 = model.backbone.forward_text(phrases_per_class[0], device=device)
        lf0, pm0, le0, layout = _align_language_outputs(out0, B=len(phrases_per_class[0]))
        global_text_layout_cache.update({
            "layout": layout,
            "T": lf0.shape[1], "Df": lf0.shape[2], "De": le0.shape[2],
        })
    layout = global_text_layout_cache["layout"]

    text_db = []
    for c in range(C):
        ph = phrases_per_class[c]
        out = model.backbone.forward_text(ph, device=device)
        lf, pm, le, _ = _align_language_outputs(out, B=len(ph))
        text_db.append([(lf[i].detach().cpu(), pm[i].detach().cpu(), le[i].detach().cpu()) for i in range(len(ph))])

    T, Df, De = global_text_layout_cache["T"], global_text_layout_cache["Df"], global_text_layout_cache["De"]
    fused_lf = torch.zeros((C, T, Df), dtype=torch.float32, device=device)
    fused_pm_u8 = torch.zeros((C, T), dtype=torch.uint8, device=device)
    fused_le = torch.zeros((C, T, De), dtype=torch.float32, device=device)

    local_chosen: Dict[int, Any] = {}
    local_bg_thr: Dict[int, float] = {}

    def _rand_subsample_1d(x: torch.Tensor, k: int) -> torch.Tensor:
        x = x.flatten()
        if k <= 0 or x.numel() <= k:
            return x
        idx = torch.randperm(x.numel())[:k]
        return x[idx].contiguous()

    def _dice_from_prob(prob_hw: torch.Tensor, gt_mask: torch.Tensor, thr: float) -> float:
        pred = prob_hw >= float(thr)
        gt = gt_mask.bool()
        tp = torch.logical_and(pred, gt).sum().item()
        fp = torch.logical_and(pred, ~gt).sum().item()
        fn = torch.logical_and(~pred, gt).sum().item()
        denom = (2.0 * tp + fp + fn)
        return float((2.0 * tp) / (denom + 1e-6))

    @torch.inference_mode()
    def _ctx_soft_prob_map(
        ctx_view: Image.Image,
        H: int,
        W: int,
        lf_raw: torch.Tensor,
        pm_raw: torch.Tensor,
        le_raw: torch.Tensor,
    ) -> torch.Tensor:
        find_stage_1 = FindStage(
            img_ids=torch.zeros(1, device=device, dtype=torch.long),
            text_ids=torch.zeros(1, device=device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        try:
            geo_prompt_1 = model._get_dummy_prompt(num_prompts=1)
        except TypeError:
            try:
                geo_prompt_1 = model._get_dummy_prompt(1)
            except TypeError:
                geo_prompt_1 = model._get_dummy_prompt()

        with amp_ctx:
            st = processor.set_image(ctx_view)
            st["backbone_out"].update({
                "language_features": lf_raw,
                "language_mask": pm_raw,
                "language_embeds": le_raw,
            })
            outg = model.forward_grounding(
                backbone_out=st["backbone_out"],
                find_input=find_stage_1,
                geometric_prompt=geo_prompt_1,
                find_target=None,
            )
            prob = build_prob_map(
                outg, H, W,
                confidence_threshold=0.0,
                topk_inst=0,
                use_sem=True,
                use_presence_instance=True,
            )
        return prob[0].detach().cpu().float()

    @torch.inference_mode()
    def _estimate_bg_thr_for_class_dice(
        crops_c: List[Tuple[List[Image.Image], Image.Image, torch.Tensor]],
        lf_raw: torch.Tensor,
        pm_raw: torch.Tensor,
        le_raw: torch.Tensor,
    ) -> float:
        mode = str(getattr(args, "bg_thr_mode", "dice")).lower()
        if mode == "none":
            return 0.0
        if not crops_c:
            return 0.0

        crops_use = crops_c[:16]

        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for (_, ctx_view, gt_mask_cpu) in crops_use:
            gt = gt_mask_cpu.detach().cpu()
            if gt.ndim != 2:
                gt = gt.squeeze()
            H, W = int(gt.shape[-2]), int(gt.shape[-1])
            prob_hw = _ctx_soft_prob_map(ctx_view, H, W, lf_raw, pm_raw, le_raw)
            if prob_hw.shape != gt.shape:
                prob_hw = interpolate(
                    prob_hw.unsqueeze(0).unsqueeze(0), (H, W),
                    mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0).cpu().float()
            pairs.append((prob_hw, gt > 0.5))

        if not pairs:
            return 0.0

        score_samples = []
        for (p, _) in pairs:
            score_samples.append(_rand_subsample_1d(p, 50000))
        samp = torch.cat(score_samples, dim=0)
        if samp.numel() == 0:
            return 0.0

        q_grid = torch.linspace(0.02, 0.98, steps=64)
        try:
            cand = torch.quantile(samp, q_grid).clamp(0.0, 1.0)
        except Exception:
            cand = torch.linspace(float(samp.min().item()), float(samp.max().item()), steps=64).clamp(0.0, 1.0)

        cand = torch.cat([cand, torch.tensor([0.0, 1.0])]).clamp(0.0, 1.0)
        cand = torch.unique(cand, sorted=True)

        best_thr = 0.0
        best_score = -1.0
        for t in cand.tolist():
            s = 0.0
            for (p, gt) in pairs:
                s += _dice_from_prob(p, gt, t)
            s /= float(len(pairs))
            if (s > best_score + 1e-6) or (abs(s - best_score) <= 1e-6 and t > best_thr):
                best_score = s
                best_thr = float(t)

        return float(max(bg_thr, min(0.5 + bg_thr, best_thr)))

    class_ids_this_rank = [c for c in range(C) if (c % world) == rank]

    @torch.inference_mode()
    def run_batch_on_views(views, H, W, lf, pm, le, find_stage, geo_prompt):
        weights = _view_weights(views)
        probs_acc = None
        for v, wv in zip(views, weights):
            with amp_ctx:
                st = processor.set_image(v)
                st["backbone_out"].update({"language_features": lf, "language_mask": pm, "language_embeds": le})
                outg = model.forward_grounding(
                    backbone_out=st["backbone_out"],
                    find_input=find_stage,
                    geometric_prompt=geo_prompt,
                    find_target=None,
                )
                prob = build_prob_map(outg, H, W, args.confidence_threshold, args.topk_inst)
            probs_acc = prob * float(wv) if probs_acc is None else (probs_acc + prob * float(wv))
        return probs_acc

    @torch.inference_mode()
    def _score_all_prompts_for_class(views, gt_mask, lf, pm, le, B):
        find_stage = FindStage(
            img_ids=torch.zeros(B, device=device, dtype=torch.long),
            text_ids=torch.arange(B, device=device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        try:
            geo_prompt = model._get_dummy_prompt(num_prompts=B)
        except TypeError:
            geo_prompt = model._get_dummy_prompt(B)

        H, W = int(gt_mask.shape[-2]), int(gt_mask.shape[-1])
        probs = run_batch_on_views(views, H, W, lf, pm, le, find_stage, geo_prompt)

        probs_f = probs.float()
        gt_sum = gt_mask.sum().clamp_min(1e-6)
        inter = (probs_f * gt_mask.unsqueeze(0)).sum(dim=(1, 2))
        denom = probs_f.sum(dim=(1, 2)) + gt_sum + 1e-6
        return (2 * inter + 1e-6) / denom

    @torch.inference_mode()
    def _score_prompts_auto_fallback(views, gt_mask, lf_cpu, pm_cpu, le_cpu):
        M = int(lf_cpu.shape[0])
        try:
            lf = to_raw_from_BTD(lf_cpu.to(device).half(), layout["language_features"])
            pm = mask_to_raw_from_BT(pm_cpu.to(device), layout["language_mask"]).bool()
            le = to_raw_from_BTD(le_cpu.to(device).half(), layout["language_embeds"])
            return _score_all_prompts_for_class(views, gt_mask, lf, pm, le, B=M)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                _stage_cleanup_cuda()
                chunk = 32 if M > 32 else M
                out = torch.zeros((M,), dtype=torch.float32, device=device)
                for s0 in range(0, M, chunk):
                    s1 = min(s0 + chunk, M)
                    sub_lf = to_raw_from_BTD(lf_cpu[s0:s1].to(device).half(), layout["language_features"])
                    sub_pm = mask_to_raw_from_BT(pm_cpu[s0:s1].to(device), layout["language_mask"]).bool()
                    sub_le = to_raw_from_BTD(le_cpu[s0:s1].to(device).half(), layout["language_embeds"])
                    out[s0:s1] = _score_all_prompts_for_class(views, gt_mask, sub_lf, sub_pm, sub_le, B=s1 - s0)
                return out
            # robust fallback: return zeros
            return torch.zeros((M,), dtype=torch.float32, device=device)

    itc = tqdm(class_ids_this_rank, desc=f"{user_name}:Select(r{rank})", dynamic_ncols=True, disable=disable_tqdm)

    for c in itc:
        ph, items = phrases_per_class[c], text_db[c]

        # Build crops for this class from selected_meta (may be empty)
        crops_c: List[Tuple[List[Image.Image], Image.Image, torch.Tensor]] = []
        try:
            sel_list = selected_meta[c] if selected_meta is not None else []
        except Exception:
            sel_list = []

        for (idx, dsid, _) in sel_list:
            try:
                sample = dataset[idx]
                img, gt = _extract_img_and_gt(sample)
            except Exception:
                continue

            img_np0 = np.asarray(img, dtype=np.uint8)

            box = _square_box_from_mask(gt, dsid, args.pad_ratio, args.min_crop_size)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            crop_np = img_np0[y1 : y2 + 1, x1 : x2 + 1].copy()
            m_np = (gt[y1 : y2 + 1, x1 : x2 + 1] == dsid).astype(np.uint8)

            ctx_view = Image.fromarray(crop_np)
            views_sel: List[Image.Image] = []
            if args.use_context_view:
                views_sel.append(ctx_view)
            if args.use_masked_view:
                bg = np.full_like(crop_np, 127, dtype=np.uint8)
                masked = np.where(m_np[..., None].astype(bool), crop_np, bg)
                views_sel.append(Image.fromarray(masked))

            if not views_sel:
                views_sel = [ctx_view]

            crops_c.append((views_sel, ctx_view, torch.from_numpy(m_np.astype(np.float32))))

        # candidate prompts to consider
        keep = list(range(min(len(ph), args.cand_topk)))
        if not keep:
            # nothing to fuse; skip safely
            local_chosen[c] = {"class": classes[c], "chosen": [], "num_reps": 0}
            local_bg_thr[c] = 0.0
            continue

        # Stage III crops: count only valid crops actually scored (crops_c)
        stage3_crops_local += int(len(crops_c))

        lf_kc = torch.stack([items[i][0] for i in keep]).float()
        pm_kc = torch.stack([items[i][1] for i in keep])
        le_kc = torch.stack([items[i][2] for i in keep]).float()

        # If no crops, fall back to uniform fusion over kept prompts
        if len(crops_c) == 0:
            w = [1.0 / float(len(keep))] * len(keep)
            selected_tensors = []
            chosen_list = []
            for wi, ii in zip(w, keep):
                selected_tensors.append((items[ii][0], items[ii][1], items[ii][2], float(wi)))
                chosen_list.append((ph[ii], float(wi), 0.0))
            lf_f, pm_f, le_f = fuse_tokens(selected_tensors)

            fused_lf[c] = lf_f.to(device)
            fused_pm_u8[c] = pm_f.to(device).to(torch.uint8)
            fused_le[c] = le_f.to(device)

            local_bg_thr[c] = 0.0 if c == bg_idx else float(bg_thr)
            local_chosen[c] = {"class": classes[c], "chosen": chosen_list, "num_reps": 0}

            if torch.cuda.is_available() and (c % 8 == 0):
                _stage_cleanup_cuda()
            continue

        pos_all = torch.zeros((len(keep),), dtype=torch.float32, device=device)
        npos = 0

        for (views_sel, _ctx_view, gt_mask_cpu) in crops_c:
            dice = _score_prompts_auto_fallback(views_sel, gt_mask_cpu.to(device), lf_kc, pm_kc, le_kc)
            pos_all += dice
            npos += 1

        if npos > 0:
            pos_all /= float(npos)

        # Top-k fusion + adaptive gating
        order = torch.argsort(pos_all, descending=True).tolist()
        top = order[: max(1, min(args.fuse_topk, len(order)))]

        if 0 not in top and pos_all[0] >= pos_all[top[0]] - 0.02:
            top = [0] + top[:-1]

        s = pos_all[top].float()
        max_s = s.max() if s.numel() > 0 else torch.tensor(0.0, device=device)
        adaptive_mask = s >= (max_s - 0.3)

        if adaptive_mask.sum() > 0:
            s_valid = s[adaptive_mask]
            top_valid = [top[i] for i in range(len(top)) if bool(adaptive_mask[i].item())]
        else:
            s_valid = s
            top_valid = top

        if s_valid.numel() == 0:
            # fallback to uniform
            w = [1.0 / float(len(top_valid))] * len(top_valid) if top_valid else [1.0]
        else:
            w = torch.softmax(s_valid / max(args.tau_w, 1e-6), dim=0).detach().cpu().tolist()

        selected_tensors = []
        chosen_list = []
        for wi, idx_local in zip(w, top_valid):
            ii = keep[idx_local]
            selected_tensors.append((items[ii][0], items[ii][1], items[ii][2], float(wi)))
            chosen_list.append((ph[ii], float(wi), float(pos_all[idx_local].item())))

        lf_f, pm_f, le_f = fuse_tokens(selected_tensors)
        fused_lf[c] = lf_f.to(device)
        fused_pm_u8[c] = pm_f.to(device).to(torch.uint8)
        fused_le[c] = le_f.to(device)

        # Per-class BG threshold
        if c == bg_idx:
            local_bg_thr[c] = 0.0
        else:
            if str(args.bg_thr_mode).lower() == "none":
                local_bg_thr[c] = float(bg_thr)
            else:
                try:
                    lf_raw_1 = to_raw_from_BTD(lf_f.unsqueeze(0).to(device).half(), layout["language_features"])
                    pm_raw_1 = mask_to_raw_from_BT(pm_f.unsqueeze(0).to(device).bool(), layout["language_mask"]).bool()
                    le_raw_1 = to_raw_from_BTD(le_f.unsqueeze(0).to(device).half(), layout["language_embeds"])
                    local_bg_thr[c] = _estimate_bg_thr_for_class_dice(crops_c, lf_raw_1, pm_raw_1, le_raw_1)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        _stage_cleanup_cuda()
                    local_bg_thr[c] = float(bg_thr)
                except Exception:
                    local_bg_thr[c] = float(bg_thr)

        local_chosen[c] = {"class": classes[c], "chosen": chosen_list, "num_reps": len(crops_c)}

        if torch.cuda.is_available() and (c % 8 == 0):
            _stage_cleanup_cuda()

    # Aggregate fused results
    if dist_is_on():
        dist.all_reduce(fused_lf, op=dist.ReduceOp.SUM)
        dist.all_reduce(fused_le, op=dist.ReduceOp.SUM)
        dist.all_reduce(fused_pm_u8, op=dist.ReduceOp.SUM)

        gathered_chosen = [None] * world
        dist.all_gather_object(gathered_chosen, local_chosen)
        if is_main():
            merged = {}
            for d in gathered_chosen:
                if isinstance(d, dict):
                    merged.update(d)
            chosen_info = [merged.get(c, {"class": classes[c], "chosen": []}) for c in range(C)]
        else:
            chosen_info = []
    else:
        chosen_info = [local_chosen.get(c, {"class": classes[c], "chosen": []}) for c in range(C)]

    # Gather per-class BG thresholds across ranks
    if dist_is_on():
        gathered_bg_thr = [None] * world
        dist.all_gather_object(gathered_bg_thr, local_bg_thr)
        if is_main():
            merged_bg = {}
            for d in gathered_bg_thr:
                if isinstance(d, dict):
                    merged_bg.update(d)
            bg_thr_per_class = [float(merged_bg.get(c, float(bg_thr))) for c in range(C)]
        else:
            bg_thr_per_class = []
    else:
        bg_thr_per_class = [float(local_bg_thr.get(c, float(bg_thr))) for c in range(C)]

    # Stage III stats
    prof["stage3_crops"] = _ddp_reduce_sum_i64(stage3_crops_local, device)
    ddp_barrier()
    prof["stage3_sec"] = _ddp_reduce_max_f64(time.perf_counter() - t_s3_start, device)

    prof["total_sec"] = float(prof["stage1_sec"] + prof["stage2_sec"] + prof["stage3_sec"])

    # --- Final Bank Construction ---
    user_bank = None
    profile_min = {
        "stage1_min": round(_sec_to_min(prof["stage1_sec"]), 4),
        "stage2_min": round(_sec_to_min(prof["stage2_sec"]), 4),
        "stage3_min": round(_sec_to_min(prof["stage3_sec"]), 4),
        "total_min": round(_sec_to_min(prof["total_sec"]), 4),
        "stage1_crops": int(prof["stage1_crops"]),
        "stage2_crops": int(prof["stage2_crops"]),
        "stage3_crops": int(prof["stage3_crops"]),
    }

    if is_main():
        fused_pm = fused_pm_u8 > 0
        bg_cfg = {
            "mode": str(args.bg_thr_mode),
            "calib_view": "context_only_dense_map",
            "objective": "mean_dice" if str(args.bg_thr_mode).lower() == "dice" else "disabled",
        }

        user_bank = {
            "user_name": user_name,
            "cfg_path": cfg_path,
            "name_path": str(name_path),
            "query_words": classes,
            "query_idx": torch.arange(C, dtype=torch.long),
            "query_weights": torch.ones(C, dtype=torch.float32),
            "bg_thr_per_class": torch.tensor(bg_thr_per_class, dtype=torch.float32),
            "text_outputs": {
                "language_features": to_raw_from_BTD(fused_lf.half(), layout["language_features"]).cpu(),
                "language_mask": mask_to_raw_from_BT(fused_pm, layout["language_mask"]).cpu().bool(),
                "language_embeds": to_raw_from_BTD(fused_le.half(), layout["language_embeds"]).cpu(),
            },
            "text_batch_dim": {k: layout[k]["batch_dim"] for k in layout},
            "meta": {
                "type": "User Concept SubBank",
                "bg_idx": int(bg_idx),
                "rep_stats": [(c, len(selected_meta[c]) if selected_meta is not None else 0, target[c]) for c in range(C)],
                "chosen_info": chosen_info,
                "dataset_classes": ds_classes,
                "dsid_to_qid": dsid_to_qid,
                "world_size": world,
                "bg_thr_cfg": bg_cfg,
                "profile": profile_min,
            },
        }

        print0(
            f"[PROFILE][{user_name}] "
            f"S1 {profile_min['stage1_min']:.2f}m ({profile_min['stage1_crops']} crops) | "
            f"S2 {profile_min['stage2_min']:.2f}m ({profile_min['stage2_crops']} crops) | "
            f"S3 {profile_min['stage3_min']:.2f}m ({profile_min['stage3_crops']} crops) | "
            f"Total {profile_min['total_min']:.2f}m"
        )

    del dataset
    _stage_cleanup_cuda()
    ddp_barrier()
    return user_bank, global_text_layout_cache, profile_min


# =========================================================
# 9. Main Entry Point
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=str, required=True, help="name=cfg.py,...")
    parser.add_argument("--dataloader_key", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--seg_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--bpe_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--extra_calib", type=str, default=None)
    parser.add_argument("--cand_topk", type=int, default=999)
    parser.add_argument("--fuse_topk", type=int, default=999)
    parser.add_argument("--tau_w", type=float, default=0.05)
    parser.add_argument("--use_context_view", action="store_true")
    parser.add_argument("--use_masked_view", action="store_true")
    parser.add_argument("--view_weight_context", type=float, default=0.9)
    parser.add_argument("--view_weight_masked", type=float, default=0.1)
    parser.add_argument("--confidence_threshold", type=float, default=0.1)
    parser.add_argument("--topk_inst", type=int, default=100)
    parser.add_argument("--pad_ratio", type=float, default=0.2)
    parser.add_argument("--min_crop_size", type=int, default=128)
    parser.add_argument("--pass2_max_epochs", type=int, default=3)
    parser.add_argument("--pass2_emb_cache", type=int, default=16384)

    parser.add_argument(
        "--bg_thr_mode",
        type=str,
        default="none",
        choices=["dice", "none"],
        help="How to estimate per-class bg threshold: 'dice' or 'none'.",
    )

    parser.add_argument("--output_pt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ddp_init()
    rank, world = get_rank(), get_world_size()

    device = torch.device(args.device)
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()

    model = build_sam3_image_model(
        bpe_path=args.bpe_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        eval_mode=True,
        compile=args.compile,
    )
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold, device=args.device)

    users = parse_users_arg(args.users)
    if is_main():
        print0(f"[INFO] World Size: {world}, Seed: {seed}")
        print0(f"[INFO] Processing Users: {users}")

    global_text_layout_cache: Dict[str, Any] = {}
    users_out: Dict[str, Any] = {}
    prof_all: Dict[str, Any] = {}

    t_all_start = time.perf_counter()

    for (user_name, cfg_path) in users:
        user_bank, global_text_layout_cache, prof_min = build_one_user_bank(
            user_name, cfg_path, args,
            model, processor, device, amp_ctx,
            rank, world, global_text_layout_cache,
        )
        if is_main():
            users_out[user_name] = user_bank
            prof_all[user_name] = prof_min

    ddp_barrier()
    total_build_sec = _ddp_reduce_max_f64(time.perf_counter() - t_all_start, device)

    if is_main():
        out_dir = os.path.dirname(args.output_pt)
        mkdir_or_exist(out_dir if out_dir else ".")

        bank = {
            "type": "Multi-User Concept Bank",
            "version": "v1.0",
            "users": users_out,
            "global_text_layout": global_text_layout_cache.get("layout", None),
            "global_text_dims": {k: global_text_layout_cache.get(k, None) for k in ["T", "Df", "De"]},
            "sam3_config": {
                "checkpoint": args.checkpoint_path,
                "conf_thresh": args.confidence_threshold,
                "topk": args.topk_inst,
            },
            "build_args": vars(args),
            "build_profile": {
                "total_min": round(_sec_to_min(total_build_sec), 4),
                "per_user": prof_all,
            },
        }

        torch.save(bank, args.output_pt)
        print0(f"[OK] Saved to {args.output_pt}")
        print0(f"[PROFILE][ALL USERS] Total build time: {_sec_to_min(total_build_sec):.2f}m")

    ddp_barrier()
    ddp_cleanup()

if __name__ == "__main__":
    main()
