import os
import glob
import gc
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image

# MMLab & Torch imports
from mmengine.config import Config
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import MODELS

# SAM3 imports
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Try importing Concept Bank Model (SAM3_OVSS)
try:
    from sam3_ovss import SAM3_OVSS
except ImportError:
    print("Warning: sam3_ovss.py not found. Concept Bank mode will fail.")

# --- Global Cache ---
CURRENT_MODE = None       
CURRENT_MODEL = None      
CURRENT_CONFIG_ID = None  
CURRENT_USER = None       

# --- Constants ---
EXAMPLES_DIR = os.path.join("assets", "demo")

# ==============================================================================
# Part 1: Helper Functions & Visualization (Side-by-Side Legend)
# ==============================================================================

def generate_palette(num_classes):
    """Generates a fixed random palette for consistency."""
    state = np.random.RandomState(42) 
    return state.randint(0, 255, size=(num_classes + 50, 3))

def draw_legend_side(vis_img, legend_items):
    """Draws the legend on the RIGHT side of the image."""
    if not legend_items:
        return vis_img

    h, w, _ = vis_img.shape
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    margin = 15
    line_height = 30
    box_size = 20
    text_color = (255, 255, 255)
    bg_color = (40, 40, 40)
    
    max_text_width = 0
    for name, _ in legend_items:
        (tw, th), _ = cv2.getTextSize(name, font, font_scale, thickness)
        max_text_width = max(max_text_width, tw)
    
    legend_w = max_text_width + box_size + 3 * margin
    legend_h = len(legend_items) * line_height + 2 * margin
    
    new_h = max(h, legend_h)
    new_w = w + legend_w
    
    canvas = np.full((new_h, new_w, 3), bg_color, dtype=np.uint8)
    canvas[:h, :w, :] = vis_img
    
    x_start = w + margin
    y_start = margin
    
    for i, (name, color) in enumerate(legend_items):
        y = y_start + i * line_height
        draw_color = (int(color[0]), int(color[1]), int(color[2])) 
        cv2.rectangle(canvas, (x_start, y), (x_start + box_size, y + box_size), draw_color, -1)
        
        text_x = x_start + box_size + margin
        text_y = y + box_size - 5 
        cv2.putText(canvas, name, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
    return canvas

def overlay_masks(image_path, pred_mask, class_names):
    """Visualizes the mask and adds a side-by-side legend."""
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    vis_img = image_np.copy()
    
    unique_ids = np.unique(pred_mask)
    palette = generate_palette(max(len(class_names), 256))
    
    overlay = vis_img.copy()
    legend_items = [] 
    labels_found = []

    for class_id in unique_ids:
        idx = int(class_id)
        if idx == 255: continue 
        if idx >= len(class_names): continue
            
        class_name = class_names[idx]
        if idx == 0 and "back" in class_name.lower():
            continue

        color = palette[idx]
        mask_bool = (pred_mask == idx)
        overlay[mask_bool] = color
        
        legend_items.append((class_name, color))
        labels_found.append(class_name)

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
    vis_img = draw_legend_side(vis_img, legend_items)
    
    return vis_img, ", ".join(sorted(list(set(labels_found))))

# ==============================================================================
# Part 2: Original SAM3 Inference Logic (Baseline)
# ==============================================================================

def get_cls_idx(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    query_words = []
    query_idx = []
    canonical_names = []

    valid_lines = [line.strip() for line in lines if line.strip()]
    
    for idx, line in enumerate(valid_lines):
        names_i = [x.strip() for x in line.split(',')]
        names_i = [x for x in names_i if x]
        if not names_i: continue

        canonical_names.append(names_i[0])
        query_words += names_i
        query_idx += [idx] * len(names_i)

    return query_words, query_idx, canonical_names

@MODELS.register_module()
class Sam3OvSeg(BaseSegmentor):
    def __init__(self, classname_path, device=torch.device('cuda'), **kwargs):
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)
        self.device = device
        
        bpe_path = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        ckpt_path = "sam3/assets/sam3.pt"
        
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found at {ckpt_path}")

        model = build_sam3_image_model(
            bpe_path=bpe_path, 
            checkpoint_path=ckpt_path, 
            device=str(device)
        )
        self.processor = Sam3Processor(model, confidence_threshold=0.5, device=str(device))
        self.query_words, self.query_idx, self.canonical_names = get_cls_idx(classname_path)

    def predict(self, inputs, data_samples):
        if data_samples is None:
            raise ValueError("data_samples cannot be None")
        
        for i in range(len(data_samples)): 
            img_path = data_samples[i].metainfo.get('img_path')
            if not img_path:
                raise ValueError("Sam3OvSeg requires 'img_path' in metainfo.")

            image = Image.open(img_path).convert('RGB')
            ori_shape = data_samples[i].metainfo['ori_shape']
            
            seg_pred = torch.zeros((1, *ori_shape), device=self.device, dtype=torch.long)
            
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                inference_state = self.processor.set_image(image)
            
            for query_word, query_idx in zip(self.query_words, self.query_idx):
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    self.processor.reset_all_prompts(inference_state)
                    inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)
                
                inst_len = inference_state['masks_logits'].shape[0]
                if inst_len > 0:
                    for inst_id in range(inst_len):
                        instance_mask = inference_state['masks'][inst_id].squeeze(dim=0)
                        seg_pred[0][instance_mask] = query_idx
            
            data_samples[i].set_data({
                'pred_sem_seg': PixelData(**{'data': seg_pred})
            })
            
        return data_samples
    
    def _forward(self, data_samples): pass
    def inference(self, img, batch_img_metas): pass
    def encode_decode(self, inputs, batch_img_metas): pass
    def extract_feat(self, inputs): pass
    def loss(self, inputs, data_samples): pass

# ==============================================================================
# Part 3: File Scanning & Utils
# ==============================================================================

def get_concept_banks():
    search_path = os.path.join("configs", "concept_bank", "*.pt")
    files = glob.glob(search_path)
    if not files: files = glob.glob("*.pt")
    return sorted(files)

def get_baseline_configs():
    search_path = os.path.join("configs", "cfg_*.py")
    files = glob.glob(search_path)
    return sorted(files)

def get_bank_users(bank_path):
    if not bank_path or not os.path.exists(bank_path): return []
    try:
        bank = torch.load(bank_path, map_location="cpu")
        if isinstance(bank, dict) and "users" in bank and isinstance(bank["users"], dict):
            return sorted(list(bank["users"].keys()))
        return []
    except: return []

def get_categorized_examples():
    """
    Returns lists of (path, caption) tuples for Gradio Gallery.
    """
    if not os.path.exists(EXAMPLES_DIR):
        print(f"Warning: Examples directory {EXAMPLES_DIR} not found.")
        return [], [], []

    # Map filenames to display labels (Captions)
    ns_map = {
        "voc.jpg": "VOC",
        "pc.jpg": "Context",
        "coco.jpg": "COCO",
        "ade.jpg": "ADE20k",
        "city.png": "Cityscapes"
    }
    
    rs_map = {
        "loveda.png": "LoveDA",
        "potsdam.png": "Potsdam",
        "vaihingen.png": "Vaihingen",
        "isaid.png": "iSAID"
    }

    # Added Drift Mapping
    cd_map = {
        "drift_1.png": "Concept Drift",
        "drift_2.jpg": "Concept Drift",
        "drift_3.jpg": "Concept Drift",
        "drift_4.jpg": "Data Drift"
    }

    # Helper to build list preserving order
    def build_list(mapping):
        data = []
        for fname, label in mapping.items():
            path = os.path.join(EXAMPLES_DIR, fname)
            if os.path.exists(path):
                data.append((path, label))
        return data

    ns_data = build_list(ns_map)
    rs_data = build_list(rs_map)
    cd_data = build_list(cd_map)
            
    return ns_data, rs_data, cd_data

# ==============================================================================
# Part 4: Main Inference Controller
# ==============================================================================

def run_inference(
    mode_selection,
    image_path,
    cb_path,
    user_name,
    cfg_path,
    conf_thr
):
    global CURRENT_MODE, CURRENT_MODEL, CURRENT_CONFIG_ID, CURRENT_USER
    
    if image_path is None: return None, "Please upload an image."
    
    target_mode = "bank" if "Concept Bank" in mode_selection else "baseline"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Intelligent Model Loading ---
    needs_reload = False
    
    if CURRENT_MODE != target_mode:
        needs_reload = True
    elif target_mode == "bank":
        if CURRENT_CONFIG_ID != cb_path or CURRENT_USER != user_name:
            needs_reload = True
    elif target_mode == "baseline":
        if CURRENT_CONFIG_ID != cfg_path:
            needs_reload = True

    if needs_reload:
        print(f"[App] Reloading Model... Mode: {target_mode}")
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            CURRENT_MODEL = None
            gc.collect()
            torch.cuda.empty_cache()
            
        try:
            if target_mode == "bank":
                if not cb_path: return None, "Please select a Concept Bank file."
                
                model = SAM3_OVSS(
                    concept_path=cb_path,
                    concept_user=user_name,
                    device=device,
                    confidence_threshold=conf_thr,
                    bg_thr=0.0,
                    ig_thr=0.05,
                    bg_idx=0,
                    topk_inst=30,
                    checkpoint_path="sam3/assets/sam3.pt",
                    bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                    compile=False
                )
                model.to(device)
                
                CURRENT_MODEL = model
                CURRENT_CONFIG_ID = cb_path
                CURRENT_USER = user_name
                
            else:
                if not cfg_path: return None, "Please select a Config file."
                
                cfg = Config.fromfile(cfg_path)
                raw_path = cfg.model.name_path
                mod_path = raw_path.replace("ext_cls", "ori_cls")
                
                print(f"[App] Config: {cfg_path} | Class Path: {mod_path}")
                if not os.path.exists(mod_path):
                     return None, f"Error: Class file not found at {mod_path}"

                model = Sam3OvSeg(classname_path=mod_path, device=torch.device(device))
                
                CURRENT_MODEL = model
                CURRENT_CONFIG_ID = cfg_path
                CURRENT_USER = None
                
            CURRENT_MODE = target_mode
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Model Load Error: {str(e)}"

    # --- Update Dynamic Parameters ---
    if hasattr(CURRENT_MODEL, 'processor'):
        CURRENT_MODEL.processor.confidence_threshold = conf_thr
    if hasattr(CURRENT_MODEL, 'confidence_threshold'):
        CURRENT_MODEL.confidence_threshold = conf_thr

    # --- Inference Execution ---
    try:
        pil_img = Image.open(image_path)
        w, h = pil_img.size
        
        data_sample = SegDataSample()
        data_sample.set_metainfo({
            'img_path': image_path,  
            'img_shape': (h, w),
            'ori_shape': (h, w),
            'reduce_zero_label': False 
        })
        
        if target_mode == "bank":
            inputs = [pil_img]
        else:
            inputs = torch.zeros((1, 3, h, w)) 
        
        results = CURRENT_MODEL.predict(inputs, [data_sample])
        pred_mask = results[0].pred_sem_seg.data.squeeze().cpu().numpy()
        
        if target_mode == "bank":
            class_names = CURRENT_MODEL.query_words
        else:
            class_names = CURRENT_MODEL.canonical_names
            
        vis_img, info = overlay_masks(image_path, pred_mask, class_names)
        
        mode_str = "Concept Bank" if target_mode == "bank" else "Original SAM3"
        sub_info = f"Dataset: {CURRENT_USER}" if target_mode == "bank" and CURRENT_USER else f"Config: {os.path.basename(CURRENT_CONFIG_ID)}"
        
        return vis_img, f"[{mode_str}] {sub_info}\nDetected: {info if info else 'None'}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Inference Error: {str(e)}"

# ==============================================================================
# Part 5: Gradio UI Construction
# ==============================================================================

def on_mode_change(mode):
    if "Concept Bank" in mode:
        banks = get_concept_banks()
        def_bank = banks[0] if banks else None
        users = get_bank_users(def_bank)
        def_user = users[0] if users else None
        return [
            gr.Dropdown(visible=True, choices=banks, value=def_bank),
            gr.Dropdown(visible=bool(users), choices=users, value=def_user),
            gr.Dropdown(visible=False, value=None)
        ]
    else:
        cfgs = get_baseline_configs()
        def_cfg = cfgs[0] if cfgs else None
        return [
            gr.Dropdown(visible=False, value=None),
            gr.Dropdown(visible=False, value=None),
            gr.Dropdown(visible=True, choices=cfgs, value=def_cfg)
        ]

def on_bank_change(bank_path):
    users = get_bank_users(bank_path)
    if users:
        return gr.Dropdown(choices=users, value=users[0], visible=True, interactive=True)
    return gr.Dropdown(choices=[], value=None, visible=False)

title = "Concept Bank for Open-Vocabulary Segmentation"
desc = """
**Comparison Demo**:
* **Concept Bank (Ours)**: Uses pre-computed concept banks (`.pt`).
* **Original SAM3 (Baseline)**: Uses original class names (`ori_cls`) and standard SAM3 inference.
"""

# Initial state load
cb_files = get_concept_banks()
def_cb = cb_files[0] if cb_files else None
def_users = get_bank_users(def_cb)
def_user = def_users[0] if def_users else None
cfg_files = get_baseline_configs()

# Load categorized examples (List of tuples: (path, label))
ns_data, rs_data, cd_data = get_categorized_examples()

with gr.Blocks(title="Concept Bank") as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(desc)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="filepath", label="Input Image")
            
            # --- Categorized Gallery (Click to Select) ---
            if ns_data or rs_data or cd_data:
                with gr.Tabs():
                    with gr.TabItem("Natural Scene"):
                        if ns_data:
                            gallery_ns = gr.Gallery(
                                value=ns_data, 
                                label="Click image to test", 
                                columns=len(ns_data), 
                                height=160, 
                                object_fit="contain",
                                show_label=True,
                                allow_preview=False
                            )
                        else:
                            gr.Markdown("_No Natural Scene images found_")
                    
                    with gr.TabItem("Remote Sensing"):
                        if rs_data:
                            gallery_rs = gr.Gallery(
                                value=rs_data, 
                                label="Click image to test", 
                                columns=len(rs_data), 
                                height=160, 
                                object_fit="contain",
                                show_label=True,
                                allow_preview=False
                            )
                        else:
                            gr.Markdown("_No Remote Sensing images found_")

                    with gr.TabItem("Drift"):
                        if cd_data:
                            gallery_cd = gr.Gallery(
                                value=cd_data, 
                                label="Click image to test", 
                                columns=len(cd_data), 
                                height=160, 
                                object_fit="contain",
                                show_label=True,
                                allow_preview=False
                            )
                        else:
                            gr.Markdown("_No Drift images found_")
            
            mode_radio = gr.Radio(
                choices=["Concept Bank (Ours)", "Original SAM3 (Baseline)"],
                value="Concept Bank (Ours)",
                label="Inference Mode"
            )
            
            cb_drop = gr.Dropdown(choices=cb_files, value=def_cb, label="Concept Bank (.pt)", visible=True)
            user_drop = gr.Dropdown(choices=def_users, value=def_user, label="Dataset (User)", visible=bool(def_users))
            cfg_drop = gr.Dropdown(choices=cfg_files, value=None, label="Dataset Config (.py)", visible=False)
            
            with gr.Accordion("Advanced Options", open=True):
                conf_slider = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="Confidence Threshold")
            
            run_btn = gr.Button("Run Segmentation", variant="primary")
            
        with gr.Column(scale=2):
            out_img = gr.Image(type="numpy", label="Result Visualization")
            # Increased lines to 10 for better visibility
            out_txt = gr.Textbox(label="Result Details", lines=2)

    # --- Interactions ---
    
    def on_select_ns(evt: gr.SelectData):
        if evt.index is not None and evt.index < len(ns_data):
            return ns_data[evt.index][0]
        return None

    def on_select_rs(evt: gr.SelectData):
        if evt.index is not None and evt.index < len(rs_data):
            return rs_data[evt.index][0]
        return None

    def on_select_cd(evt: gr.SelectData):
        """Auto-configure dropdowns based on drift image type."""
        path = None
        cb_update = gr.update()
        user_update = gr.update()
        cfg_update = gr.update()

        if evt.index is not None and evt.index < len(cd_data):
            path = cd_data[evt.index][0]
            fname = os.path.basename(path)
            
            # Logic for automatic selection
            if fname in ["drift_1.png", "drift_2.jpg", "drift_3.jpg"]: # Concept Drift
                # Settings for Concept Bank Mode
                cb_path = "configs/concept_bank/cb_sam3_ns.pt"
                user_val = "coco_object"
                # IMPORTANT: Get valid choices to prevent validation error!
                valid_users = get_bank_users(cb_path) 
                
                cb_update = gr.update(value=cb_path)
                user_update = gr.update(value=user_val, choices=valid_users)
                
                # Settings for Baseline Mode
                cfg_update = gr.update(value="configs/cfg_coco_object.py")
                
            elif fname == "drift_4.jpg": # Data Drift
                # Settings for Concept Bank Mode
                cb_path = "configs/concept_bank/cb_sam3_rs.pt"
                user_val = "isaid"
                # IMPORTANT: Get valid choices to prevent validation error!
                valid_users = get_bank_users(cb_path)
                
                cb_update = gr.update(value=cb_path)
                user_update = gr.update(value=user_val, choices=valid_users)
                
                # Settings for Baseline Mode
                cfg_update = gr.update(value="configs/cfg_isaid.py")

        return path, cb_update, user_update, cfg_update

    if ns_data:
        gallery_ns.select(
            fn=on_select_ns, 
            inputs=None, 
            outputs=input_img
        )
    
    if rs_data:
        gallery_rs.select(
            fn=on_select_rs, 
            inputs=None, 
            outputs=input_img
        )

    if cd_data:
        gallery_cd.select(
            fn=on_select_cd,
            inputs=None,
            outputs=[input_img, cb_drop, user_drop, cfg_drop]
        )

    mode_radio.change(on_mode_change, inputs=[mode_radio], outputs=[cb_drop, user_drop, cfg_drop])
    cb_drop.change(on_bank_change, inputs=[cb_drop], outputs=[user_drop])
    
    run_btn.click(
        fn=run_inference,
        inputs=[mode_radio, input_img, cb_drop, user_drop, cfg_drop, conf_slider],
        outputs=[out_img, out_txt]
    )

if __name__ == "__main__":
    print("Starting Gradio Demo...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)