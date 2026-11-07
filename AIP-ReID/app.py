import hashlib
import os
import re
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, send_from_directory
from torchvision import transforms
from werkzeug.utils import secure_filename

from config import cfg
from model.make_model_clipreid import make_model

# =========================
# CONFIG
# =========================
CONFIG_FILE = "configs/person/vit_clipreid.yml"
#20
CHECKPOINT_PATH = "ViT-B-16_90.pth"
CHECKPOINT_PATH_STAGE1 = r"D:\Uni_Documents\nam4_HKI\KLTN\CLIP-ReID\CNIF-ReID_app\ViT-B-16_stage1_120.pth"
UPLOAD_DIR = "uploads"
CACHE_DIR = "cache"
PROGRESS_LOG_EVERY = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
DEFAULT_DATASET = "market1501"
DEFAULT_METHOD = "cosine"
DEFAULT_TOPK = 10
MAX_TOPK = 200

DATASET_OPTIONS: Dict[str, Dict[str, str]] = {
    "market1501": {
        "label": "Market1501",
        "gallery_dir": "gallery/market",
        "cache_file": os.path.join(CACHE_DIR, "gallery_features.pt"),
    },
    "msmt17": {
        "label": "MSMT17",
        "gallery_dir": "gallery/msmt17",
        "cache_file": os.path.join(CACHE_DIR, "msmt17_gallery_features.pt"),
    },
}

app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

model = None
model_stage1 = None   # model dành riêng cho token
transform_test = None

dataset_indices = {
    dataset_key: {
        "features": None,
        "paths": [],
        "ready": False,
    }
    for dataset_key in DATASET_OPTIONS
}

# =========================
# FILTER JINJA2
# =========================
@app.template_filter('token_preview')
def token_preview_filter(tensor):
    if tensor is None:
        return "None"
    vec = tensor.squeeze().tolist()
    preview = ", ".join(f"{x:.4f}" for x in vec[:6])
    return f"[{preview}, ...])"

# =========================
# HELPERS
# =========================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def strip_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    return new_state

def infer_num_classes_from_ckpt(state_dict):
    if "classifier.weight" in state_dict:
        return state_dict["classifier.weight"].shape[0]
    raise RuntimeError("Không tìm thấy classifier.weight trong checkpoint.")

def load_cfg():
    cfg.defrost()
    cfg.merge_from_file(CONFIG_FILE)
    cfg.MODEL.DEVICE = DEVICE
    if hasattr(cfg.MODEL, "DIST_TRAIN"):
        cfg.MODEL.DIST_TRAIN = False
    cfg.freeze()

def build_transform():
    height, width = cfg.INPUT.SIZE_TEST
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

def load_reid_model():
    global model
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    checkpoint = strip_module_prefix(checkpoint)
    num_classes = infer_num_classes_from_ckpt(checkpoint)
    camera_num = 0
    view_num = 0
    net = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    net.load_state_dict(checkpoint, strict=True)
    net.to(DEVICE)
    net.eval()
    model = net
    print(f"[INFO] Loaded model from {CHECKPOINT_PATH}")
    print(f"[INFO] num_classes = {num_classes}")
def load_stage1_model():
    global model_stage1
    if not os.path.exists(CHECKPOINT_PATH_STAGE1):
        print(f"[WARN] Không tìm thấy stage1 checkpoint: {CHECKPOINT_PATH_STAGE1}")
        model_stage1 = None
        return
    
    checkpoint = torch.load(CHECKPOINT_PATH_STAGE1, map_location="cpu")
    checkpoint = strip_module_prefix(checkpoint)
    
    # Tạo model stage1 với num_class=0 (không cần classifier)
    net = make_model(cfg, num_class=0, camera_num=0, view_num=0)
    
    # Lọc checkpoint: chỉ giữ các key có trong net và shape khớp
    model_dict = net.state_dict()
    filtered_checkpoint = {}
    skipped_keys = []
    for k, v in checkpoint.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_checkpoint[k] = v
            else:
                skipped_keys.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
        else:
            skipped_keys.append(f"{k} (not in model)")
    
    if skipped_keys:
        print(f"[INFO] Stage1: bỏ qua {len(skipped_keys)} key do không khớp hoặc không cần thiết.")
        for sk in skipped_keys[:5]:
            print(f"  - {sk}")
    
    # Load filtered state_dict (strict=False vẫn an toàn)
    missing, unexpected = net.load_state_dict(filtered_checkpoint, strict=False)
    if missing:
        print(f"[INFO] Stage1 missing keys (ignored): {missing[:5]}...")
    if unexpected:
        print(f"[INFO] Stage1 unexpected keys (ignored): {unexpected[:5]}...")
    
    net.to(DEVICE)
    net.eval()
    model_stage1 = net
    print(f"[INFO] Loaded stage1 model from {CHECKPOINT_PATH_STAGE1}")
def extract_feature_from_pil(img_pil: Image.Image) -> torch.Tensor:
    x = transform_test(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model(x, cam_label=None, view_label=None)
        feat = F.normalize(feat, p=2, dim=1)
    return feat.squeeze(0).cpu()

def extract_attribute_tokens(img_pil: Image.Image):
    """
    Trích xuất 6 token từ ảnh dùng model_stage1.
    Trả về dict chứa các tensor shape [1,512] trên CPU.
    """
    if model_stage1 is None:
        return None
    x = transform_test(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, token_feats = model_stage1(x, get_image=True)   # token_feats: [1, 1+N, 512]
        core = model_stage1.module if hasattr(model_stage1, 'module') else model_stage1
        (person_token, top_token, underneath_token,
         hairstyle_token, shoes_token, carrying_token) = core.inversion_prompt_learner(token_feats)
    return {
        'person': person_token.squeeze(0).cpu(),
        'top': top_token.squeeze(0).cpu(),
        'underneath': underneath_token.squeeze(0).cpu(),
        'hairstyle': hairstyle_token.squeeze(0).cpu(),
        'shoes': shoes_token.squeeze(0).cpu(),
        'carrying': carrying_token.squeeze(0).cpu()
    }

def get_dataset_config(dataset_key: str) -> Dict[str, str]:
    if dataset_key not in DATASET_OPTIONS:
        raise ValueError(f"Dataset không hợp lệ: {dataset_key}")
    return DATASET_OPTIONS[dataset_key]

def collect_gallery_files(gallery_dir: str, valid_exts: Tuple[str, ...]) -> List[str]:
    all_files = []
    for root, _, filenames in os.walk(gallery_dir):
        for fname in filenames:
            if fname.startswith("-1"):
                continue
            if fname.lower().endswith(valid_exts):
                all_files.append(os.path.join(root, fname))
    all_files.sort()
    return all_files

def build_gallery_signature(all_files: List[str], gallery_dir: str) -> str:
    chunks = []
    for path in all_files:
        stat = os.stat(path)
        rel_path = os.path.relpath(path, gallery_dir).replace("\\", "/")
        chunks.append(f"{rel_path}|{stat.st_size}|{stat.st_mtime_ns}")
    if os.path.exists(CHECKPOINT_PATH):
        ckpt_stat = os.stat(CHECKPOINT_PATH)
        chunks.append(f"checkpoint|{ckpt_stat.st_size}|{ckpt_stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(chunks).encode("utf-8")).hexdigest()

def load_gallery_cache(cache_file: str, signature: str):
    if not os.path.exists(cache_file):
        return None, None
    try:
        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
        cached_signature = cache.get("signature")
        if cached_signature != signature:
            print(f"[INFO] Cache mismatch ở {cache_file}, sẽ rebuild gallery features.")
            return None, None
        cached_features = cache.get("features")
        cached_paths = cache.get("paths")
        if not isinstance(cached_features, torch.Tensor) or not isinstance(cached_paths, list):
            return None, None
        return cached_features, cached_paths
    except Exception as e:
        print(f"[WARNING] Cache lỗi ở {cache_file}: {e}")
        return None, None

def save_gallery_cache(cache_file: str, signature: str, features: torch.Tensor, paths: List[str]):
    payload = {
        "signature": signature,
        "features": features,
        "paths": paths,
    }
    torch.save(payload, cache_file)
    print(f"[INFO] Saved gallery cache to {cache_file}")

def build_gallery_index(dataset_key: str):
    dataset_cfg = get_dataset_config(dataset_key)
    gallery_dir = dataset_cfg["gallery_dir"]
    cache_file = dataset_cfg["cache_file"]
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    if not os.path.exists(gallery_dir):
        raise RuntimeError(f"Dataset '{dataset_cfg['label']}' chưa sẵn sàng vì chưa có thư mục gallery: {gallery_dir}")
    all_files = collect_gallery_files(gallery_dir, valid_exts)
    if len(all_files) == 0:
        raise RuntimeError(f"Dataset '{dataset_cfg['label']}' chưa có ảnh gallery trong thư mục {gallery_dir}")
    signature = build_gallery_signature(all_files, gallery_dir)
    cached_features, cached_paths = load_gallery_cache(cache_file, signature)
    if cached_features is not None and cached_paths is not None:
        normalized_cached_paths = []
        for p in cached_paths:
            if not isinstance(p, str):
                normalized_cached_paths = []
                break
            normalized_cached_paths.append(p.replace("\\", "/").lstrip("/"))
        if len(normalized_cached_paths) != len(cached_paths):
            print(f"[INFO] Cache paths không hợp lệ cho {dataset_key}, sẽ rebuild.")
        elif any(not os.path.exists(os.path.join(gallery_dir, p)) for p in normalized_cached_paths):
            print(f"[INFO] Cache paths không khớp gallery hiện tại cho {dataset_key}, sẽ rebuild.")
        else:
            dataset_indices[dataset_key]["features"] = cached_features
            dataset_indices[dataset_key]["paths"] = normalized_cached_paths
            dataset_indices[dataset_key]["ready"] = True
            print(f"[INFO] Loaded cached gallery features for {dataset_key}: {cached_features.shape}")
            return
    print(f"[INFO] Đang trích đặc trưng cho dataset {dataset_key}: {len(all_files)} ảnh gallery...")
    feats = []
    rel_paths = []
    for idx, path in enumerate(all_files, start=1):
        try:
            img = Image.open(path).convert("RGB")
            feat = extract_feature_from_pil(img)
            feats.append(feat)
            rel_paths.append(os.path.relpath(path, gallery_dir).replace("\\", "/"))
        except Exception as e:
            print(f"[WARNING] Bỏ qua ảnh lỗi {path}: {e}")
        if idx == 1 or idx % PROGRESS_LOG_EVERY == 0 or idx == len(all_files):
            print(f"[INFO] [{dataset_key}] Tiến độ trích xuất: {idx}/{len(all_files)}")
    if len(feats) == 0:
        raise RuntimeError(f"Không trích được feature cho dataset {dataset_cfg['label']}")
    features = torch.stack(feats, dim=0)
    dataset_indices[dataset_key]["features"] = features
    dataset_indices[dataset_key]["paths"] = rel_paths
    dataset_indices[dataset_key]["ready"] = True
    save_gallery_cache(cache_file, signature, features, rel_paths)
    print(f"[INFO] [{dataset_key}] Gallery features shape: {features.shape}")

def ensure_dataset_ready(dataset_key: str):
    if dataset_indices[dataset_key]["ready"]:
        return
    build_gallery_index(dataset_key)

def parse_identity_camera_from_path(rel_path: str) -> Tuple[str, str]:
    """Parse person ID and camera label from image filename.

    Compatible with Market1501/MSMT17 style names like: 0002_c3_....jpg
    """
    filename = os.path.basename(rel_path)
    match = re.search(r"([-\d]+)_c(\d+)", filename)
    if match is None:
        return "Unknown", "Unknown"
    person_id = match.group(1)
    camera_label = f"c{int(match.group(2))}"
    return person_id, camera_label

def search_similar_images(query_img_pil: Image.Image, dataset_key: str, topk: int = 10, method: str = "cosine"):
    ensure_dataset_ready(dataset_key)
    query_feat = extract_feature_from_pil(query_img_pil).unsqueeze(0)
    gallery_features = dataset_indices[dataset_key]["features"]
    gallery_paths = dataset_indices[dataset_key]["paths"]
    gallery_dir = DATASET_OPTIONS[dataset_key]["gallery_dir"]
    g_feats = F.normalize(gallery_features, p=2, dim=1)
    topk = min(topk, len(gallery_paths))
    results = []
    if method == "cosine":
        sims = torch.mm(query_feat, g_feats.t()).squeeze(0)
        values, indices = torch.topk(sims, k=topk, largest=True)
        for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            rel_path = str(gallery_paths[idx]).replace("\\", "/").lstrip("/")
            person_id, camera = parse_identity_camera_from_path(rel_path)
            results.append({
                "rank": rank,
                "path": rel_path,
                "full_path": os.path.join(gallery_dir, rel_path).replace("\\", "/"),
                "metric_name": "Cosine",
                "metric_value": round(score, 6),
                "person_id": person_id,
                "camera": camera,
            })
    elif method == "euclidean":
        dists = torch.cdist(query_feat, g_feats, p=2).squeeze(0)
        values, indices = torch.topk(dists, k=topk, largest=False)
        for rank, (dist, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            rel_path = str(gallery_paths[idx]).replace("\\", "/").lstrip("/")
            person_id, camera = parse_identity_camera_from_path(rel_path)
            results.append({
                "rank": rank,
                "path": rel_path,
                "full_path": os.path.join(gallery_dir, rel_path).replace("\\", "/"),
                "metric_name": "Euclidean",
                "metric_value": round(dist, 6),
                "person_id": person_id,
                "camera": camera,
            })
    else:
        raise ValueError("method phải là 'cosine' hoặc 'euclidean'")
    return results

def safe_parse_topk(value) -> int:
    try:
        topk = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TOPK
    return max(1, min(MAX_TOPK, topk))

def render_home(**kwargs):
    context = {
        "results": None,
        "query_image": None,
        "selected_dataset": DEFAULT_DATASET,
        "selected_method": DEFAULT_METHOD,
        "selected_topk": DEFAULT_TOPK,
        "dataset_options": DATASET_OPTIONS,
        "error": None,
        "info": None,
        "tokens": None,
        "show_tokens": False,
    }
    context.update(kwargs)
    return render_template("index.html", **context)

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET"])
def index():
    info = None
    if os.path.exists(DATASET_OPTIONS[DEFAULT_DATASET]["gallery_dir"]):
        info = f"Dataset mặc định: {DATASET_OPTIONS[DEFAULT_DATASET]['label']}"
    return render_home(info=info)

@app.route("/search", methods=["POST"])
def search():
    dataset_key = request.form.get("dataset", DEFAULT_DATASET)
    method = request.form.get("method", DEFAULT_METHOD)
    topk = safe_parse_topk(request.form.get("topk", DEFAULT_TOPK))
    show_tokens = request.form.get("show_tokens") == "true"

    if dataset_key not in DATASET_OPTIONS:
        dataset_key = DEFAULT_DATASET
    if method not in {"cosine", "euclidean"}:
        method = DEFAULT_METHOD

    if "query_image" not in request.files:
        return render_home(
            selected_dataset=dataset_key,
            selected_method=method,
            selected_topk=topk,
            show_tokens=show_tokens,
            error="Không tìm thấy file upload."
        )
    file = request.files["query_image"]
    if file.filename == "":
        return render_home(
            selected_dataset=dataset_key,
            selected_method=method,
            selected_topk=topk,
            show_tokens=show_tokens,
            error="Bạn chưa chọn ảnh."
        )
    if not allowed_file(file.filename):
        return render_home(
            selected_dataset=dataset_key,
            selected_method=method,
            selected_topk=topk,
            show_tokens=show_tokens,
            error="Định dạng ảnh không hợp lệ."
        )
    file_ext = file.filename.rsplit(".", 1)[1].lower()
    saved_name = f"query_preview.{file_ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(saved_name))
    file.save(save_path)

    try:
        query_img = Image.open(save_path).convert("RGB")
        results = search_similar_images(query_img, dataset_key=dataset_key, topk=topk, method=method)
        tokens = None
        if show_tokens:
            tokens = extract_attribute_tokens(query_img)
        info = ""
    except Exception as e:
        return render_home(
            selected_dataset=dataset_key,
            selected_method=method,
            selected_topk=topk,
            show_tokens=show_tokens,
            query_image=saved_name,
            error=f"Lỗi khi truy vấn: {e}"
        )
    return render_home(
        results=results,
        query_image=saved_name,
        selected_dataset=dataset_key,
        selected_method=method,
        selected_topk=topk,
        info=info,
        tokens=tokens,
        show_tokens=show_tokens,
    )

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/gallery/<dataset_key>/<path:filename>")
def gallery_file(dataset_key, filename):
    if dataset_key not in DATASET_OPTIONS:
        return "Dataset không hợp lệ", 404
    return send_from_directory(DATASET_OPTIONS[dataset_key]["gallery_dir"], filename)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    load_cfg()
    transform_test = build_transform()
    load_reid_model()
    load_stage1_model()   # load model stage1 để trích token

    # Build cho market nếu có
    if os.path.exists(DATASET_OPTIONS["market1501"]["gallery_dir"]):
        try:
            build_gallery_index("market1501")
        except Exception as e:
            print(f"[WARNING] Không thể build market: {e}")
    if os.path.exists(DATASET_OPTIONS["msmt17"]["gallery_dir"]):
        try:
            build_gallery_index("msmt17")
        except Exception as e:
            print(f"[WARNING] Không thể build msmt17: {e}")

    app.run(host="0.0.0.0", port=5000, debug=True)