import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import cfg
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from utils.logger import setup_logger

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    pids = []
    camids = []
    with torch.no_grad():
        for imgs, pid, camid, camid_batch, viewid, img_path in dataloader:
            imgs = imgs.to(device)
            feat = model(imgs)
            features.append(feat.cpu())
            pids.extend(pid)
            camids.extend(camid)

    features = torch.cat(features, dim=0)
    pids = np.asarray(pids, dtype=np.int64)
    camids = np.asarray(camids, dtype=np.int64)
    return features, pids, camids

def evaluate(model, val_loader, num_query, device):
    """Tính CMC và mAP"""
    features, pids, camids = extract_features(model, val_loader, device)
    # Split query và gallery
    qf = features[:num_query]
    q_pids = pids[:num_query]
    q_camids = camids[:num_query]
    gf = features[num_query:]
    g_pids = pids[num_query:]
    g_camids = camids[num_query:]
    
    from utils.metrics import euclidean_distance, eval_func
    distmat = euclidean_distance(qf, gf)
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50)
    return cmc, mAP

def plot_cmc(cmc, save_path=None):
    """Vẽ CMC curve với trục y zoom từ 85 đến 100, chia mỗi 0.1"""
    ranks = np.arange(1, len(cmc) + 1)
    cmc_percent = cmc * 100

    plt.figure(figsize=(10, 6))
    plt.plot(ranks, cmc_percent, marker='o', markersize=4, linewidth=2)

    plt.xlabel('Rank', fontsize=14)
    plt.ylabel('Matching Rate (%)', fontsize=14)
    plt.title('CMC Curve', fontsize=16)

    plt.xlim(1, len(cmc))
    plt.ylim(85, 100)

    # chia trục y mỗi 0.1
    plt.yticks(np.arange(80, 100.1, 0.5))

    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved CMC curve to {save_path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CMC for ReID model")
    parser.add_argument("--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str)
    parser.add_argument("--weight", type=str, required=True, help="path to model weight .pth")
    parser.add_argument("--output_dir", default="logs/eval", help="directory to save results")
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Merge config
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.WEIGHT = args.weight  # ghi đè đường dẫn weight
    cfg.freeze()

    # Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("eval", args.output_dir, if_train=False)
    logger.info(f"Evaluating model from {args.weight}")

    # Load data
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    logger.info(f"Number of query: {num_query}, gallery: {len(val_loader.dataset) - num_query}")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.to(device)

    # Evaluate
    cmc, mAP = evaluate(model, val_loader, num_query, device)

    # In kết quả
    logger.info(f"mAP: {mAP:.2%}")
    for rank in [1, 5, 10, 20]:
        logger.info(f"Rank-{rank}: {cmc[rank-1]:.2%}")

    # Lưu kết quả
    np.save(os.path.join(args.output_dir, 'cmc.npy'), cmc)
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"mAP: {mAP:.4f}\n")
        for i, acc in enumerate(cmc):
            f.write(f"Rank-{i+1}: {acc:.4f}\n")

    # Vẽ CMC curve
    plot_cmc(cmc, save_path=os.path.join(args.output_dir, 'cmc_curve.png'))