HUONG DAN CHAY CODE VA DEMO APP (TIENG VIET)

### Cai dat moi truong

```
conda create -n clipreid python=3.8
conda activate clipreid
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install yacs
pip install timm
pip install scikit-image
pip install tqdm
pip install ftfy
pip install regex
```

### Chuan bi du lieu

Tai cac bo du lieu ([Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775)) ve thu muc `your_dataset_dir`.

### Huong dan huan luyen (Training)

Neu muon chay CLIP-ReID (ViT-based) cho Market-1501, sua cuoi file configs/person/vit_aipreid.yml thanh:

```
DATASETS:
   NAMES: ('market1501')
   ROOT_DIR: ('your_dataset_dir')
OUTPUT_DIR: 'your_output_dir'
```

Sau do chay:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/person/vit_aipreid.yml
```

Neu muon chay CLIP-ReID (ViT-based) cho MSMT17, sua cuoi file configs/person/vit_aipreid.yml thanh:

```
DATASETS:
   NAMES: ('msmt17')
   ROOT_DIR: ('your_dataset_dir')
OUTPUT_DIR: 'your_output_dir'
```

Sau do chay:

```
CUDA_VISIBLE_DEVICES=0 python train_clipreid.py --config_file configs/person/vit_aipreid.yml
```

### Danh gia (Evaluation)

Vi du, neu muon test CLIP-ReID (ViT-based) cho MSMT17:

```
CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/person/vit_aipreid.yml TEST.WEIGHT 'your_trained_checkpoints_path/ViT-B-16_60.pth'
```

### Chay web demo

#### Cach sap xep dataset cho gallery (web app)

Thu muc `AIP-ReID/gallery` da bi xoa, hay tao lai va copy anh gallery vao dung duong dan.
App se quet thu muc `gallery` de quy, chi can giu cau truc ben duoi.

Market1501:

```
AIP-ReID/
   gallery/
      market/
         bounding_box_test/
            0001_c1s1_001051_00.jpg
            ...
```

MSMT17:

```
AIP-ReID/
   gallery/
      msmt17/
         bounding_box_test/
            0001_0001_0001_0001_0001.jpg
            ...
```

Ghi chu:

- Neu da tai dataset, co the copy (hoac symlink) `bounding_box_test` vao cac duong dan `gallery` tren.
- Ten anh giu nguyen; app bo qua file bat dau bang `-1`.
- Muon doi duong dan, sua `DATASET_OPTIONS` trong `AIP-ReID/app.py`.

Duong dan dataset (gallery):

```
AIP-ReID/gallery/market
AIP-ReID/gallery/msmt17/bounding_box_test
```

Duong dan file pth (checkpoint) cho web:

```
AIP-ReID/ViT-B-16_90.pth
AIP-ReID/ViT-B-16_stage1_120.pth
```

Neu luu checkpoint o noi khac, sua cac bien trong AIP-ReID/app.py:

```
CHECKPOINT_PATH = "ViT-B-16_90.pth"
CHECKPOINT_PATH_STAGE1 = "ViT-B-16_stage1_120.pth"
```

Chay lenh:

```
cd AIP-ReID
python app.py
```

Sau khi chay web demo, se tao 2 file cache trong thu muc `cache/`.

### Ve duong cong CMC

```
CUDA_VISIBLE_DEVICES=0 python eval_cmc.py ---config_file configs/person/vit_aipreid.yml TEST.WEIGHT 'your_trained_checkpoints_path/ViT-B-16_60.pth'
```
