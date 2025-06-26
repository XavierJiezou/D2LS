<div align="center">

<!-- <img src="/images/logo.png" alt="logo" width="200" height="200"> -->

# D2LS

Dynamic Dictionary Learning for Remote Sensing Image Segmentation

<!--[![Overleaf](https://img.shields.io/badge/Overleaf-Open-green?logo=Overleaf&style=flat)](https://www.overleaf.com/project/6695fd4634d7fee5d0b838e5)-->

<!--Love the project? Please consider [donating](https://paypal.me/xavierjiezou?country.x=C2&locale.x=zh_XC) to help it improve!-->

![framework](/images/framework.png)


</div>


## Install

```
conda create -n d2ls python=3.8
conda activate d2ls
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Prepare Data

Prepare the following folders to organize this repo:
```none
D2LS
├── network
├── config
├── tools
├── model_weights (save the model weights)
├── fig_results (save the masks predicted)
├── lightning_logs (CSV format training logs)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
│   ├── uavid
│   │   ├── uavid_train (original)
│   │   ├── uavid_val (original)
│   │   ├── uavid_test (original)
│   │   ├── uavid_train_val (Merge uavid_train and uavid_val)
│   │   ├── train (processed)
│   │   ├── val (processed)
│   │   ├── train_val (processed)
│   ├── potsdam
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── grass
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── cloud
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
```

## Data Preprocessing

Download Datasets
- [ISPRS Vaihingen, Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
- [UAVid](https://uavid.nl/)
- [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)

Configure the folder as shown in 'Folder Structure' above.

**UAVid**

```
python tools/uavid_patch_split.py --input-dir "data/uavid/uavid_train_val" --output-img-dir "data/uavid/train_val/images" --output-mask-dir "data/uavid/train_val/masks" --mode "train" --split-size-h 1024 --split-size-w 1024 --stride-h 1024 --stride-w 1024
```
```
python tools/uavid_patch_split.py --input-dir "data/uavid/uavid_train" --output-img-dir "data/uavid/train/images" --output-mask-dir "data/uavid/train/masks" --mode 'train' --split-size-h 1024 --split-size-w 1024 --stride-h 1024 --stride-w 1024
```
```
python tools/uavid_patch_split.py --input-dir "data/uavid/uavid_val" --output-img-dir "data/uavid/val/images" --output-mask-dir "data/uavid/val/masks" --mode 'val' --split-size-h 1024 --split-size-w 1024 --stride-h 1024 --stride-w 1024
```

**Vaihingen**

The [paper]() contains the identity splits for all datasets.


- Using 3 zip files: ISPRS_semantic_labeling_Vaihingen.zip, ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip, ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip
- 'gts_for_participants' folder of ISPRS_semantic_labeling_Vaihingen.zip --> train_masks
- Files in the 'top' folder of ISPRS_semantic_labeling_Vaihingen.zip that correspond to train(+val) ID --> train_images
- Files of ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip that correspond to test ID --> test_masks
- Files of ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip that correspond to test ID --> test_masks_eroded
- Files in the 'top' folder of ISPRS_semantic_labeling_Vaihingen.zip that correspond to test ID --> train_images

```
python GeoSeg/tools/vaihingen_patch_split.py --img-dir "data/vaihingen/train_images" --mask-dir "data/vaihingen/train_masks" --output-img-dir "data/vaihingen/train/images_1024" --output-mask-dir "data/vaihingen/train/masks_1024" --mode "train" --split-size 1024 --stride 512
```
```
python GeoSeg/tools/vaihingen_patch_split.py --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks_eroded" --output-img-dir "data/vaihingen/test/images_1024" --output-mask-dir "data/vaihingen/test/masks_1024" --mode "val" --split-size 1024 --stride 1024 --eroded
```
```
python GeoSeg/tools/vaihingen_patch_split.py --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks" --output-img-dir "data/vaihingen/test/images_1024" --output-mask-dir "data/vaihingen/test/masks_1024_rgb" --mode "val" --split-size 1024 --stride 1024 --gt
```

**Potsdam**

- Using 3 zip files: 2_Ortho_RGB.zip, 5_Labels_all.zip, 5_Labels_for_all_no_Boundary.zip
- Files of 2_Ortho_RGB.zip that correspond to train(+val) ID --> train_images
- Files of 2_Ortho_RGB.zip that correspond to test ID --> test_images
- Files of 5_Labels_all.zip that correspond to train(+val) ID --> train_masks
- Files of 5_Labels_all.zip that correspond to test ID --> test_masks
- Files of 5_Labels_for_all_noBoundary.zip that correspond to test ID --> test_masks_eroded

```
python tools/potsdam_patch_split.py --img-dir "data/potsdam/train_images" --mask-dir "data/potsdam/train_masks" --output-img-dir "data/potsdam/train/images_1024" --output-mask-dir "data/potsdam/train/masks_1024" --mode "train" --split-size 1024 --stride 1024 --rgb-image
```
```
python tools/potsdam_patch_split.py --img-dir "data/potsdam/test_images" --mask-dir "data/potsdam/test_masks_eroded" --output-img-dir "data/potsdam/test/images_1024" --output-mask-dir "data/potsdam/test/masks_1024" --mode "val" --split-size 1024 --stride 1024 --eroded --rgb-image
```
```
python tools/potsdam_patch_split.py --img-dir "data/potsdam/test_images" --mask-dir "data/potsdam/test_masks" --output-img-dir "data/potsdam/test/images_1024" --output-mask-dir "data/potsdam/test/masks_1024_rgb" --mode "val" --split-size 1024 --stride 1024 --gt --rgb-image
```

**LoveDA**

```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
```
```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
```
```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
```
```
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```

**Grass and Cloud**

You can download Grass Dataset and Cloud Dataset from [KTDA](https://huggingface.co/datasets/XavierJiezou/ktda-datasets)

## Training

"-c" means the path of the config, use different **config** to train different models.

```
python train.py -c config/loveda/d2ls.py
```


## Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format


**Vaihingen**

```
python test_vaihingen.py -c config/vaihingen/d2ls.py -o fig_results/vaihingen/d2ls --rgb -t 'd4'
```


**Potsdam**

```
python test_potsdam.py -c config/potsdam/d2ls.py -o fig_results/potsdam/d2ls --rgb -t 'lr'
```


**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))

- To get RGB files:
```
python test_loveda.py -c config/loveda/d2ls.py -o fig_results/loveda/d2ls --rgb -t "d4"
```

- For submitting to the online test site:
```
python test_loveda.py -c config/loveda/d2ls.py -o fig_results/loveda/d2ls -t "d4"
```


**UAVid** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/7302))

```
python test_uavid.py -i "data/uavid/uavid_test" -c config/uavid/d2ls.py -o fig_results/uavid/d2ls -t "lr" -ph 1152 -pw 1024 -b 2 -d "uavid"
```
