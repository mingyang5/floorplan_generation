# HouseDiffusion adapted for Modified Swiss Dwellings dataset

## 1. Installation
**1. Clone the repo and install the requirements:**

```bash
# cd house_diffusion

mamba install -f pytorchhousediffusion.yml
mamba activate pytorchhousediffusion

pip install -e .
```

## 2. Training Process

```bash
cd model/scripts

CUDA_VISIBLE_DEVICES=1 python image_train.py --dataset modified_swiss_dwellings --batch_size 64 --set_name train --train_num_steps 80000 --timeout 36:00:00 --save_interval 10000 --test_interval 5000 --use_wall_self_attention true
```

## 3. Inference
To sample floorplans, and store resulting images (and pickles of predictions), run image_inference_msd.py:

```bash
cd model/scripts

python image_inference_msd.py --dataset modified_swiss_dwellings --batch_size 4 --set_name test --model_path /home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/ckpts/openai_2024_11_02_15_43_14_565720/model038000.pt --path_struct "/home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/datasets/modified-swiss-dwellings-v2/test/struct_in" --dataset_name modified_swiss_dwellings

CUDA_VISIBLE_DEVICES=0 python image_inference_msd_old.py --dataset modified_swiss_dwellings --batch_size 4 --set_name val --model_path /home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/scripts/ckpts/openai_2024_11_26_14_37_29_456830/ema_0.9999_080000.pt --path_struct "/home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/datasets/modified-swiss-dwellings-v2/test/struct_in" --dataset_name modified_swiss_dwellings --post_prefix ema_0.9999_080000 --gather_all_ids False

CUDA_VISIBLE_DEVICES=0 python image_inference_msd_old.py --dataset modified_swiss_dwellings --batch_size 4 --set_name val --model_path /home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/scripts/ckpts/openai_2024_12_09_21_17_08_698119/ema_0.9999_070000.pt --path_struct "/home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/datasets/modified-swiss-dwellings-v2/test/struct_in" --dataset_name modified_swiss_dwellings --post_prefix ema_0.9999_070000_v2 --gather_all_ids False
```

```bash
CUDA_VISIBLE_DEVICES=3 python image_inference_msd_old.py --dataset modified_swiss_dwellings --batch_size 4 --set_name val --model_path /home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/scripts/ckpts/openai_2024_12_10_08_23_47_351275/ema_0.9999_070000.pt --path_struct "/home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/datasets/modified-swiss-dwellings-v2/test/struct_in" --dataset_name modified_swiss_dwellings --post_prefix ema_0.9999_070000_layer_12 --gather_all_ids False

CUDA_VISIBLE_DEVICES=5 python image_inference_msd_old.py --dataset modified_swiss_dwellings --batch_size 4 --set_name val --model_path /home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/scripts/ckpts/openai_2024_12_10_08_23_47_351275/model070000.pt --path_struct "/home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/datasets/modified-swiss-dwellings-v2/test/struct_in" --dataset_name modified_swiss_dwellings --post_prefix model070000_layer_12 --gather_all_ids False

```

## 4. Sampling visualization from original house_diffusion repo

```bash
python image_sample.py --dataset modified_swiss_dwellings --batch_size 4 --set_name val --num_samples 4 --model_path ckpts_hpc/exp/model214000.pt --use_wall_self_attention true --save_gif true
```

