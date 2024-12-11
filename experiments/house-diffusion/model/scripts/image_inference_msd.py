"""
Generate predictions and save them as images.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch as th
import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from house_diffusion.respace import SpacedDiffusion
from house_diffusion import dist_util, logger
from house_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)
from house_diffusion.modified_swiss_dwellings_housediffusion_dataset import (
    get_dataloader_modified_swiss_dwellings,
    gather_ids,
)
from house_diffusion import modified_swiss_dwellings_housediffusion_dataset
from house_diffusion.transformer import TransformerModel
from house_diffusion.plotting.plot_from_feats import plot_from_batch, draw_from_batch

import pickle


def create_argparser():
    defaults = dict(
        dataset='',
        clip_denoised=True,
        num_samples=10000,
        # batch_size=16,
        batch_size=4,
        use_ddim=False,
        model_path="/home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/scripts/ckpts/openai_2024_11_26_14_37_29_456830/ema_0.9999_080000.pt",
        override_use_augmentation=False,
        path_struct="/home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/datasets/modified-swiss-dwellings-v2/test/struct_in",
        # dataset_name=modified_swiss_dwellings_housediffusion_dataset.DEFAULT_DATASET_PATH,  # 'modified_swiss_dwellings_topojson_processing'
        # gather_all_ids=True,
        dataset_name='modified_swiss_dwellings',
        gather_all_ids=False,
        save_prefix="inference_msd",
        post_prefix="",
        set_name="val",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_model(args):
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()
    return model, diffusion


def load_dataset(args, todo_ids=None):
    return get_dataloader_modified_swiss_dwellings(
        batch_size=args.batch_size,
        set_name=args.set_name,
        override_use_augmentation=args.override_use_augmentation,
        override_shuffle=False,
        dataset_name=args.dataset_name,
        ids_list=todo_ids,
    )


def dict_to_device(d: dict, device) -> dict:
    return {key: value.to(device) for key, value in d.items()}


class HouseDiffusionInference:
    def __init__(self, model: TransformerModel, diffusion: SpacedDiffusion, use_ddim=False, clip_denoised=True, analog_bit=False) -> None:
        self.model = model
        self.diffusion = diffusion
        self.clip_denoised = clip_denoised
        self.analog_bit = analog_bit
        assert not use_ddim, "use_ddim is not supported"
        self.sample_fn = diffusion.p_sample_loop

    def _sample(self, shape, cond_kwargs) -> dict:
        with th.no_grad():
            cond_kwargs = dict_to_device(cond_kwargs, dist_util.dev())
            for key in cond_kwargs:
                cond_kwargs[key] = cond_kwargs[key].cuda()
            sample_dict = self.sample_fn(
                self.model,
                shape,
                clip_denoised=self.clip_denoised,
                model_kwargs=cond_kwargs,
                analog_bit=self.analog_bit,
                return_every_nth=10,
                return_dict=True
            )
            return sample_dict

    def sample_with_gt(self, data_sample_gt, model_kwargs):
        sample_dict = self._sample(data_sample_gt.shape, model_kwargs)
        sample = sample_dict["samples"]
        timesteps = sample_dict["timesteps"]
        sample_gt = data_sample_gt.unsqueeze(0)

        sample_and_gt = {
            "sample": sample.permute([0, 1, 3, 2]).cpu(),
            "timesteps": timesteps,
            "sample_gt": sample_gt.permute([0, 1, 3, 2]).cpu(),
            "model_kwargs": dict_to_device(model_kwargs, "cpu"),
            "id": model_kwargs["id"],
        }
        return sample_and_gt


def plot_predictions(sample_and_gt, file="inference_sample.png", dpi=100):
    batch_size = sample_and_gt["sample"].shape[1]
    fig, axs = plt.subplots(2, batch_size, figsize=(20, 10), dpi=dpi)
    sample = sample_and_gt["sample"]
    sample_gt = sample_and_gt["sample_gt"]
    model_kwargs = sample_and_gt["model_kwargs"]

    for i in range(batch_size):
        plot_from_batch(sample, model_kwargs, i, ax=axs[0][i])
        plot_from_batch(sample_gt, model_kwargs, i, ax=axs[1][i])
        axs[0][i].set_title(f"Prediction {model_kwargs['id'][i]}")
        axs[1][i].set_title(f"Ground Truth {model_kwargs['id'][i]}")

    fig.savefig(file)
    plt.close(fig)


def load_structural_img(id, path_struct):
    return np.load(os.path.join(path_struct, f"{id}.npy"))[:, :, 0].astype(np.uint8).T


def save_prediction_images(sample_and_gt, path_struct, submission_folder, submission_gt_folder, submission_with_structure):
    os.makedirs(submission_folder, exist_ok=True)
    os.makedirs(submission_gt_folder, exist_ok=True)
    os.makedirs(submission_with_structure, exist_ok=True)

    sample = sample_and_gt["sample"]
    sample_gt = sample_and_gt["sample_gt"]
    model_kwargs = sample_and_gt["model_kwargs"]

    for i in range(sample.shape[1]):
        id = model_kwargs["id"][i]
        try:
            pred_i = draw_from_batch(sample, model_kwargs, i, time_step=-1, draw_outline=False)
            pred_gt_i = draw_from_batch(sample_gt, model_kwargs, i, time_step=-1, draw_outline=False)
            structural_img = load_structural_img(id, path_struct)
            pred_structure_i = draw_from_batch(sample, model_kwargs, i, time_step=-1, draw_outline=False, structural_img=structural_img)

            Image.fromarray(pred_i.T).save(os.path.join(submission_folder, f'{id}.png'))
            Image.fromarray(pred_gt_i.T).save(os.path.join(submission_gt_folder, f'{id}.png'))
            Image.fromarray(pred_structure_i.T).save(os.path.join(submission_with_structure, f'{id}.png'))
        except Exception as e:
            print(f"Failed to save {id}: {e}")


def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)

    if args.analog_bit:
        raise NotImplementedError("Analog bit should be false")

    dist_util.setup_dist()
    logger.configure(log_suffix="_inference")
    logger.log_config(vars(args))

    logger.log("Creating model and diffusion...")
    model, diffusion = load_model(args)
    inference = HouseDiffusionInference(model, diffusion, args.use_ddim, args.clip_denoised, args.analog_bit)

    # args.set_name='val'
    save_prefix = f"{args.save_prefix}/{args.dataset_name}_{args.set_name}_{args.post_prefix}"             # 修改 save_dir name
    os.makedirs(save_prefix, exist_ok=True)

    if args.gather_all_ids:
        all_ids = gather_ids(f"../datasets/{args.dataset_name}/house_dicts")
    else:
        # ids_csv = f"../datasets/{args.dataset_name}/{args.set_name}_ids.csv"
        ids_csv = f"/home/mingyang/workspace/layout/floorplan/experiments/house-diffusion/model/datasets/{args.dataset_name}/{args.set_name}_ids.csv"
        all_ids = pd.read_csv(ids_csv, header=None).values.flatten().tolist()

    if os.path.exists(save_prefix):
        done_ids = gather_ids(save_prefix)
        todo_ids = sorted(set(all_ids) - set(done_ids))
    else:
        todo_ids = all_ids

    data = load_dataset(args, todo_ids)
    for i, (data_sample_gt, model_kwargs) in enumerate(data):
        sample_and_gt = inference.sample_with_gt(data_sample_gt, model_kwargs)

        with open(f"{save_prefix}/sample_and_gt_{i}.pkl", "wb") as f:
            pickle.dump(sample_and_gt, f)

        save_prediction_images(
            sample_and_gt,
            args.path_struct,
            submission_folder=f"{save_prefix}/submission",
            submission_gt_folder=f"{save_prefix}/submission_gt",
            submission_with_structure=f"{save_prefix}/submission_with_structure"
        )


if __name__ == "__main__":
    main()
