import copy
import functools
import os
import typing

import blobfile as bf
import torch as th
from torch.optim import AdamW

from house_diffusion.respace import SpacedDiffusion
from house_diffusion.rplanhg_datasets import load_rplanhg_structural_data
from house_diffusion.transformer import TransformerModel

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

import pandas as pd
import datetime
from tqdm import tqdm

# from image_inference_msd import HouseDiffusionInference, plot_predictions
from scripts.image_inference_msd import HouseDiffusionInference, plot_predictions

INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model: TransformerModel,
        diffusion: SpacedDiffusion,
        data: typing.Iterator[typing.Tuple[th.Tensor, typing.Dict[str, th.Tensor]]],
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        analog_bit=None,
        timeout=None,
        data_val: dict = {},
        test_interval=None,
        train_num_steps=None
    ):
        self.analog_bit = analog_bit
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size

        self.sync_cuda = th.cuda.is_available()

        timeout = pd.to_timedelta(timeout).to_pytimedelta() if timeout else None
        self.timeout_time = datetime.datetime.now() + timeout if timeout else None

        self.data_val = data_val
        self.test_interval = test_interval
        self.train_num_steps = train_num_steps

        if timeout:
            logger.log(f"Ending run after: {timeout}, at: {self.timeout_time}")
        else:
            logger.log("No timeout set")

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

    def update_lr(self):
        step = self.step + self.resume_step
        lr = self.lr * (0.1 ** (step // 100000))
        logger.log(f"Step {self.step} ({self.step + self.resume_step=}): Updating learning rate to {lr}")
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def run_loop(self):
        self.update_lr()

        while (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):  
            batch, cond = next(self.data)                               # torch.Size([48, 2, 289])
            self.run_step(batch, cond)                                  
            if (self.step + self.resume_step) % 100000 == 0:
                print(f"Current steps: {self.step}, update learning rate...")
                self.update_lr()
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.test_interval and self.step % self.test_interval == 0:
                self.test()
            if self.step % self.save_interval == 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
            
            if self.train_num_steps and self.step >= self.train_num_steps:
                logger.log("Reach total training steps, exiting...")
                self.save()  # Save final checkpoint before exiting
                return
            
            if self.timeout_time and datetime.datetime.now() > self.timeout_time:
                logger.log("Training time exceed timeout, exiting...")
                self.save()
                return

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch: th.Tensor, cond: typing.Dict[str, th.Tensor]):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch: th.Tensor, cond: typing.Dict[str, th.Tensor]):
        self.mp_trainer.zero_grad()
        for i in tqdm(range(0, batch.shape[0], self.microbatch), desc="Training Processing"):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())                      # 
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())      # house_diffusion.resample.UniformSampler

            losses = self.diffusion.training_losses(
                self.model, micro, t, model_kwargs=micro_cond, analog_bit=self.analog_bit   # self.analog_bit -> False
            )

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt" if rate else f"model{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
        opt_path = bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt")
        with bf.BlobFile(opt_path, "wb") as f:
            th.save(self.opt.state_dict(), f)

    def test(self):
        logger.log("Testing...")
        # for key, dl in self.data_val.items():
        for key, dl in tqdm(self.data_val.items(), desc="Testing Processing", unit="dataset"):
            data_sample_gt, model_kwargs = next(iter(dl))
            model = self.mp_trainer.model
            model.eval()
            inference = HouseDiffusionInference(self.model, self.diffusion)
            sample_and_gt = inference.sample_with_gt(data_sample_gt, model_kwargs)
            model.train()
            image_name = f"inference_example_{key}_{(self.step+self.resume_step):06d}.png"
            with bf.BlobFile(bf.join(get_blob_logdir(), image_name), "wb") as f:
                plot_predictions(sample_and_gt, f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
