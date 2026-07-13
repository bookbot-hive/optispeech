"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""

import inspect
from abc import ABC
from typing import Any, Dict

import numpy as np
import torch
import torchaudio
from lightning import LightningModule

from optispeech.utils import get_pylogger, plot_attention, plot_tensor
from optispeech.utils.segments import get_segments, get_segments_numpy

log = get_pylogger(__name__)


def scheduler_steps_per_optimizer(max_steps: int) -> int:
    """Return the scheduler horizon for each of the two manual optimizers.

    Lightning's ``global_step`` advances when an optimizer actually steps. The
    configured trainer limit is therefore already expressed after manual
    gradient accumulation; dividing by the accumulation factor again would
    make the cosine schedule finish too early.
    """
    if max_steps <= 0:
        raise ValueError("trainer.max_steps must be a positive integer")
    return max_steps // 2


def gradient_accumulation_state(
    batch_idx: int, accumulation_batches: int | None, is_last_batch: bool
) -> tuple[float, bool]:
    """Return loss divisor and whether to update for this microbatch.

    The final accumulation group may contain fewer microbatches than requested.
    Stepping it at the epoch boundary prevents gradients leaking into the next
    epoch, while scaling by its actual size keeps the gradient magnitude stable.
    """
    if accumulation_batches is None:
        return 1.0, True
    if accumulation_batches < 1:
        raise ValueError("gradient_accumulate_batches must be at least 1")
    group_position = (batch_idx % accumulation_batches) + 1
    should_apply_gradients = group_position == accumulation_batches or is_last_batch
    divisor = group_position if is_last_batch else accumulation_batches
    return float(divisor), should_apply_gradients


class BaseLightningModule(LightningModule, ABC):
    def _process_batch(self, batch):
        sids = batch["sids"]
        lids = batch["lids"]
        gen_outputs = self.generator(
            x=batch["x"].to(self.device),
            x_lengths=batch["x_lengths"].to(self.device),
            mel=batch["mel"].to(self.device),
            mel_lengths=batch["mel_lengths"].to(self.device),
            pitches=batch["pitches"].to(self.device),
            energies=batch["energies"].to(self.device),
            sids=sids.to(self.device) if sids is not None else None,
            lids=lids.to(self.device) if lids is not None else None,
        )
        segment_size = gen_outputs["segment_size"]
        seg_gt_wav = get_segments_numpy(
            x=np.expand_dims(batch["wav"], 1),
            start_idxs=gen_outputs["start_idx"] * self.hop_length,
            segment_size=segment_size * self.hop_length,
        )
        seg_gt_wav = torch.from_numpy(seg_gt_wav.squeeze(1)).type_as(gen_outputs["wav_hat"])
        gen_outputs["wav"] = seg_gt_wav
        return gen_outputs

    def configure_optimizers(self):
        gen_params = [
            {"params": self.generator.parameters()},
        ]
        disc_params = [
            {"params": self.discriminator.parameters()},
        ]
        opt_gen = self.hparams.optimizer(gen_params)
        opt_disc = self.hparams.optimizer(disc_params)

        # Lightning counts actual manual optimizer steps in global_step. Since
        # generator and discriminator alternate, each receives about half of
        # trainer.max_steps regardless of gradient accumulation.
        max_steps = scheduler_steps_per_optimizer(self.trainer.max_steps)

        if "num_training_steps" in self.hparams.scheduler.keywords:
            self.hparams.scheduler.keywords["num_training_steps"] = max_steps
        scheduler_gen = self.hparams.scheduler(opt_gen, last_epoch=getattr("self", "ckpt_loaded_epoch", -1))
        scheduler_disc = self.hparams.scheduler(opt_disc, last_epoch=getattr("self", "ckpt_loaded_epoch", -1))
        return (
            [opt_gen, opt_disc],
            [{"scheduler": scheduler_gen, "interval": "step"}, {"scheduler": scheduler_disc, "interval": "step"}],
        )

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.trainer.max_epochs - 1:
            # Workaround to always save the last epoch until the bug is fixed in lightning (https://github.com/Lightning-AI/lightning/issues/4539)
            self.trainer.check_val_every_n_epoch = 1

    def training_step(self, batch, batch_idx, **kwargs):
        # manual gradient accumulation
        gradient_accumulate_batches = self.train_args.gradient_accumulate_batches
        loss_scaling_factor, should_apply_gradients = gradient_accumulation_state(
            batch_idx=batch_idx,
            accumulation_batches=gradient_accumulate_batches,
            is_last_batch=self.trainer.is_last_batch,
        )
        # Generator pretraining
        train_discriminator = self.global_step >= self.train_args.pretraining_steps
        # Extract generator/discriminator optimizer/scheduler
        opt_g, opt_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()
        # train generator
        self.toggle_optimizer(opt_g)
        loss_g, wav_outputs = self.training_step_g(batch, train_discriminator=train_discriminator)
        # Scale (grad accumulate)
        loss_g /= loss_scaling_factor
        self.manual_backward(loss_g)
        if should_apply_gradients:
            self.clip_gradients(
                opt_g, gradient_clip_val=self.train_args.gradient_clip_val, gradient_clip_algorithm="norm"
            )
            opt_g.step()
            sched_g.step()
            opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)
        # train discriminator
        if not train_discriminator:
            # we're still in pretraining
            return
        self.toggle_optimizer(opt_d)
        if not self.train_args.cache_generator_outputs:
            wav_outputs = None
        loss_d = self.training_step_d(batch, wav_outputs=wav_outputs)
        # Scale (grad accumulate)
        loss_d /= loss_scaling_factor
        self.manual_backward(loss_d)
        if should_apply_gradients:
            self.clip_gradients(
                opt_d, gradient_clip_val=self.train_args.gradient_clip_val, gradient_clip_algorithm="norm"
            )
            opt_d.step()
            sched_d.step()
            opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

    def training_step_g(self, batch, train_discriminator):
        log_outputs = {}
        gen_outputs = self._process_batch(batch)
        gen_am_loss = gen_outputs["loss"]
        log_outputs.update(
            {
                "total_loss/train_am_loss": gen_am_loss.item(),
                "gen_subloss/train_alighn_loss": gen_outputs["align_loss"].item(),
                "gen_subloss/train_duration_loss": gen_outputs["duration_loss"].item(),
                "gen_subloss/train_pitch_loss": gen_outputs["pitch_loss"].item(),
                "gen_subloss/train_energy_loss": gen_outputs["energy_loss"].item(),
            }
        )
        wav, wav_hat = gen_outputs["wav"], gen_outputs["wav_hat"]
        if train_discriminator:
            gen_adv_loss, log_dict = self.discriminator.forward_gen(wav, wav_hat)
            log_outputs["total_loss/train_gen_adv_loss"] = gen_adv_loss.item()
            log_outputs.update({
            f"gen_adv_loss/train_{key}": value.item() if isinstance(value, torch.Tensor) else value
                for (key, value) in log_dict.items()
            })
        else:
            gen_adv_loss = 0.0
        loss = gen_am_loss + gen_adv_loss
        log_outputs["total_loss/generator"] = loss.item()
        log_dict = {}
        for (name, value) in log_outputs.items():
            if isinstance(value, torch.Tensor):
                log_dict[name] = value.detach().cpu()
            else:
                log_dict[name] = value
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.data_args.batch_size,
        )
        return loss, (wav.detach(), wav_hat.detach())

    def training_step_d(self, batch, wav_outputs=None):
        log_outputs = {}
        if wav_outputs is None:
            # Don't train generator in discriminator's turn
            with torch.no_grad():
                gen_outputs = self._process_batch(batch)
            wav, wav_hat = gen_outputs["wav"], gen_outputs["wav_hat"]
        else:
            wav, wav_hat = wav_outputs
        loss, log_dict = self.discriminator.forward_disc(wav, wav_hat)
        log_outputs["total_loss/discriminator"] = loss.item()
        log_outputs.update({
            f"discriminator/{key}": value.item() if isinstance(value, torch.Tensor) else value
            for key, value in log_dict.items()
        })
        log_dict = {}
        for (name, value) in log_outputs.items():
            if isinstance(value, torch.Tensor):
                log_dict[name] = value.detach().cpu()
            else:
                log_dict[name] = value
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.data_args.batch_size,
        )
        return loss

    def on_validation_epoch_start(self):
        if self.train_args.evaluate_utmos:
            from optispeech.vendor.metrics.UTMOS import UTMOSScore

            if not hasattr(self, "utmos_model"):
                self.utmos_model = UTMOSScore(device=self.device)

    def validation_step(self, batch, batch_idx, **kwargs):
        log_outputs = {}
        gen_outputs = self._process_batch(batch)
        gen_am_loss = gen_outputs["loss"]
        log_outputs.update(
            {
                "total_loss/val_am_loss": gen_outputs["loss"].item(),
                "gen_subloss/val_alighn_loss": gen_outputs["align_loss"].item(),
                "gen_subloss/val_duration_loss": gen_outputs["duration_loss"].item(),
                "gen_subloss/val_pitch_loss": gen_outputs["pitch_loss"].item(),
                "gen_subloss/val_energy_loss": gen_outputs["energy_loss"].item()
            }
        )
        wav, wav_hat = gen_outputs["wav"], gen_outputs["wav_hat"]
        gen_adv_loss, log_dict = self.discriminator.get_val_loss(wav, wav_hat)
        log_outputs["total_loss/val_gen_adv_loss"] = gen_adv_loss.item()
        for key, value in log_dict.items():
            scalar = value.item() if isinstance(value, torch.Tensor) else value
            log_outputs[f"val/{key}"] = scalar
        # perceptual eval
        audio_16_khz = torchaudio.functional.resample(wav, orig_freq=self.sample_rate, new_freq=16000)
        audio_hat_16khz = torchaudio.functional.resample(wav_hat, orig_freq=self.sample_rate, new_freq=16000)
        if self.train_args.evaluate_periodicity:
            from optispeech.vendor.metrics.periodicity import calculate_periodicity_metrics

            periodicity_loss, perio_pitch_loss, f1_score = calculate_periodicity_metrics(audio_16_khz, audio_hat_16khz)
            log_outputs.update(
                {
                    "val/periodicity_loss": periodicity_loss.item(),
                    "val/perio_pitch_loss": perio_pitch_loss.item(),
                    "val/f1_score": f1_score.item(),
                }
            )
        if self.train_args.evaluate_utmos:
            utmos_score = self.utmos_model.score(audio_hat_16khz.unsqueeze(1)).mean()
            utmos_loss = 5 - utmos_score
            log_outputs["val/utmos_score"] = (
                utmos_score.item() if isinstance(utmos_score, torch.Tensor) else float(utmos_score)
            )
            log_outputs["val/utmos_loss"] = (
                utmos_loss.item() if isinstance(utmos_loss, torch.Tensor) else float(utmos_loss)
            )
        else:
            utmos_loss = torch.zeros(1, device=self.device)
        if self.train_args.evaluate_pesq:
            from pesq import pesq

            pesq_score = 0.0
            for ref, deg in zip(audio_16_khz.cpu().numpy(), audio_hat_16khz.cpu().numpy()):
                pesq_score += pesq(16000, ref, deg, "wb", on_error=1)
            pesq_score /= len(audio_16_khz)
            pesq_loss = 5 - pesq_score
            log_outputs["val/pesq_score"] = float(pesq_score)
            log_outputs["val/pesq_loss"] = float(pesq_loss)
        else:
            pesq_loss = torch.zeros(1, device=self.device)
        if self.train_args.evaluate_mcd:
            from optispeech.vendor.metrics.mcd import calculate_mcd

            mcd = calculate_mcd(wav, wav_hat, sample_rate=self.sample_rate)
            log_outputs["val/mcd"] = float(mcd)
        total_loss = gen_am_loss + gen_adv_loss + utmos_loss + pesq_loss
        log_outputs["total_loss/val_total"] = total_loss.item()
        log_dict = {}
        for (name, value) in log_outputs.items():
            if isinstance(value, torch.Tensor):
                log_dict[name] = value.detach().cpu()
            else:
                log_dict[name] = value
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.data_args.batch_size,
        )

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    gt_wav = one_batch["wav"][i].squeeze()
                    self.logger.experiment.add_audio(
                        f"wav/original_{i}", gt_wav, self.global_step, self.sample_rate
                    )
                    mel = one_batch["mel"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"mel/original_{i}",
                        plot_tensor(mel.detach().squeeze().float().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )
            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                synth_out = self.generator.synthesise(x=x[:, :x_lengths], x_lengths=x_lengths)
                wav_hat = synth_out["wav"].squeeze().float().detach().cpu().numpy()
                mel_hat = self.data_args.feature_extractor.get_mel(wav_hat)
                self.logger.experiment.add_image(
                    f"mel/generated_{i}",
                    plot_tensor(mel_hat.squeeze()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_audio(f"wav/generated_{i}", wav_hat, self.global_step, self.sample_rate)

    def test_step(self, batch, batch_idx):
        pass

    def on_before_optimizer_step(self, optimizer):
        interval = self.train_args.get("gradient_log_interval", 100)
        if not interval or self.global_step % interval != 0:
            return

        gradients = [
            parameter.grad.detach().norm(2)
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if not gradients:
            return

        total_norm = torch.stack(gradients).norm(2)
        generator_parameters = {id(parameter) for parameter in self.generator.parameters()}
        first_parameter = next(parameter for group in optimizer.param_groups for parameter in group["params"])
        optimizer_name = "generator" if id(first_parameter) in generator_parameters else "discriminator"
        self.log(
            f"grad_norm/{optimizer_name}_total",
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed with respect to `gradient_accumulate_batches`
        """
        if self.train_args.gradient_accumulate_batches is not None:
            global_step = self.trainer.fit_loop.total_batch_idx // self.train_args.gradient_accumulate_batches
        else:
            global_step = self.trainer.fit_loop.total_batch_idx
        return int(global_step)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init
