import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from omegaconf import DictConfig
import hydra
import time
import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any
from modules.utils.correct_checkpoints import correct_checkpoint_keys

class LightningModel(pl.LightningModule):
    """A PyTorch Lightning wrapper for non-lightning models.
    
    This class wraps a regular PyTorch model to make it compatible with PyTorch Lightning's
    training infrastructure. It handles the training loop, validation, and testing steps.
    
    Args:
        model (torch.nn.Module): The PyTorch model to wrap
        cfg (DictConfig): Configuration for the model, optimizer, and scheduler
    """
    
    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg

        # Instantiate the loss function from the configuration
        self.criterion = hydra.utils.instantiate(cfg.training.loss)

        # Instantiate metrics for training, validation, and testing phases
        for metric_name, metric_cfg in self.cfg.training.metrics.items():
            train_metric = hydra.utils.instantiate(metric_cfg)
            val_metric = hydra.utils.instantiate(metric_cfg)
            test_metric = hydra.utils.instantiate(metric_cfg)
            setattr(self, f"train_{metric_name}", train_metric)
            setattr(self, f"val_{metric_name}", val_metric)
            setattr(self, f"test_{metric_name}", test_metric)

        self.save_hyperparameters(cfg)

    def forward(self, batch_x):
        return self.model(batch_x)

    def step(self, batch, batch_idx, phase):
        if isinstance(batch, dict):
            batch_x, batch_y = batch["x"], batch["y"]
        else:
            batch_x, batch_y = batch
        if self.cfg.training.get("task", "forecasting") == "forecasting":
            # forecasting task
            batch_y_hat = self(batch_x)

            if phase == "train":
                loss_len = min(batch_y_hat.shape[-2], (self.cfg.dataset.pred_len + self.cfg.dataset.label_len))
                batch_y_hat = batch_y_hat[:, -loss_len:, :]
                batch_y = batch_y[:, -loss_len:, :]
            else: # val or test
                batch_y_hat = batch_y_hat[:, -self.cfg.dataset.pred_len :, :]
                batch_y = batch_y[:, -self.cfg.dataset.pred_len :, :]

            loss = self.criterion(batch_y_hat, batch_y)
        elif self.cfg.training.get("task", "forecasting") == "classification":
            # classification task
            batch_logits = self(batch_x)
            batch_logprob = F.log_softmax(batch_logits, dim=1)
            batch_y_hat = torch.argmax(batch_logprob, dim=-1)
            
            loss = self.criterion(batch_logprob, batch_y)
        else:
            # regression task
            batch_y_hat = self(batch_x) # This one expects outs of shape (B, out_n), not squeezed, unlike the classificaiton tasks
            loss = self.criterion(batch_y_hat, batch_y.float())

        # logging metrics
        for metric_name in self.cfg.training.metrics.keys():
            metric = getattr(self, f"{phase}_{metric_name}")
            metric_value = metric(torch.flatten(batch_y_hat.detach()), torch.flatten(batch_y.detach()))
            self.log(
                f"{phase}_{metric_name}",
                metric_value,
                on_step= phase == "train",
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        self.log(f"{phase}_loss", loss.detach(), on_step= phase == "train", on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, phase='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, phase='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        # Instantiate optimizer: separating weight decay for bias and non-bias parameters
        if self.cfg.training.get("bias_weights_decay_zero", False) and ("weight_decay" in self.cfg.training.optimizer) and self.cfg.training.optimizer.weight_decay:
            # Create parameter groups: non-bias parameters get the specified weight decay,
            # while bias parameters get a weight decay of 0.
            param_groups = [
                {"params": [p for n, p in self.model.named_parameters() if "bias" not in n]},
                {"params": [p for n, p in self.model.named_parameters() if "bias" in n], "weight_decay": 0.0},
            ]
            optimizer = hydra.utils.instantiate(self.cfg.training.optimizer, param_groups)
        else:
            optimizer = hydra.utils.instantiate(self.cfg.training.optimizer, params=self.model.parameters())
        
        # Instantiate scheduler
        scheduler_cfg = self.cfg.training.scheduler
        scheduler = self.instantiate_scheduler(scheduler_cfg, optimizer)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        else:
            return [optimizer], [scheduler]

    def instantiate_scheduler(self, scheduler_cfg, optimizer):
        if isinstance(scheduler_cfg, DictConfig) and '_target_' in scheduler_cfg:
            if 'schedulers' in scheduler_cfg:
                schedulers = [
                    self.instantiate_scheduler(s_cfg, optimizer) for s_cfg in scheduler_cfg.schedulers
                ]
                kwargs = {k: v for k, v in scheduler_cfg.items() if k not in ['_target_', 'schedulers']}
                scheduler_class = hydra.utils.get_class(scheduler_cfg._target_)
                return scheduler_class(optimizer, schedulers=schedulers, **kwargs)
            else:
                kwargs = {k: v for k, v in scheduler_cfg.items() if k != '_target_'}
                scheduler_class = hydra.utils.get_class(scheduler_cfg._target_)
                return scheduler_class(optimizer, **kwargs)
        else:
            raise ValueError("Invalid scheduler configuration")

    def on_epoch_end(self):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        gc.collect()
    
    def on_before_optimizer_step(self, optimizer):
        # log if asked
        if self.cfg.training.get("verbose", False):
            # Split parameters into bias and non-bias groups
            bias_params = []
            nonbias_params = []
            for name, param in self.model.named_parameters():
                if "bias" in name:
                    bias_params.append(param)
                else:
                    nonbias_params.append(param)

            # Log parameter statistics for bias parameters
            if bias_params:
                all_bias_params = torch.nn.utils.parameters_to_vector(bias_params)
                self.log("train_parambias_mean", all_bias_params.mean(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log("train_parambias_std", all_bias_params.std(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            # Log parameter statistics for non-bias parameters
            if nonbias_params:
                all_nonbias_params = torch.nn.utils.parameters_to_vector(nonbias_params)
                self.log("train_paramnonbias_mean", all_nonbias_params.mean(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log("train_paramnonbias_std", all_nonbias_params.std(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            # Split gradients into bias and non-bias groups.
            # Note: Only include parameters that have a gradient (p.grad is not None).
            bias_grads = []
            nonbias_grads = []
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if "bias" in name:
                        bias_grads.append(param.grad)
                    else:
                        nonbias_grads.append(param.grad)

            # Log gradient statistics for bias parameters
            if bias_grads:
                all_bias_grads = torch.nn.utils.parameters_to_vector(bias_grads)
                self.log("train_gradbias_mean", all_bias_grads.mean(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log("train_gradbias_std", all_bias_grads.std(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

            # Log gradient statistics for non-bias parameters
            if nonbias_grads:
                all_nonbias_grads = torch.nn.utils.parameters_to_vector(nonbias_grads)
                self.log("train_gradnonbias_mean", all_nonbias_grads.mean(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                self.log("train_gradnonbias_std", all_nonbias_grads.std(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True) 

def lightning_train_model(model: nn.Module, cfg: DictConfig, train_loader, val_loader, test_loader=None) -> Dict[str, Any]:
    """
    Train a model using the provided configuration and data loaders.
    
    Args:
        model: The PyTorch model to train
        cfg: Configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)
        
    Returns:
        Dictionary containing training results
    """
    logger = logging.getLogger("Train")
    
    # Create Lightning model wrapper
    lightning_model = LightningModel(model=model, cfg=cfg)
    
    # Compile model if requested
    if cfg.training.get("compile", False):
        lightning_model = torch.compile(lightning_model)
    
    # Create callbacks
    callbacks = []
    if "callbacks" in cfg.training:
        for cb_cfg in cfg.training.callbacks.values():
            callback = hydra.utils.instantiate(cb_cfg)
            callbacks.append(callback)
    
    # Create loggers
    loggers = [
        hydra.utils.instantiate(logger_cfg) for logger_cfg in cfg.training.loggers.values()
    ]
    
    # Create trainer
    trainer_cfg = cfg.training.trainer.copy()
    if cfg.training.device == "cpu" or not torch.cuda.is_available():
        trainer_cfg.devices = 1
        trainer_cfg.accelerator = "cpu"
    
    trainer = hydra.utils.instantiate(
        trainer_cfg,
        callbacks=callbacks,
        logger=loggers,
    )
    
    # Setup checkpoint paths
    checkpoint_dir = Path(cfg.training.callbacks.model_checkpoint.dirpath)
    zero_ckpt_path = checkpoint_dir / "zero.ckpt"
    last_ckpt_path = checkpoint_dir / "last.ckpt"
    best_ckpt_path = checkpoint_dir / "best.ckpt"
    
    # Only main process saves checkpoints
    is_main_process = int(os.environ.get("SLURM_PROCID", 0)) == 0
    if is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.strategy._lightning_module = lightning_model
        trainer.save_checkpoint(str(zero_ckpt_path))
        logger.info(f"Saved initial model state at {zero_ckpt_path}")
        
    # Check if last checkpoint exists
    ckpt_path = str(last_ckpt_path) if last_ckpt_path.exists() else None
    if ckpt_path:
        logger.info(f"Resuming training from {ckpt_path}")
    else:
        logger.info("No existing checkpoint found. Starting training from scratch.")
    
    # Start training
    if (train_loader is None):
        logger.info("Dataset percentage of 0, skipping training.")
        if is_main_process:
            shutil.copy(str(zero_ckpt_path), str(best_ckpt_path))
            shutil.copy(str(zero_ckpt_path), str(last_ckpt_path))
    else:
        logger.info("Starting training...")
        trainer.fit(
            lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )
    
    # Testing
    logger.info("Testing the model using the checkpoints")
    test_dataloader = test_loader if test_loader is not None else val_loader
    results_zero = trainer.test(dataloaders=test_dataloader, ckpt_path=str(zero_ckpt_path))
    
    if (train_loader is not None):
        results_last = trainer.test(dataloaders=test_dataloader, ckpt_path='last')
        results_best = trainer.test(dataloaders=test_dataloader, ckpt_path='best')
    else:
        results_last = results_zero
        results_best = results_zero
    
    # Cleanup
    if is_main_process:
        logger.info("Correcting checkpoint keys")
        if "model_checkpoint" in cfg.training.callbacks:
            del trainer
            del val_loader
            del test_loader
            del lightning_model
            del model
            correct_checkpoint_keys(f"{cfg.training.callbacks.model_checkpoint.dirpath}/best.ckpt")
            correct_checkpoint_keys(f"{cfg.training.callbacks.model_checkpoint.dirpath}/last.ckpt")
            correct_checkpoint_keys(f"{cfg.training.callbacks.model_checkpoint.dirpath}/zero.ckpt")
        
        logger.info("Training completed successfully")
        return {
            "zero": results_zero[0] if len(results_zero) > 0 else None,
            "best": results_best[0],
            "last": results_last[0]
        }
    return None
    
