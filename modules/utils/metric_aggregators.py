import torch
import torchmetrics
from typing import Dict, List, Any, Optional
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import Logger

class MetricAggregator:
    def __init__(self, num_classes: Optional[int] = None, phases: List[str] = ["train"], metrics: List[str] = ["acc", "cm", "f1"], device: str = "cpu", loggers: Optional[List[Logger]] = None, task='multiclass') -> None:
        self.metrics = metrics
        self.aggregators: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.device = device
        self.num_classes = num_classes
        self.step_num = 1
        self.loggers = loggers if loggers is not None else []
        if task == 'classification': self.task = 'multiclass'
        elif task == 'multilabel classification': self.task = 'multilabel'
        for phase in phases:
            self.aggregators[phase] = {}
            self.results[phase] = {}

    def init_agg(self, phase: str = "train", metric: str = "acc") -> None:
        if phase not in self.aggregators:
            self.aggregators[phase] = {}
            self.results[phase] = {}
        if metric == "acc":
            if self.num_classes is not None:
                self.aggregators[phase][metric] = torchmetrics.Accuracy(task=self.task, num_classes=self.num_classes, num_labels=self.num_classes).to(self.device)
        elif metric == "cm":
            if self.num_classes is not None:
                self.aggregators[phase][metric] = torchmetrics.ConfusionMatrix(task=self.task, num_classes=self.num_classes, num_labels=self.num_classes).to(self.device)
        elif metric == "f1":
            if self.num_classes is not None:
                self.aggregators[phase][metric] = torchmetrics.F1Score(task=self.task, num_classes=self.num_classes, num_labels=self.num_classes).to(self.device)
        elif metric == "r2":
            if self.num_classes is not None:
                self.aggregators[phase][metric] = torchmetrics.R2Score(num_outputs=self.num_classes).to(self.device)
        elif metric == "mse":
            if self.num_classes is not None:
                self.aggregators[phase][metric] = torchmetrics.MeanSquaredError(num_outputs=self.num_classes).to(self.device)
        elif metric == "mae":
            if self.num_classes is not None:
                self.aggregators[phase][metric] = torchmetrics.MeanAbsoluteError().to(self.device)
        else:
            self.aggregators[phase][metric] = torchmetrics.MeanMetric().to(self.device)

    def step(self, y_hat: torch.Tensor, y: torch.Tensor, phase: str = "train", **kwargs: Any) -> None:
        if phase not in self.aggregators:
            self.aggregators[phase] = {}
            self.results[phase] = {}
        for metric in list(kwargs.keys()) + self.metrics:
            if metric not in self.aggregators[phase]:
                self.init_agg(phase=phase, metric=metric)
        for metric in ["acc", "cm", "f1", "mse", "mae", "r2"]:
            if metric in self.metrics:
                self.aggregators[phase][metric](y_hat.to(self.device), y.to(self.device))
        for k, v in kwargs.items():
            if k in self.aggregators[phase]:
                self.aggregators[phase][k](v.to(self.device))
        for logger in self.loggers:
            logger.log_metrics({f"{phase}_{k}_step":v.detach().cpu().tolist() for k,v in kwargs.items()}, step=self.step_num)
        self.step_num+=1

    def compute(self, phase: str = "train") -> Dict[str, Any]:
        for k in self.aggregators[phase].keys():
            self.results[phase][f"{phase}_{k}_epoch"] = self.aggregators[phase][k].compute().detach().cpu().tolist()
        for logger in self.loggers:
            logger.log_metrics({f"{phase}_{k}_epoch":self.results[phase][f"{phase}_{k}_epoch"] for k in self.aggregators[phase].keys()}, step=self.step_num)
        return self.results[phase]

    def reset(self, phase: str = "train") -> None:
        for k in self.aggregators[phase].keys():
            self.aggregators[phase][k].reset()
    
    def log(self):
        """
        Log the computed metrics to TensorBoard.

        This method computes the metrics for the current phase and logs them to TensorBoard.
        It iterates through each metric and logs its value under the corresponding tag.
        """
        results = self.compute()  # Compute the metrics for the current phase

        # Log to TensorBoard
        if self.logger_tensorboard:
            # The tag for each metric is its name, and the values are filtered from the results
            metrics_dict = {k:v for k,v in results.items() if "_cm_" not in k}
            self.logger_tensorboard.log_metrics(metrics_dict, step=self.current_step)
            self.logger_csv.log_metrics(metrics_dict, step=self.current_step)

        return results
    
    def close(self):
        """
        Close the TensorBoard writer.

        This method should be called at the end of the training or validation process.
        It ensures that all pending events have been written to disk and properly closes the writer.
        """
        if self.logger_tensorboard:
            # Close the TensorBoard writer if it exists
            self.logger_tensorboard.save()
            self.logger_tensorboard.finalize("success")
            self.logger_csv.save()
            self.logger_csv.finalize("success")

    def set_early_stopping(self, patience: int):
        """
        Set the early stopping parameters.

        Args:
            patience (int): Number of epochs to wait after last time the monitored metric improved.
        """
        self.early_stopping_patience = patience
        self.early_stopping_best = float('inf')
        self.early_stopping_counter = 0

    def check_early_stopping(self, current_metric_value: float) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            current_metric_value (float): current value of the early stopping metric.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        if current_metric_value < self.early_stopping_best:
            self.early_stopping_best = current_metric_value
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        if self.early_stopping_counter >= self.early_stopping_patience:
            return True  # Stop training
        return False  # Continue training