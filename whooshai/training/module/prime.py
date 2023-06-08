import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class EarlyStopping:
    """
    Can be used to control early stopping with a Trainer class. Any object can be used instead which
    implements the method check_stopping and and provides the attribute save_dir
    """

    def __init__(
        self,
        head=0,
        metric="loss",
        save_dir=None,
        mode="min",
        patience=0,
        min_delta=0.001,
        min_evals=0,
    ):
        """
        :param head: the prediction head referenced by the metric.
        :param save_dir: the directory where to save the final best model, if None, no saving.
        :param metric: name of dev set metric to monitor (default: loss) to get extracted from the 0th head or
                       a function that extracts a value from the trainer dev evaluation result.
                       NOTE: this is different from the metric to get specified for the processor which defines how
                       to calculate one or more evaluation matric values from prediction/target sets, while this
                       specifies the name of one particular such metric value or a method to calculate that value
                       from the result returned from a processor metric.
        :param mode: "min" or "max"
        :param patience: how many evaluations to wait after the best evaluation to stop
        :param min_delta: minimum difference to a previous best value to count as an improvement.
        :param min_evals: minimum number of evaluations to wait before using eval value
        """
        self.head = head
        self.metric = metric
        self.save_dir = save_dir
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.min_evals = min_evals
        self.eval_values = []  # for more complex modes
        self.n_since_best = None
        if mode == "min":
            self.best_so_far = 1.0e99
        elif mode == "max":
            self.best_so_far = -1.0e99
        else:
            raise Exception("Mode must be 'min' or 'max'")

    def check_stopping(self, eval_result):
        """
        Provide the evaluation value for the current evaluation. Returns true if stopping should occur.
        This will save the model, if necessary.

        :param eval: the current evaluation result
        :return: a tuple (stopprocessing, savemodel, evalvalue) indicating if processing should be stopped
                 and if the current model should get saved and the evaluation value used.
        """

        if isinstance(self.metric, str):
            eval_value = eval_result[self.head][self.metric]
        else:
            eval_value = self.metric(eval_result)
        self.eval_values.append(float(eval_value))
        stopprocessing, savemodel = False, False
        if len(self.eval_values) <= self.min_evals:
            return stopprocessing, savemodel, eval_value
        if self.mode == "min":
            delta = self.best_so_far - eval_value
        else:
            delta = eval_value - self.best_so_far
        if delta > self.min_delta:
            self.best_so_far = eval_value
            self.n_since_best = 0
            if self.save_dir:
                savemodel = True
        else:
            self.n_since_best += 1
        if self.n_since_best > self.patience:
            stopprocessing = True
        return stopprocessing, savemodel, eval_value


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        adj: Dict,
        features: Dict,
        labels: torch.Tensor,
        optimizer,
        epochs: int,
        device: str,
        log_loss_every: int = 1,
        lr_schedule=None,
        evaluate_every=100,
        eval_report=True,
        use_amp=None,
        grad_acc_steps=1,
        checkpoint_every=None,
        checkpoint_root_dir=None,
        checkpoints_to_keep=3,
        logger: Optional = None,
        from_epoch=0,
        from_step=0,
        global_step=0,
        evaluator_test=True,
        disable_tqdm=False,
        max_grad_norm=1.0,
    ):
        self.model = model
        self.adj = adj
        self.features = features
        self.labels = labels
        self.device = device

        self.epochs = epochs
        self.from_epoch = from_epoch
        self.from_step = from_step
        self.optimizer = optimizer

    def train(self, pathM=None, mode: str = "GAT", labels=None, idx_train=None, idx_val=None, idx_test=None):
        device = self.device
        features = self.features.to(device)
        adj = self.adj.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)

        best_score, best_epoch = 0.0, -1

        for epoch in range(self.from_epoch, self.epochs):
            _, loss_value = self.step_per_epoch(
                epoch=epoch, features=features, adj=adj, labels=labels, idx_train=idx_train, mode=mode
            )

            score = self.test_per_epoch(idx=idx_test, epoch=epoch, labels=labels, with_path=True)
            if score > best_score:
                best_score, best_epoch = _, epoch
                torch.save(self.model.state_dict(), str(Path(os.getcwd()) / "weights" / "zaeleillaep.pkl"))
            print(f"Epoch {str(epoch)} / {str(self.epochs)} ### {str(loss_value)} --- {str(_)} --- {str(score)}")

        print(f"Best @ {str(best_epoch)} ### {str(best_score)}")

    def step_per_epoch(self, epoch, features, adj, labels, idx_train, pathM: str = None, mode: str = "GAT"):
        time.time()
        model = self.model.train()
        self.optimizer.zero_grad()
        output = model(self.features, self.adj, pathM=pathM, mode=mode)
        preds, targets = output[idx_train], labels[idx_train]
        loss = F.nll_loss(preds, targets)
        acc = accuracy(preds, labels=targets)
        loss.backward()
        self.optimizer.step()
        # TODO: update monitor
        time.time()
        return acc.item(), loss.item()

    def test_per_epoch(self, labels, epoch=0, idx=None, pathM: str = None, with_path=False, mode: str = "GAT"):
        model = self.model.eval()
        output = model(self.features, self.adj, pathM=pathM, genPath=with_path, mode=mode)
        # loss_test = F.nll_loss(output[idx], labels[idx])

        acc_test = accuracy(output[idx], labels[idx])

        return acc_test.item()
