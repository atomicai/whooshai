import logging
import os
import pathlib
from typing import Optional

import wandb
from whooshai.logging.mask import ILoggerMask

logger = logging.getLogger(__name__)


class IWBLogger(ILoggerMask):
    """
    Weights and biases logger. See <a href="https://docs.wandb.ai">here</a> for more details.
    """

    experiment = None
    save_dir = str(pathlib.Path(os.getcwd()) / ".wandb")
    offset_step = 0
    sync_step = True
    prefix = ""
    log_checkpoint = False

    @classmethod
    def init_experiment(
        cls,
        experiment_name,
        project_name,
        api: Optional[str] = None,
        notes=None,
        tags=None,
        entity=None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        _id: Optional[str] = None,
        log_checkpoint: Optional[bool] = False,
        sync_step: Optional[bool] = True,
        prefix: Optional[str] = "",
        notebook: Optional[str] = None,
        **kwargs,
    ):
        if offline:
            os.environ["WANDB_MODE"] = "dryrun"
        if api is not None:
            os.environ["WANDB_API_KEY"] = api
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = wandb.util.generate_id() if _id is None else _id
        os.environ["WANDB_NOTEBOOK_NAME"] = notebook if notebook else "atomicai"

        if wandb.run is not None:
            cls.end_run()

        cls.experiment = wandb.init(
            resume=sync_step,
            name=experiment_name,
            dir=save_dir,
            project=project_name,
            notes=notes,
            tags=tags,
            entity=entity,
            **kwargs,
        )

        cls.offset_step = cls.experiment.step
        cls.prefix = prefix
        cls.sync_step = sync_step
        cls.log_checkpoint = log_checkpoint

        return cls(tracking_uri=cls.experiment.url)

    @classmethod
    def end_run(cls):
        if cls.experiment is not None:
            # Global step saving for future resuming
            cls.offset_step = cls.experiment.step
            # Send all checkpoints to WB server
            if cls.log_checkpoint:
                wandb.save(os.path.join(cls.save_dir, "*ckpt"))
            cls.experiment.finish()

    def log_metrics(self, metrics, step, **kwargs):
        assert self.experiment is not None, "Initialize experiment first by calling `WANDBLogger.init_experiment(...)`"
        metrics = {f"{self.prefix}{k}": v for k, v in metrics.items()}
        if self.sync_step and step + self.offset_step < self.experiment.step:
            logger.warning("Trying to log at a previous step. Use `sync_step=False`")
        if self.sync_step:
            self.experiment.log(metrics, step=(step + self.offset_step) if step is not None else None)
        elif step is not None:
            self.experiment.log({**metrics, 'step': step + self.offset_step}, **kwargs)
        else:
            self.experiment.log(metrics)

    def log_params(self, params):
        assert self.experiment is not None, "Initialize experiment first by calling `WANDBLogger.init_experiment(...)`"
        self.experiment.config.update(params, allow_val_change=True)

    def log_artifacts(self, artifacts):
        raise NotImplementedError()


__all__ = ["IWBLogger"]
