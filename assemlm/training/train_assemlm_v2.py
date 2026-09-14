"""Train the AssemLM 2.0 continuous-pose model from scratch."""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Any, Tuple
from torch.utils.data import DataLoader
import numpy as np
import time
import random
import shutil

# Third-Party Libraries
import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

# Local Modules
from assemlm.training.trainer_utils.trainer_tools import normalize_dotlist_args
from assemlm.model.framework import build_framework
from assemlm.training.trainer_utils.trainer_tools import TrainerUtils
from assemlm.training.trainer_utils.trainer_tools import build_param_lr_groups
from assemlm.model.modules.point_encoder.vn_dgcnn.utils import bgs, bgdR, get_6d_rot_loss
from assemlm.model.modules.point_encoder.vn_dgcnn.models.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
logger = get_logger(__name__)
wandb = None


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Publish a JSON metadata file only after complete serialization."""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".partial")
    temporary_path.unlink(missing_ok=True)
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_save_config(cfg, path: Path) -> None:
    """Serialize OmegaConf YAML to a temporary file before publishing it."""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".partial")
    temporary_path.unlink(missing_ok=True)
    try:
        OmegaConf.save(cfg, temporary_path)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_torch_save(state, path: Path) -> None:
    """Write a weight file atomically so disk-full/interruption cannot corrupt it."""
    path = Path(path)
    temporary_path = path.with_name(path.name + ".partial")
    temporary_path.unlink(missing_ok=True)
    try:
        torch.save(state, temporary_path)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_copy_file(source: Path, destination: Path) -> None:
    """Copy a checkpoint sidecar without exposing a partially copied target."""
    source = Path(source)
    destination = Path(destination)
    temporary_path = destination.with_name(destination.name + ".partial")
    temporary_path.unlink(missing_ok=True)
    try:
        shutil.copy2(source, temporary_path)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # save config
        _atomic_save_config(cfg, output_dir / "config.yaml")
        with (output_dir / "config.yaml").open("r", encoding="utf-8") as f_yaml:
            yaml_cfg = yaml.safe_load(f_yaml)
        _atomic_write_json(output_dir / "config.json", yaml_cfg)

    return output_dir


def _cfg_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def validate_release_config(cfg) -> None:
    """Reject options outside the published AssemLM 2.0 training path."""
    errors = []
    if cfg.framework.name != "AssemLM2":
        errors.append("framework.name must be AssemLM2")
    if not _cfg_bool(cfg.framework.pose_regression.enabled):
        errors.append("framework.pose_regression.enabled must be true")
    if errors:
        raise ValueError("Invalid AssemLM 2.0 release configuration: " + "; ".join(errors))


def _dist_rank(default=0) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return default


def seed_everything(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


from assemlm.dataloader import build_dataloader
from assemlm.dataloader.hdf5_datasets import set_augmentation_seed

def prepare_data(cfg, accelerator, tokenizer=None, processor=None) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    logger.info(
        "Creating HDF5 training mixture `%s`",
        cfg.datasets.assemble_data.train_mix,
    )

    pointllm_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.assemble_data.dataset_py, tokenizer=tokenizer, processor=processor)
    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return pointllm_train_dataloader

def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    logical_param_groups = build_param_lr_groups(model=model, cfg=cfg)
    param_groups = []
    for logical_group in logical_param_groups:
        parameters_by_dtype = {}
        for parameter in logical_group["params"]:
            parameters_by_dtype.setdefault(parameter.dtype, []).append(parameter)

        for parameter_dtype, parameters in parameters_by_dtype.items():
            dtype_group = {
                key: value for key, value in logical_group.items() if key != "params"
            }
            original_name = dtype_group.get("name", "group")
            dtype_name = str(parameter_dtype).removeprefix("torch.")
            dtype_group["name"] = f"{original_name}:{dtype_name}"
            dtype_group["params"] = parameters
            param_groups.append(dtype_group)

    for parameter_group in param_groups:
        group_dtypes = {parameter.dtype for parameter in parameter_group["params"]}
        if len(group_dtypes) != 1:
            raise RuntimeError(
                "DeepSpeed optimizer group contains mixed parameter dtypes: "
                f"name={parameter_group.get('name')}, dtypes={group_dtypes}"
            )

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader, self.vlm_val_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        # Number of fully completed training epochs. Epoch checkpoints are
        # written only after the final optimizer update of an epoch.
        self.vla_epoch_count = 0
        self.vla_batches_in_current_epoch = 0
        self.num_update_steps_per_epoch = None
        self.total_batch_size = self._calculate_total_batch_size()
        self.wandb_enabled = False

        self.eval_dir = os.path.join(self.config.output_dir, "eval_results")
        os.makedirs(self.eval_dir, exist_ok=True)

    def prepare_training(self):
        """Seed and initialize a fresh distributed training run."""
        print("Preparing training...")
        rank = _dist_rank()
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        if _cfg_bool(getattr(self.config.trainer, "reproducible_training", False)):
            deterministic = _cfg_bool(getattr(self.config.trainer, "deterministic_training", False))
            seed_everything(seed, deterministic=deterministic)
        else:
            set_seed(seed)
        print(f"Process {rank}: Set random seed to {seed}")
        self.accelerator.print("Initializing all AssemLM 2.0 weights from scratch.")

        freeze_modules = self.config.trainer.freeze_modules
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        print(f"Process {rank}: Frozen modules: {freeze_modules if freeze_modules else 'None'}")

        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader, self.lr_scheduler = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.lr_scheduler,
        )

        self._init_wandb()
        self._init_checkpointing()
        print("Training preparation complete.")

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.assemble_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize W&B without embedding credentials in the repository."""
        global wandb
        if not self.accelerator.is_main_process:
            return
        mode = os.environ.get("WANDB_MODE", "offline").strip().lower()
        if mode in {"disabled", "disable", "off", "false", "0"}:
            return
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled but the 'wandb' package is unavailable. "
                "Install the release dependencies or set WANDB_MODE=disabled."
            ) from exc
        wandb = wandb_module
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=False)
        wandb.init(
            name=self.config.run_id,
            dir=os.path.join(self.config.output_dir, "wandb"),
            project=getattr(self.config, "wandb_project", "assemlm-v2"),
            entity=getattr(self.config, "wandb_entity", None),
            group=getattr(self.config, "wandb_group", "train"),
            mode=mode,
        )
        self.wandb_enabled = True

    def _init_checkpointing(self):
        """Create the checkpoint directory for a fresh run."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print("Checkpointing initialized for a new run.")

    def _save_checkpoint(self):
        save_start = getattr(self.config.trainer, "save_start_step", 0)
        save_end = getattr(self.config.trainer, "save_end_step", float('inf'))

        # If save_end_step is finite and reachable, we additionally save a *full*
        # recoverable checkpoint at steps_{save_end_step}. Other step snapshots
        # remain weight-only to control storage.
        save_end_int = None
        try:
            if save_end != float("inf"):
                save_end_int = int(save_end)
        except Exception:
            save_end_int = None

        # latest checkpoint should NOT be constrained by save_start/save_end.
        # Only the optional step snapshot (steps_{step}) is constrained.
        do_step_snapshot = save_start <= self.completed_steps <= save_end
        do_full_end_snapshot = (
            do_step_snapshot
            and save_end_int is not None
            and self.completed_steps == save_end_int
        )

        latest_dir = os.path.join(self.checkpoint_dir, "latest")
        latest_prev_dir = os.path.join(self.checkpoint_dir, "latest_prev")
        latest_pending_dir = os.path.join(self.checkpoint_dir, "latest.partial")

        # Build the complete checkpoint under a private name. Existing
        # recoverable checkpoints are not touched until every rank finishes
        # writing the new state and rank 0 publishes its metadata.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            shutil.rmtree(latest_pending_dir, ignore_errors=True)
        self.accelerator.wait_for_everyone()

        self.accelerator.save_state(latest_pending_dir)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            trainer_state = {
                "completed_steps": self.completed_steps,
                "vla_epoch_count": getattr(self, "vla_epoch_count", 0),
                "total_batch_size": self.total_batch_size,
                "last_lr": float(self.lr_scheduler.get_last_lr()[0]) if self.lr_scheduler else None,
                "save_time": time.time(),
                "config_path": os.path.join(self.config.output_dir, "config.yaml"),
            }
            _atomic_write_json(
                Path(latest_pending_dir) / "trainer_state.json",
                trainer_state,
            )
            # Rotate only after the pending checkpoint is complete. If a
            # prior interrupted publish left only latest_prev, preserve it.
            if os.path.exists(latest_dir):
                shutil.rmtree(latest_prev_dir, ignore_errors=True)
                os.replace(latest_dir, latest_prev_dir)
            os.replace(latest_pending_dir, latest_dir)

        self.accelerator.wait_for_everyone()

        # Optional step snapshot in steps_{step} (constrained by save window)
        if do_step_snapshot:
            ckpt_dir = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            if do_full_end_snapshot:
                # Save full recoverable state at the end-step snapshot.
                # NOTE: save_state must be called by all processes.
                pending_ckpt_dir = f"{ckpt_dir}.partial"
                if self.accelerator.is_main_process:
                    shutil.rmtree(pending_ckpt_dir, ignore_errors=True)
                self.accelerator.wait_for_everyone()
                self.accelerator.save_state(pending_ckpt_dir)
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    _atomic_write_json(
                        Path(pending_ckpt_dir) / "trainer_state.json",
                        trainer_state,
                    )
                    if os.path.exists(ckpt_dir):
                        shutil.rmtree(ckpt_dir, ignore_errors=True)
                    os.replace(pending_ckpt_dir, ckpt_dir)
                    self.accelerator.print(f" Full End-Step Checkpoint saved at {ckpt_dir}")
            else:
                # Weight-only snapshot for intermediate steps.
                if self.accelerator.is_main_process:
                    pending_ckpt_dir = f"{ckpt_dir}.partial"
                    shutil.rmtree(pending_ckpt_dir, ignore_errors=True)
                    try:
                        os.makedirs(os.path.join(pending_ckpt_dir, "pytorch_model"), exist_ok=True)

                        # Only copy mp_rank_00_model_states.pt (consolidated weights)
                        src_weight = os.path.join(latest_dir, "pytorch_model", "mp_rank_00_model_states.pt")
                        if not os.path.isfile(src_weight):
                            raise FileNotFoundError(
                                f"Latest checkpoint is missing consolidated weights: {src_weight}"
                            )
                        dst_weight = os.path.join(
                            pending_ckpt_dir,
                            "pytorch_model",
                            "mp_rank_00_model_states.pt",
                        )
                        _atomic_copy_file(Path(src_weight), Path(dst_weight))

                        # Copy trainer state for metadata
                        _atomic_copy_file(
                            Path(latest_dir) / "trainer_state.json",
                            Path(pending_ckpt_dir) / "trainer_state.json",
                        )
                        if os.path.exists(ckpt_dir):
                            shutil.rmtree(ckpt_dir, ignore_errors=True)
                        os.replace(pending_ckpt_dir, ckpt_dir)
                    except Exception:
                        shutil.rmtree(pending_ckpt_dir, ignore_errors=True)
                        raise
                    self.accelerator.print(f" Weights Snapshot saved at {ckpt_dir}")

        if self.accelerator.is_main_process:
            self.accelerator.print(f" Full Latest Checkpoint updated at {latest_dir} (backup at {latest_prev_dir})")

        self.accelerator.wait_for_everyone()

        # self.accelerator.wait_for_everyone()

    def _save_epoch_checkpoint(self, completed_epoch: int):
        """Save one independently resumable full state at a completed epoch.

        Unlike step snapshots, epoch checkpoints are not controlled by
        save_start_step/save_end_step. ``accelerator.save_state`` stores the
        model, optimizer, scheduler, scaler/DeepSpeed state, and RNG state in
        the same recoverable format as ``checkpoints/latest``.
        """
        completed_epoch = int(completed_epoch)
        if completed_epoch <= 0:
            raise ValueError(f"completed_epoch must be positive, got {completed_epoch}.")

        epoch_dir_name = f"epoch_{completed_epoch:04d}_steps_{self.completed_steps}"
        epoch_checkpoint_dir = os.path.join(self.checkpoint_dir, "epochs", epoch_dir_name)
        pending_epoch_dir = f"{epoch_checkpoint_dir}.partial"

        self.accelerator.wait_for_everyone()
        # Build the epoch snapshot privately on every rank.  A process crash
        # must leave the previous complete epoch checkpoint untouched and a
        # later run must never resume from a half-written directory.
        if self.accelerator.is_main_process:
            shutil.rmtree(pending_epoch_dir, ignore_errors=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(pending_epoch_dir)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            trainer_state = {
                "completed_steps": self.completed_steps,
                "vla_epoch_count": completed_epoch,
                "total_batch_size": self.total_batch_size,
                "last_lr": float(self.lr_scheduler.get_last_lr()[0]) if self.lr_scheduler else None,
                "save_time": time.time(),
                "config_path": os.path.join(self.config.output_dir, "config.yaml"),
                "checkpoint_kind": "completed_epoch",
            }
            _atomic_write_json(Path(pending_epoch_dir) / "trainer_state.json", trainer_state)
            if os.path.exists(epoch_checkpoint_dir):
                shutil.rmtree(epoch_checkpoint_dir, ignore_errors=True)
            os.replace(pending_epoch_dir, epoch_checkpoint_dir)
            self.accelerator.print(
                f" Full Epoch Checkpoint saved: epoch={completed_epoch}, "
                f"step={self.completed_steps}, path={epoch_checkpoint_dir}"
            )

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics at the configured interval."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                metrics = dict(metrics)
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                epoch_divisor = self.num_update_steps_per_epoch or len(self.vla_train_dataloader)
                metrics["epoch"] = round(self.completed_steps / epoch_divisor, 2)

                # Record the same supervised tokens used by the main LM loss.
                # The internal key keeps its old name for compatibility, while
                # W&B uses explicit supervised-token metric names.
                per_pose_loss = metrics.pop("per_pose_loss", None)
                if per_pose_loss is not None:
                    if isinstance(per_pose_loss, torch.Tensor):
                        per_pose_values = per_pose_loss.detach().float().cpu().flatten().tolist()
                    elif isinstance(per_pose_loss, (list, tuple, np.ndarray)):
                        per_pose_values = np.asarray(per_pose_loss, dtype=np.float32).reshape(-1).tolist()
                    else:
                        per_pose_values = [float(per_pose_loss)]

                    metrics["supervised_token_count"] = len(per_pose_values)
                    if per_pose_values:
                        metrics["supervised_token_loss_mean"] = float(np.mean(per_pose_values))
                        for idx, value in enumerate(per_pose_values):
                            metrics[f"supervised_token_loss/pos_{idx}"] = float(value)

                if self.wandb_enabled:
                    wandb.log(metrics, step=self.completed_steps)
                logger.info(f"Step {self.completed_steps}, Metrics: {metrics}")

    def _create_data_iterators(self):
        """create data iterators"""
        # Epoch-indexed seeding keeps each distributed shuffle deterministic.
        dataloader_set_epoch = getattr(self.vla_train_dataloader, "set_epoch", None)
        sampler = getattr(self.vla_train_dataloader, "sampler", None)
        if callable(dataloader_set_epoch):
            dataloader_set_epoch(int(self.vla_epoch_count))
        elif callable(getattr(sampler, "set_epoch", None)):
            sampler.set_epoch(int(self.vla_epoch_count))
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vla_batches_in_current_epoch = 0

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            # Epoch counting is performed immediately after its final optimizer
            # update in train(), before epoch evaluation/checkpointing. Here we
            # only create the iterator for the already-counted next epoch.
            self._create_data_iterators()
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        gradient_accumulation_steps = max(
            1,
            int(getattr(self.accelerator, "gradient_accumulation_steps", 1)),
        )
        batches_per_epoch = int(len(self.vla_train_dataloader))
        if batches_per_epoch <= 0:
            raise ValueError("Training dataloader must contain at least one batch per epoch.")
        self.num_update_steps_per_epoch = (
            batches_per_epoch + gradient_accumulation_steps - 1
        ) // gradient_accumulation_steps
        epoch_eval_enabled = _cfg_bool(
            getattr(self.config.trainer, "epoch_eval_enabled", False)
        )
        epoch_checkpoint_enabled = _cfg_bool(
            getattr(self.config.trainer, "epoch_checkpoint_enabled", False)
        )
        self.accelerator.print(
            "Epoch boundary actions: "
            f"batches_per_epoch={batches_per_epoch}, "
            f"updates_per_epoch={self.num_update_steps_per_epoch}, "
            f"epoch_eval_enabled={epoch_eval_enabled}, "
            f"epoch_checkpoint_enabled={epoch_checkpoint_enabled}"
        )

        # create progress bar
        #  checkpoint  completed_steps
        #  0/maximum
        progress_bar = tqdm(
            total=int(self.config.trainer.max_train_steps),
            initial=int(self.completed_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()
            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            completed_epoch = None
            self.vla_batches_in_current_epoch += 1
            if self.vla_batches_in_current_epoch > batches_per_epoch:
                raise RuntimeError(
                    "Training dataloader yielded more batches than len(dataloader): "
                    f"seen={self.vla_batches_in_current_epoch}, expected={batches_per_epoch}."
                )
            dataloader_epoch_complete = (
                self.vla_batches_in_current_epoch == batches_per_epoch
            )
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            if dataloader_epoch_complete:
                if not self.accelerator.sync_gradients:
                    raise RuntimeError(
                        "The final batch of an epoch did not synchronize gradients. "
                        "The distributed dataloader/gradient-accumulation configuration "
                        "cannot produce a recoverable epoch boundary."
                    )
                completed_epoch = int(self.vla_epoch_count) + 1
                self.vla_epoch_count = completed_epoch
                step_metrics["completed_epoch"] = completed_epoch

            # evaluate model
            t_start_eval = time.perf_counter()
            # step_metrics = {} # test
            eval_start = getattr(self.config.trainer, "eval_start_step", 0)
            eval_end = getattr(self.config.trainer, "eval_end_step", float('inf'))

            step_eval_due = (
                self.completed_steps % self.config.trainer.eval_interval == 0
                and self.config.datasets.assemble_data.eval_with_val
                and eval_start <= self.completed_steps <= eval_end
            )

            if completed_epoch is not None and epoch_eval_enabled:
                # Epoch evaluation is intentionally independent of eval_interval,
                # eval_start_step, and eval_end_step. If a step evaluation is due
                # at the same update, this single epoch evaluation supersedes it
                # and avoids evaluating the identical model twice.
                epoch_result_dir = (
                    f"epochs/epoch_{int(completed_epoch):04d}_steps_{self.completed_steps}"
                )
                step_metrics = self.eval_pose(
                    step_metrics,
                    result_dir_name=epoch_result_dir,
                )
            elif step_eval_due:
                step_metrics = self.eval_pose(step_metrics)
            t_end_eval = time.perf_counter()
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                        "eval_times": f"{t_end_eval - t_start_eval:.3f}",
                    }
                )
            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            # save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()
            if completed_epoch is not None and epoch_checkpoint_enabled:
                # This full checkpoint is independent of the step save window.
                # It is saved after epoch evaluation succeeds, so every emitted
                # epoch checkpoint has a matching completed evaluation.
                self._save_epoch_checkpoint(completed_epoch)
            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_pose(self, step_metrics: dict = None, result_dir_name: str = None) -> dict:
        """
        Evaluate the model on both validation and a subset of training data.
        """
        if step_metrics is None:
            step_metrics = {}

        # IMPORTANT (memory/stability under deepspeed/accelerate):
        # Run eval forward/generate on ALL ranks (each rank evaluates a shard).
        # Only rank0 logs/saves results. Running eval only on rank0 can cause
        # parameter gathering / persistent buffers on a single GPU and lead to
        # apparent memory growth after each eval.

        # 1) Evaluate on Validation Set (if exists)
        if self.vlm_val_dataloader is not None:
            self._eval_on_dataset(
                self.vlm_val_dataloader.dataset,
                self.vlm_val_dataloader.collate_fn,
                prefix="eval",
                step_metrics=step_metrics,
                result_dir_name=result_dir_name,
            )
        else:
            if self.accelerator.is_main_process:
                self.accelerator.print("Warning: vlm_val_dataloader is None, skipping validation set evaluation.")

        # 2) Evaluate on Training Subset
        if self.vla_train_dataloader is not None:
            train_ds = getattr(self.vla_train_dataloader, "dataset", None)
            train_collate = getattr(self.vla_train_dataloader, "collate_fn", None)
            if train_ds is not None:
                self._eval_on_dataset(
                    train_ds,
                    train_collate,
                    prefix="train_eval",
                    step_metrics=step_metrics,
                    result_dir_name=result_dir_name,
                )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return step_metrics

    @staticmethod
    def _format_pa_threshold(threshold) -> str:
        return f"{float(threshold):g}"

    def _get_pa_thresholds(self) -> list:
        raw_thresholds = str(self.config.trainer.pa_thresholds)
        thresholds = [float(value.strip()) for value in raw_thresholds.split(",")]
        if not thresholds or any(threshold <= 0 for threshold in thresholds):
            raise ValueError("trainer.pa_thresholds must contain positive values.")
        return thresholds

    def _eval_on_dataset(
        self,
        dataset_eval,
        collate_fn,
        prefix,
        step_metrics,
        result_dir_name: str = None,
    ) -> None:
        """
        Core evaluation logic on a specific dataset.
        """
        import gc
        from torch.utils.data import Subset, DataLoader

        model = self.model
        model.eval()

        total_samples = len(dataset_eval)
        num_samples = self.config.datasets.assemble_data.get("num_samples", 50)  # Use a reasonable default

        eval_seed = int(getattr(self.config.trainer, "eval_seed", 42))
        set_augmentation_seed(dataset_eval, eval_seed)
        indices = list(range(total_samples))
        eval_rng = random.Random(eval_seed)
        eval_indices = eval_rng.sample(indices, min(num_samples, total_samples))

        # Shard eval indices across distributed ranks to avoid doing eval only on rank0.
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        eval_indices_rank = eval_indices[rank::world_size]
        eval_dataset = Subset(dataset_eval, eval_indices_rank)
        eval_batch_size = (
            self.config.datasets.assemble_data.get("eval_batch_size", None)
            or self.config.datasets.assemble_data.get("per_device_batch_size", 1)
            or 1
        )
        eval_batch_size = max(1, int(eval_batch_size))

        # Deterministic shuffle for eval (per-rank) to keep runs reproducible.
        eval_gen = torch.Generator()
        eval_gen.manual_seed(eval_seed + int(rank))

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
            generator=eval_gen,
            drop_last=False,
        )

        # Reuse chamfer module to avoid repeated CUDA extension init.
        if not hasattr(self, "_eval_chamfer_dist_fn") or self._eval_chamfer_dist_fn is None:
            self._eval_chamfer_dist_fn = dist_chamfer_3D.chamfer_3DDist()
        chamfer_dist_fn = self._eval_chamfer_dist_fn
        device = self.accelerator.device
        all_results = []
        pa_thresholds = self._get_pa_thresholds()
        pa_threshold_labels = [self._format_pa_threshold(threshold) for threshold in pa_thresholds]

        # NOTE:
        # Do NOT use torch.inference_mode() here.
        # Some model components may cache intermediate tensors during eval/generate.
        # Inference-mode tensors cannot participate in autograd later, and can crash
        # the next training step with:
        #   "Inference tensors cannot be saved for backward"
        # torch.no_grad() still avoids gradient tracking, but produces normal tensors
        # that remain autograd-compatible if they are (unexpectedly) re-used.
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc=f"Evaluating on {prefix}", leave=False):
                # 0. Move batch to device manually
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                        batch[k] = torch.stack(v).to(device)

                # Prepare GT & Inputs
                instructions = batch.get("lang")

                gt_trans_all = batch.get("src_trans")
                gt_rot_6d_all = batch.get("src_rot")
                if gt_trans_all is None or gt_rot_6d_all is None:
                    pre_pose = batch.get("pre_pose")
                    if pre_pose is not None:
                        gt_trans_all = pre_pose[:, :3].reshape(-1, 3, 1)
                        gt_rot_6d_all = pre_pose[:, 3:9]

                src_pc_all = batch.get("src_pc")

                # Forward pass for the continuous pose loss.
                forward_outputs = model.forward(batch)
                batch_pose_loss = forward_outputs["pose_loss"].item()

                # Generate Poses [B, 9]
                generated_poses = model.generate(batch)

                # Metric Calculation
                # Eval metrics mix model outputs from bf16/autocast paths with
                # dataset tensors. Normalize metric tensors to fp32 before
                # rotation/chamfer math so torch.bmm never sees mixed dtypes.
                generated_poses = generated_poses.to(device=device, dtype=torch.float32)
                gt_trans_all = gt_trans_all.to(device=device, dtype=torch.float32)
                gt_rot_6d_all = gt_rot_6d_all.to(device=device, dtype=torch.float32)
                src_pc_all = src_pc_all.to(device=device, dtype=torch.float32)

                valid_mask = (generated_poses[:, 0] != -100.0) & (generated_poses != 0).any(dim=-1)
                valid_mask = valid_mask & torch.isfinite(generated_poses).all(dim=-1)

                pred_t = generated_poses[:, :3].view(-1, 3, 1)
                pred_rot_6d = generated_poses[:, 3:9]

                batch_gd = get_6d_rot_loss(gt_rot_6d_all, pred_rot_6d)
                batch_rmse_t = (pred_t - gt_trans_all).pow(2).mean(dim=1).sqrt().squeeze(-1)

                pred_R = bgs(pred_rot_6d.reshape(-1, 2, 3).permute(0, 2, 1))
                gt_R = bgs(gt_rot_6d_all.reshape(-1, 2, 3).permute(0, 2, 1))

                transformed_src_pc_pred = pred_R.transpose(1, 2) @ src_pc_all + pred_t
                transformed_src_pc_gt = gt_R.transpose(1, 2) @ src_pc_all + gt_trans_all
                d1, d2, _, _ = chamfer_dist_fn(transformed_src_pc_gt.permute(0, 2, 1), transformed_src_pc_pred.permute(0, 2, 1))
                batch_cd = 0.5 * (torch.mean(d1, dim=-1) + torch.mean(d2, dim=-1))
                transformed_src_pc_pred_r = pred_R.transpose(1, 2) @ src_pc_all + gt_trans_all
                d1_r, d2_r, _, _ = chamfer_dist_fn(
                    transformed_src_pc_gt.permute(0, 2, 1),
                    transformed_src_pc_pred_r.permute(0, 2, 1),
                )
                batch_cd_r = 0.5 * (torch.mean(d1_r, dim=-1) + torch.mean(d2_r, dim=-1))
                batch_pa_values = {
                    f"PA({label})": (batch_cd < threshold).float()
                    for label, threshold in zip(pa_threshold_labels, pa_thresholds)
                }

                for b_idx in range(len(instructions)):
                    is_valid = valid_mask[b_idx].item()
                    res = {"valid": is_valid, "pose_loss": batch_pose_loss, "instruction": instructions[b_idx]}
                    if is_valid:
                        res.update({
                            "GD": batch_gd[b_idx].item(),
                            "RMSE(T)": batch_rmse_t[b_idx].item(),
                            "CD": batch_cd[b_idx].item(),
                            "CD(R)": batch_cd_r[b_idx].item(),
                        })
                        for metric_name, pa_values in batch_pa_values.items():
                            res[metric_name] = pa_values[b_idx].item()
                    all_results.append(res)

                # Explicitly drop references to large CUDA tensors each iteration
                del forward_outputs, generated_poses, valid_mask
                del pred_t, pred_rot_6d, pred_R, gt_R
                del transformed_src_pc_pred, transformed_src_pc_pred_r, transformed_src_pc_gt
                del d1, d2, d1_r, d2_r, batch_cd, batch_cd_r, batch_pa_values
                gc.collect()
                torch.cuda.empty_cache()

        # Gather results to rank0 for logging/saving.
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            gathered = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(all_results, gathered, dst=0)
            if rank == 0:
                all_results = [r for part in gathered for r in (part or [])]
            else:
                all_results = []

        # Summarize / log / save only on rank0
        if rank == 0:
            valid_res = [r for r in all_results if r["valid"]]
            mean_metrics = {}
            if all_results:
                mean_metrics = {
                    f"{prefix}/pose_loss_mean": float(np.mean([r["pose_loss"] for r in all_results])),
                    f"{prefix}/valid_rate": len(valid_res) / len(all_results),
                }
                if valid_res:
                    mean_metrics.update({
                        f"{prefix}/GD_mean": float(np.mean([r["GD"] for r in valid_res])),
                        f"{prefix}/RMSE(T)_mean": float(np.mean([r["RMSE(T)"] for r in valid_res])),
                        f"{prefix}/CD_mean": float(np.mean([r["CD"] for r in valid_res])),
                        f"{prefix}/CD(R)_mean": float(np.mean([r["CD(R)"] for r in valid_res])),
                    })
                    for label in pa_threshold_labels:
                        metric_key = f"PA({label})"
                        mean_metrics[f"{prefix}/{metric_key}_mean"] = float(
                            np.mean([r[metric_key] for r in valid_res])
                        )
                step_metrics.update(mean_metrics)
                self.accelerator.print(f"--- Step {self.completed_steps} {prefix} Stats ---")
                for k, v in mean_metrics.items():
                    self.accelerator.print(f"  {k}: {v:.6f}")

                if self.wandb_enabled:
                    wandb.log(mean_metrics, step=self.completed_steps)

                    if prefix == "eval":
                        pa_columns = [f"PA({label})" for label in pa_threshold_labels]
                        columns = ["index", "instruction", "valid", "GD", "RMSE(T)", "CD", "CD(R)"]
                        columns.extend(pa_columns)
                        columns.append("pose_loss")
                        table_data = []
                        for i, r in enumerate(all_results):
                            row = [
                                i,
                                r.get("instruction", ""),
                                r.get("valid", False),
                                r.get("GD", None),
                                r.get("RMSE(T)", None),
                                r.get("CD", None),
                                r.get("CD(R)", None),
                            ]
                            row.extend([r.get(column, None) for column in pa_columns])
                            row.append(r.get("pose_loss", None))
                            table_data.append(row)
                        wandb.log({f"{prefix}/summary_table": wandb.Table(columns=columns, data=table_data)}, step=self.completed_steps)

            if result_dir_name is None:
                result_dir_name = f"steps_{self.completed_steps}"
            eval_step_dir = os.path.join(self.eval_dir, result_dir_name)
            os.makedirs(eval_step_dir, exist_ok=True)
            _atomic_write_json(
                Path(eval_step_dir) / f"{prefix}_results.json",
                all_results,
            )

        gc.collect()
        torch.cuda.empty_cache()
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.assemble_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            self.model.train()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)

                pose_loss = output_dict["pose_loss"]
                per_pose_loss = output_dict.get("per_pose_loss", None)
                total_loss = pose_loss

            self.accelerator.backward(total_loss)
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.trainer.gradient_clipping
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            # AcceleratedOptimizer applies zero_grad only on synchronized
            # accumulation steps. Calling it before backward would clear the
            # gradients accumulated by preceding microbatches at the start of
            # the final microbatch.
            self.optimizer.zero_grad(set_to_none=True)

        step_metrics = {
            "assemble_pose_loss": pose_loss.item(),
            "per_pose_loss": per_pose_loss
        }
        for metric_key in (
            "pose_regression_loss",
            "translation_l1_loss",
            "rotation_geodesic_loss",
            "pred_pose_mean_abs",
        ):
            metric_value = output_dict.get(metric_key, None)
            if metric_value is None:
                continue
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.detach().float().mean().item()
            step_metrics[metric_key] = metric_value
        return step_metrics

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            _atomic_torch_save(state_dict, Path(final_checkpoint) / "pytorch_model.pt")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and self.wandb_enabled:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    # Delay distributed initialization until the real training entry is
    # invoked. Importing this module (including help and static checks) must
    # not create process groups or inspect launcher state.
    accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin())
    accelerator.print(accelerator.state)
    validate_release_config(cfg)
    logger.info("VLA Training :: Warming Up")
    if _cfg_bool(getattr(cfg.trainer, "reproducible_training", False)):
        base_seed = int(getattr(cfg, "seed", 42))
        deterministic = _cfg_bool(getattr(cfg.trainer, "deterministic_training", False))
        seed_everything(base_seed, deterministic=deterministic)
        logger.info(
            f"Reproducible training seed initialized before model/data creation: "
            f"seed={base_seed}, deterministic={deterministic}"
        )

    setup_directories(cfg=cfg)
    pvlm = build_framework(cfg)
    assemble_train_dataloader = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        tokenizer=pvlm.pvlm_interface.tokenizer,
        processor=pvlm.pvlm_interface.processor
    )

    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=pvlm, cfg=cfg)
    trainer = VLATrainer(
        cfg=cfg,
        model=pvlm,
        vla_train_dataloader=assemble_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()
    logger.info("AssemLM 2.0 training complete.")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to the v2 YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
