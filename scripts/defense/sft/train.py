"""
OLMo-3-7B SFT for RunPod — adapted from OLMo-core (Beaker infrastructure removed).

This script is launched with torchrun; it performs full fine-tuning using
HSDP + activation checkpointing (no LoRA), matching the OLMo-core SFT recipe.

Usage:
  torchrun --nproc_per_node=8 train.py RUN_NAME /path/to/olmo_checkpoint \\
      --output_dir ./outputs/run_name \\
      --dataset_path ./data/tokenized/model_X \\
      [--seq_len 4096] \\
      [--num_nodes 1] \\
      [--global_batch_size 262144] \\
      [--epochs 2] \\
      [--hf_repo sma1-rmarud/olmo3-sft-original] \\
      [--wandb_project olmo3_safety_sft]

Use a new output directory for each run. This recipe uses LoadStrategy.never;
automatic resume is not supported.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, cast

from rich import print

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_rank, get_rank
from olmo_core.io import copy_dir, dir_is_empty, get_parent, join_path
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.callbacks.wandb import WandBCallback
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import (
    TransformerContextParallelConfig,
)
from olmo_core.utils import prepare_cli_environment, seed_all

log = logging.getLogger(__name__)

DEFAULT_SEQ_LEN = 4096
DEFAULT_NUM_NODES = 1
GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", 8))
MAX_RANK_MICROBATCH_SIZE_TOKENS = 16_384  # 4 seqs @ seq=4096 per microbatch (80GB OOM safe)


@dataclass
class BatchSizeConfig:
    global_batch_size_tokens: int
    sequence_length: int
    world_size: int
    rank_microbatch_size_tokens: int = field(init=False)
    rank_microbatch_size_sequences: int = field(init=False)
    grad_accum_steps: int = field(init=False)
    cp_degree: Optional[int] = None

    def __post_init__(self):
        max_tok = MAX_RANK_MICROBATCH_SIZE_TOKENS

        if self.sequence_length > max_tok:
            min_cp = 2
            while self.sequence_length // min_cp > max_tok:
                min_cp *= 2
            self.cp_degree = min_cp

        cp_factor = self.cp_degree or 1
        dp_world = self.world_size // cp_factor
        rank_batch = self.global_batch_size_tokens // dp_world

        if rank_batch > max_tok * cp_factor:
            self.grad_accum_steps = 1
            while rank_batch // self.grad_accum_steps > max_tok * cp_factor:
                self.grad_accum_steps *= 2
            self.rank_microbatch_size_tokens = rank_batch // self.grad_accum_steps
        else:
            self.rank_microbatch_size_tokens = rank_batch
            self.grad_accum_steps = 1

        assert self.rank_microbatch_size_tokens % self.sequence_length == 0, (
            f"rank_microbatch_size_tokens ({self.rank_microbatch_size_tokens}) "
            f"must be divisible by sequence_length ({self.sequence_length})"
        )
        self.rank_microbatch_size_sequences = (
            self.rank_microbatch_size_tokens // self.sequence_length
        )


@dataclass
class SFTConfig(Config):
    run_name: str
    model: TransformerConfig
    dataset: Optional[NumpyPackedFSLDatasetConfig]
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int

    @classmethod
    def build(
        cls,
        *,
        run_name: str,
        checkpoint: str,
        output_dir: str,
        work_dir: str,
        dataset_path: str,
        seq_len: int,
        num_nodes: int,
        global_batch_size: int,
        epochs: int,
        overrides: List[str],
        init_seed: int = 33333,
        wandb_project: Optional[str] = None,
    ) -> "SFTConfig":
        # Tokenizer — dolma2 vocab (same as allenai/Olmo-3-7B-Instruct-SFT)
        tokenizer_config = TokenizerConfig(
            vocab_size=100278,
            eos_token_id=100257,
            pad_token_id=100277,
            identifier="allenai/Olmo-3-7B-Instruct-SFT",
        )

        clean_path = dataset_path.rstrip("/")
        dataset_config = NumpyPackedFSLDatasetConfig(
            tokenizer=tokenizer_config,
            work_dir=work_dir,
            paths=[f"{clean_path}/token_ids_part_*.npy"],
            expand_glob=True,
            label_mask_paths=[f"{clean_path}/labels_mask_*.npy"],
            generate_doc_lengths=True,
            long_doc_strategy=LongDocStrategy.truncate,
            sequence_length=seq_len,
        )

        bs = BatchSizeConfig(
            sequence_length=seq_len,
            world_size=num_nodes * GPUS_PER_NODE,
            global_batch_size_tokens=global_batch_size,
        )

        if get_local_rank() == 0:
            print(f"Batch size config: {bs}")

        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )

        cp_config: Optional[TransformerContextParallelConfig] = None
        if bs.cp_degree:
            cp_config = (
                TransformerContextParallelConfig.llama3(degree=bs.cp_degree)
                if dataset_config.generate_doc_lengths
                else TransformerContextParallelConfig.zig_zag(degree=bs.cp_degree)
            )

        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            shard_degree=GPUS_PER_NODE // (bs.cp_degree or 1),
        )

        model = TransformerConfig.olmo3_7B(
            vocab_size=tokenizer_config.padded_vocab_size(),
        ).with_rope_scaling(
            YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
        )

        config = SFTConfig(
            run_name=run_name,
            model=model,
            dataset=None,
            data_loader=NumpyDataLoaderConfig(
                global_batch_size=bs.global_batch_size_tokens, seed=34521, num_workers=4
            ),
            train_module=TransformerTrainModuleConfig(
                rank_microbatch_size=bs.rank_microbatch_size_tokens,
                max_sequence_length=bs.sequence_length,
                z_loss_multiplier=None,
                compile_model=True,
                optim=SkipStepAdamWConfig(
                    lr=1e-05,
                    weight_decay=0.1,
                    betas=(0.9, 0.95),
                    compile=False,
                ),
                dp_config=dp_config,
                cp_config=cp_config,
                ac_config=ac_config,
                scheduler=CosWithWarmup(
                    warmup_fraction=0.1,
                    alpha_f=0.0,
                ),
                max_grad_norm=1.0,
            ),
            trainer=TrainerConfig(
                save_folder=output_dir,
                load_strategy=LoadStrategy.never,
                checkpointer=CheckpointerConfig(
                    save_thread_count=1, load_thread_count=32, throttle_uploads=True
                ),
                save_overwrite=True,
                metrics_collect_interval=10,
                cancel_check_interval=10,
                max_duration=Duration.epochs(epochs),
            )
            .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback("garbage_collector", GarbageCollectorCallback())
            .with_callback(
                "checkpointer",
                CheckpointerCallback(
                    save_interval=None, ephemeral_save_interval=None, save_async=True
                ),
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=run_name,
                    project=wandb_project or f"{run_name}-sft",
                    enabled=wandb_project is not None,
                    cancel_check_interval=10,
                ),
            ),
            init_seed=init_seed,
        ).merge(overrides)

        config.dataset = dataset_config
        print(config)
        return config


def train(checkpoint: str, config: SFTConfig, no_save_tokenizer: bool):
    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)

    if config.dataset is None:
        raise RuntimeError("config.dataset is None")

    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset, dp_process_group=train_module.dp_process_group
    )
    trainer = config.trainer.build(train_module, data_loader)

    if not no_save_tokenizer and get_rank() == 0:
        tokenizer_path = join_path(get_parent(dataset.paths[0]), "tokenizer")
        if not dir_is_empty(tokenizer_path):
            dest = join_path(trainer.save_folder, "tokenizer")
            if dir_is_empty(dest):
                log.info(f"Copying tokenizer → {dest}")
                copy_dir(tokenizer_path, dest)

    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    log.info("Loading checkpoint...")
    if not trainer.maybe_load_checkpoint(trainer.save_folder):
        # convert_hf_to_olmo.py 는 {checkpoint}/model_and_optim/ 에만 DCP metadata
        # 를 둔다. trainer.load_checkpoint() 는 dir/.metadata 또는 step*/ 구조를
        # 기대하므로 한 레벨 아래를 직접 가리켜야 한다.
        base_weights_dir = os.path.join(checkpoint, "model_and_optim")
        log.info(f"Starting from pre-training checkpoint: {base_weights_dir}")
        trainer.load_checkpoint(base_weights_dir, load_trainer_state=False)
    else:
        log.info(f"Resumed from save folder: {trainer.save_folder}")

    trainer.fit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_name", help="Run name (used for W&B and checkpoint dir naming)")
    parser.add_argument(
        "checkpoint",
        help="OLMo-core format checkpoint to start from (run convert_hf_to_olmo.py first)",
    )
    parser.add_argument("--output_dir", required=True, help="Where to save checkpoints")
    parser.add_argument(
        "--work_dir",
        default=None,
        help="Dataset cache dir (default: <output_dir>/.dataset_cache)",
    )
    parser.add_argument("--dataset_path", required=True, help="Path to tokenized NumpyPackedFSL data")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--num_nodes", type=int, default=DEFAULT_NUM_NODES)
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Total epochs to train in a new output directory; automatic resume is not supported.",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=None,
        help="Global batch size in tokens (default: 256 * seq_len = 1,048,576 tokens)",
    )
    parser.add_argument(
        "--hf_repo",
        default=None,
        help="HuggingFace repo to upload the converted HF checkpoint to (rank 0 only)",
    )
    parser.add_argument(
        "--wandb_project",
        default=None,
        help="W&B project name (omit to disable W&B logging)",
    )
    parser.add_argument(
        "--no_save_tokenizer",
        action="store_true",
        help="Skip copying the tokenizer into the checkpoint dir",
    )

    args, overrides = parser.parse_known_args()

    output_dir = str(Path(args.output_dir).resolve())
    if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
        raise FileExistsError(f"Use a new training output directory: {output_dir}")
    prepare_training_environment()
    rank = get_rank()
    work_dir = args.work_dir or str(Path(output_dir) / ".dataset_cache")
    global_batch_size = args.global_batch_size or (256 * args.seq_len)

    config = SFTConfig.build(
        run_name=args.run_name,
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        work_dir=work_dir,
        dataset_path=args.dataset_path,
        seq_len=args.seq_len,
        num_nodes=args.num_nodes,
        global_batch_size=global_batch_size,
        epochs=args.epochs,
        overrides=overrides,
        wandb_project=args.wandb_project,
    )

    try:
        train(args.checkpoint, config, args.no_save_tokenizer)
    finally:
        teardown_training_environment()

    # ── Post-training: convert to HF and upload (rank 0 only) ─────────────────
    if rank == 0:
        import glob as _glob
        import subprocess

        # Find the final checkpoint step directory (saved by CheckpointerCallback)
        # Numeric sort — lexical sort would rank step1000 before step500.
        step_dirs = sorted(
            _glob.glob(f"{output_dir}/step*"),
            key=lambda p: int(os.path.basename(p).replace("step", "")),
        )
        if not step_dirs:
            raise RuntimeError("No final step checkpoint found; HF conversion cannot proceed")
        else:
            final_ckpt = step_dirs[-1]
            hf_out = str(Path(output_dir) / "hf_model")
            log.info(f"Converting {final_ckpt} → {hf_out}")

            core = Path(os.environ.get(
                "OLMO_CORE_DIR", Path(__file__).resolve().parents[3] / "external/OLMo-core"
            ))
            from export_checkpoint import export_command
            subprocess.run(
                export_command(core, args.checkpoint, args.dataset_path,
                               final_ckpt, hf_out, args.seq_len),
                check=True,
            )

            print(f"Saved local HuggingFace checkpoint: {hf_out}")
            if args.hf_repo:
                from huggingface_hub import HfApi
                api = HfApi()
                api.create_repo(repo_id=args.hf_repo, repo_type="model", private=False, exist_ok=True)
                api.upload_folder(
                    folder_path=hf_out, repo_id=args.hf_repo, repo_type="model",
                    commit_message=f"Upload {args.run_name}",
                )
                print(f"Uploaded to https://huggingface.co/{args.hf_repo}")
