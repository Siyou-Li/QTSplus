# -*- encoding: utf-8 -*-
# @File        :   train_stage1.py
# @Time        :   2025/09/20 01:19:33
# @Author      :   Siyou
# @Description :

import sys
print(sys.path)
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import transformers
import json
from dataclasses import dataclass, field

from transformers import AutoTokenizer, AutoImageProcessor

from src.model.language_model import (
    QTSplusLlama3_ForCausalLM,
    QTSplusQwen2_5_VLTextForCausalLM,
    QTSplusInternLM2_ForCausalLM,
    QTSplusQwen2_ForCausalLM,
)
from src.model.vision_encoder import Qwen2_5_VLVisionProcessor, LlavaSiglipVisionProcessor
from src.model.language_model import Qwen2_5_VLTextForCausalLM
from src.train.qts_plus_trainer import QTSplusTrainer

import os
import torch._dynamo
torch._dynamo.config.suppress_errors = False
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME

# Global acceleration toggles for speed/VRAM
try:
    # Enable TF32 for faster matmuls on Ampere+ GPUs (no accuracy loss for training)
    torch.backends.cuda.matmul.allow_tf32 = True
    # Hint PyTorch to use the fastest float32 matmul path
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    # Prefer Flash / memory-efficient attention kernels via SDPA backend when available
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
except Exception:
    pass


kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

from torch.distributed.elastic.multiprocessing.errors import record
@record
@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    pretrain_lm_model: Optional[str] = field(default="pretrained_models/Qwen2.5-VL-3B-Instruct-LM", metadata={"help": "Path to the LLM or MLLM."})
    lm_model_type: Optional[str] = field(default="qwen2_5_vl_causal_lm", metadata={"help": "qwen2_5_vl_causal_lm"})

    freeze_lm: bool = field(default=True)
    pretrain_mllm: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})
    lm_embed_size: int = field(default=2048, metadata={"help": "LLM embedding size."})

    # vision
    vision_tower: Optional[str] = field(default=None, metadata={"help": "Vision tower type"})
    pretrain_vision_model: str = field(default="pretrained_models/Qwen2.5-VL-3B-Instruct-Vision", metadata={"help": "Path to pretrained model for ViViT."})
    vision_processor: str = field(default="pretrained_models/Qwen2.5-VL-3B-Instruct-Vision", metadata={"help": "Path to processor for ViViT."})
    freeze_vision_model: bool = field(default=False)
    vision_embed_size: int = field(default=2048, metadata={"help": "Vision embedding size."})

    # wandb
    wandb_project_name: Optional[str] = field(default="QTS+", metadata={"help": "wandb project name"})
    wandb_run_name: Optional[str] = field(default="test", metadata={"help": "wandb run name"})

    # QTS+ config
    enable_qts_plus: bool = field(default=True, metadata={"help": "Enable QTS+ tokenizer."})
    qts_plus_n_heads: int = field(default=8, metadata={"help": "Number of heads in QTS+."})
    qts_plus_tau_s: float = field(default=0.1, metadata={"help": "Tau_s in QTS+."})
    qts_plus_nmax: int = field(default=2560, metadata={"help": "Number of max outputs tokens in QTS+."})
    qts_plus_rho_min: float = field(default=0.05, metadata={"help": "Minimum compression ratio in QTS+."})
    qts_plus_rho_max: float = field(default=0.5, metadata={"help": "Maximum compression ratio in QTS+."})
    qts_plus_block_dropout: float = field(default=0.0, metadata={"help": "Block dropout rate in QTS+."})
    qts_plus_reencode: bool = field(default=True, metadata={"help": "Enable tiny re-encoding transformer after selection."})
    qts_plus_scoring_layers: int = field(default=1, metadata={"help": "Number of cross-attention scoring layers in QTS+."})
    qts_plus_reencode_layers: int = field(default=1, metadata={"help": "Number of tiny re-encoding transformer layers in QTS+."})
    project_text_if_needed: bool = field(default=False, metadata={"help": "Project text embeddings if needed."})

    # QTS+ training control
    freeze_qts_scoring_layers: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Freeze QTS+ scoring layer weights."
                " If omitted (None), defaults to: trainable when not using LoRA;"
                " frozen when LoRA is enabled."
            )
        },
    )

    # QTS+ layers are always initialized from Qwen2.5 LM; no random-init toggles

    # Loss lambda
    lambda_t: float = field(default=1.0, metadata={"help": "Attention FLOPs proxy loss weight."})
    lambda_m: float = field(default=1.7, metadata={"help": "KV-cache proxy loss weight."})
    lambda_s: float = field(default=0.05, metadata={"help": "Smoothness loss weight."})

@dataclass
class DataArguments:
    train_jsonl_path: str = field(default="", metadata={"help": "Path to caption data."})
    train_base_path: str = field(default="", metadata={"help": "Path to image data."})
    val_jsonl_path: str = field(default="", metadata={"help": "Path to caption data."})
    val_base_path: str = field(default="", metadata={"help": "Path to image data."})
    dataset_type: str = field(
        default="vscq",
        metadata={
            "help": (
                "Type of dataset: vscq (choice), vqa (qa), or llava-video-178k "
                "(LLaVA-Video-178K JSONL with conversations/video fields)."
            )
        },
    )
    video_max_frames: int = field(
        default=40,
        metadata={"help": "Max frames sampled from a frame-folder/video for Qwen/LLaVA datasets."},
    )
    video_min_frames: int = field(
        default=1,
        metadata={"help": "Min frames sampled (train only) from a frame-folder/video for Qwen/LLaVA datasets."},
    )
    video_sampling: str = field(
        default="uniform",
        metadata={"help": "Frame sampling method for frame-folders/videos: uniform|rand."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "./checkpoint"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    # Use integer step intervals for compatibility with HF Trainer
    eval_steps: float = 0.1
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 0.1  # use integer step interval
    # Enable gradient checkpointing by default to save VRAM
    gradient_checkpointing: bool = True
    # Pin host memory for faster H2D transfers
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 2
    report_to: str = "wandb"
    max_grad_norm: float = 1.0
    # Checkpoint/resume
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a Trainer checkpoint to resume from (e.g. `.../checkpoint-2000`). "
                "Use `last`/`latest`/`true` to automatically pick the most recent checkpoint in `output_dir`."
            )
        },
    )

def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds==filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}

def preprocess_logits_for_metrics(logits, labels):
    # Move predictions to CPU and detach to avoid GPU accumulation during evaluation
    return torch.argmax(logits, dim=-1).detach().cpu()

def _require_single_token(tokenizer: transformers.PreTrainedTokenizer, token: str, label: str) -> int:
    """Assert that `token` exists in the tokenizer as a single token and return its id."""
    try:
        enc = tokenizer(token, add_special_tokens=False)
        ids = enc.get("input_ids")
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, list) or len(ids) != 1:
            raise ValueError(f"Tokenize -> {ids}")
        tid = int(tokenizer.convert_tokens_to_ids(token))
        unk = getattr(tokenizer, "unk_token_id", None)
        if unk is not None and tid == int(unk):
            raise ValueError("token id == unk_token_id")
        if int(ids[0]) != tid:
            raise ValueError(f"tokenize id {ids[0]} != convert id {tid}")
        return tid
    except Exception as e:
        raise ValueError(f"{label} must be a single tokenizer token ({token}). Details: {e}") from e


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

    # Processing assets are saved by trainer._save(), but keep a safe fallback here.
    try:
        proc = getattr(trainer, "processing_class", None)
        if proc is not None and hasattr(proc, "save_pretrained"):
            proc.save_pretrained(output_dir)
    except Exception:
        pass
    try:
        tok = getattr(trainer, "tokenizer", None)
        if tok is not None and hasattr(tok, "save_pretrained"):
            tok.save_pretrained(output_dir)
    except Exception:
        pass

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Apply LoRA only on the LM backbone. Exclude all non-LM modules explicitly.
    # Note: top-level `model` var is QTSplus*; its LM backbone lives under attribute `model`.
    ignore_keywords = [
        'vision_tower',    # vision encoder
        'mm_projector',    # multimodal projector / adapters
        'embed_tokens',    # token embedding layer (kept frozen, no LoRA)
        'tok_embeddings',  # InternLM2 token embedding layer
        'lm_head',         # output head
        'output',          # InternLM2 output head
        'qts_plus',        # QTS+ specific modules
        'adatok',          # any aux token adapters
    ]
    for name, module in model.named_modules():
        # Keep only modules that are part of the LM backbone tree (`model.`)
        if not name.startswith('model.'):
            continue
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)


@dataclass
class DataCollator:
    """Minimal collation to support both legacy and Qwen2.5-VL datasets."""

    def __init__(self, seg_enable: bool = False) -> None:
        self.seg_enable = seg_enable

    def __call__(self, batch: list) -> dict:
        # Filter out None items from batch (missing vision files)
        batch = [b for b in batch if b is not None]

        if len(batch) == 0:
            return {}

        collated = {}
        keys = batch[0].keys()
        for k in keys:
            v = batch[0][k]
            if isinstance(v, torch.Tensor):
                collated[k] = torch.stack([b[k] for b in batch], dim=0)
            else:
                # ``vision_token_index`` is identical across the batch in legacy dataset
                collated[k] = batch[0][k] if k == "vision_token_index" else [b[k] for b in batch]
        return collated


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Normalize/validate generic video sampling args (used by Qwen/LLaVA datasets).
    data_args.video_sampling = str(getattr(data_args, "video_sampling", "uniform") or "uniform").lower()
    if data_args.video_sampling in {"random"}:
        data_args.video_sampling = "rand"
    if data_args.video_sampling not in {"uniform", "rand"}:
        raise ValueError(f"--video_sampling must be one of {{uniform, rand}}, got: {data_args.video_sampling}")
    if int(getattr(data_args, "video_min_frames", 0)) > int(getattr(data_args, "video_max_frames", 0)):
        data_args.video_min_frames = int(data_args.video_max_frames)

    local_rank = training_args.local_rank
    if local_rank == 0:
        wandb.init(project=model_args.wandb_project_name, name=model_args.wandb_run_name)

    # -------------------------
    # Resume-from-checkpoint
    # -------------------------
    # HF Trainer can restore: model weights, optimizer/scheduler/scaler state, RNG state,
    # and TrainerState (incl. global_step) when `resume_from_checkpoint` is provided.
    resume_checkpoint = None
    # Respect explicit user request first.
    user_resume = getattr(training_args, "resume_from_checkpoint", None)
    if isinstance(user_resume, str) and user_resume.strip():
        user_resume = user_resume.strip()
        if user_resume.lower() in {"1", "true", "yes", "y", "last", "latest", "auto"}:
            resume_checkpoint = get_last_checkpoint(training_args.output_dir)
            if resume_checkpoint is None:
                raise ValueError(
                    f"`--resume_from_checkpoint {user_resume}` was set but no checkpoint was found in "
                    f"`output_dir={training_args.output_dir}`."
                )
        else:
            resume_checkpoint = user_resume
        if resume_checkpoint is not None and not os.path.isdir(resume_checkpoint):
            raise ValueError(f"--resume_from_checkpoint must be an existing directory, got: {resume_checkpoint}")

    # If not explicitly set, auto-resume from the latest checkpoint in output_dir when present.
    if resume_checkpoint is None and not getattr(training_args, "overwrite_output_dir", False):
        last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
        if last_checkpoint is not None:
            resume_checkpoint = last_checkpoint
            rank0_print(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            # Guard against accidentally overwriting an existing run directory that has no checkpoints.
            # (Note: we perform this check *before* writing any new metadata files into output_dir.)
            if os.path.isdir(training_args.output_dir) and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty, "
                    "but no Trainer checkpoint was found. Use `--overwrite_output_dir` to train from scratch, "
                    "or set `--resume_from_checkpoint` to a specific checkpoint directory."
                )

    if not os.path.exists(training_args.output_dir) and local_rank == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)

    # save args to output_dir as json file
    if local_rank == 0:
        with open(os.path.join(training_args.output_dir, "training_args.json"), "w") as f:
            json.dump(training_args.to_dict(), f, indent=4)
        with open(os.path.join(training_args.output_dir, "model_args.json"), "w") as f:
            json.dump(model_args.__dict__, f, indent=4)
        with open(os.path.join(training_args.output_dir, "data_args.json"), "w") as f:
            json.dump(data_args.__dict__, f, indent=4)
        rank0_print("Saved training arguments to ", training_args.output_dir)
    if local_rank == 0:
        rank0_print("="*20 + " Tokenizer preparation " + "="*20)

    lm_type = (model_args.lm_model_type or "").lower()
    use_llama = "llama_3" in lm_type or "llama3" in lm_type or "llama-3" in lm_type
    vision_type = (model_args.vision_tower or "").lower() if model_args.vision_tower is not None else ""

    # Guard against a common misconfiguration: InternVL vision tower requires InternLM2 LM.
    if vision_type in {"internvl2_5_vision", "internvl_vision"} and not ("internlm2" in lm_type or "internvl" in lm_type):
        raise ValueError(
            "InternVL vision tower requires an InternLM2-based LM wrapper. "
            "Set `--lm_model_type qts_plus_internlm2_causal_lm` and `--pretrain_lm_model pretrained_models/InternVL2_5-8B-LM`."
        )
    if vision_type in {"llava_siglip_vision"} and "qwen2" not in lm_type:
        raise ValueError(
            "LLaVA SigLIP vision head requires a Qwen2-based LM wrapper. "
            "Set `--lm_model_type qts_plus_qwen2_causal_lm` and `--pretrain_lm_model pretrained_models/LLaVA-Video-7B-Qwen2-LM`."
        )

    # Vision processor + tokenizer selection.
    if vision_type in {"internvl2_5_vision", "internvl_vision"}:
        processor = AutoImageProcessor.from_pretrained(model_args.vision_processor, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.pretrain_lm_model,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        num_new_tokens = 0
    elif vision_type in {"llava_siglip_vision"}:
        processor = LlavaSiglipVisionProcessor.from_pretrained(model_args.vision_processor, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.pretrain_lm_model,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        num_new_tokens = 0
    else:
        processor = Qwen2_5_VLVisionProcessor.from_pretrained(model_args.vision_processor)
        if use_llama:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.pretrain_lm_model,
                cache_dir=training_args.cache_dir,
                fix_mistral_regex=True,
            )
            # Llama tokenizers usually have no PAD; use EOS as PAD for batching.
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        else:
            tokenizer = processor.tokenizer
            # Ensure special tokens are well-defined and consistent across tokenizer/model
            tokenizer.pad_token = "<|endoftext|>"
            tokenizer.eos_token = "<|im_end|>"
            tokenizer.bos_token = "<|endoftext|>"
            tokenizer.padding_side = "right"

        # Ensure multimodal placeholder tokens exist as single tokens across tokenizers.
        # (Embeddings are not used for <|video_pad|> because we splice in vision features.)
        num_new_tokens = tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|image_pad|>", "<|video_pad|>"]}
        )

    # Validate placeholder tokens are present as single tokens (critical for QTS+ integration).
    if vision_type in {"internvl2_5_vision", "internvl_vision"}:
        _require_single_token(tokenizer, "<IMG_CONTEXT>", "InternVL image placeholder token")
        _require_single_token(tokenizer, "<img>", "InternVL <img> start token")
        _require_single_token(tokenizer, "</img>", "InternVL </img> end token")
    elif vision_type in {"llava_siglip_vision"}:
        _require_single_token(tokenizer, "<image>", "LLaVA <image> placeholder token")
    else:
        _require_single_token(tokenizer, "<|video_pad|>", "Qwen video placeholder token")
        _require_single_token(tokenizer, "<|image_pad|>", "Qwen image placeholder token")

    # Convert special tokens to token IDs and set related arguments
    model_args.vocab_size = len(tokenizer)
    rank0_print("vocab_size: ", model_args.vocab_size)
    rank0_print("special tokens: ", tokenizer.special_tokens_map)

    rank0_print("="*20 + " Model preparation " + "="*20)
    rank0_print("Enabled QTS+: ", model_args.enable_qts_plus)
    rank0_print("QTS+ Re-encode: ", model_args.qts_plus_reencode)
    rank0_print("QTS+ Scoring layers: ", model_args.qts_plus_scoring_layers)
    rank0_print("QTS+ Re-encode layers: ", model_args.qts_plus_reencode_layers)
    rank0_print("Project text if needed: ", model_args.project_text_if_needed)

    if model_args.vision_tower is not None:
        if 'qwen2_5_vl_causal_lm' in model_args.lm_model_type:
            rank0_print("Base model: ", model_args.pretrain_lm_model)
            model = QTSplusQwen2_5_VLTextForCausalLM.from_pretrained(
                model_args.pretrain_lm_model,
                cache_dir=training_args.cache_dir,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif "qts_plus_qwen2_causal_lm" in lm_type:
            rank0_print("Base model: ", model_args.pretrain_lm_model)
            model = QTSplusQwen2_ForCausalLM.from_pretrained(
                model_args.pretrain_lm_model,
                cache_dir=training_args.cache_dir,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif use_llama:
            rank0_print("Base model: ", model_args.pretrain_lm_model)
            model = QTSplusLlama3_ForCausalLM.from_pretrained(
                model_args.pretrain_lm_model,
                cache_dir=training_args.cache_dir,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif "internlm2" in lm_type or "internvl" in lm_type:
            rank0_print("Base model: ", model_args.pretrain_lm_model)
            model = QTSplusInternLM2_ForCausalLM.from_pretrained(
                model_args.pretrain_lm_model,
                cache_dir=training_args.cache_dir,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.lm_model_type}")
    else:
        model = Qwen2_5_VLTextForCausalLM.from_pretrained(
            model_args.pretrain_lm_model,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    # Cast model weights to bf16 to halve VRAM usage (compute remains bf16 via autocast)
    try:
        model.to(dtype=torch.bfloat16)
    except Exception:
        pass

    # Resize token embeddings if we added special tokens to the tokenizer.
    try:
        if isinstance(num_new_tokens, int) and num_new_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
            # Keep dtype consistent after resize (new rows may be fp32).
            try:
                model.to(dtype=torch.bfloat16)
            except Exception:
                pass
    except Exception:
        pass

    model.config.use_cache = False

    # Align model config and generation config to tokenizer to silence PAD/BOS/EOS mismatch warnings
    try:
        pad_id = getattr(tokenizer, "pad_token_id", None)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is not None:
            model.config.pad_token_id = pad_id
        if bos_id is not None:
            model.config.bos_token_id = bos_id
        if eos_id is not None:
            model.config.eos_token_id = eos_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            if pad_id is not None:
                model.generation_config.pad_token_id = pad_id
            if bos_id is not None:
                model.generation_config.bos_token_id = bos_id
            if eos_id is not None:
                model.generation_config.eos_token_id = eos_id
    except Exception as e:
        rank0_print(f"[warn] Unable to align model/generation config with tokenizer: {e}")

    # Persist placeholder token ids into the model config so QTS+ can locate them.
    try:
        if vision_type in {"internvl2_5_vision", "internvl_vision"}:
            model.config.image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        elif vision_type in {"llava_siglip_vision"}:
            model.config.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        else:
            model.config.video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
            model.config.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    except Exception:
        pass

    if model_args.freeze_lm:
        # Freeze only the LM weights; keep vision / mm_projector / QTS+ trainable.
        for name, p in model.named_parameters():
            if any(k in name for k in ("vision_tower", "mm_projector", "qts_plus")):
                continue
            p.requires_grad = False

    model.enable_input_require_grads()
    # Prefer SDPA attention implementation for speed + memory efficiency
    try:
        if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "flash_attention_2"
    except Exception:
        pass

    if training_args.gradient_checkpointing:
        # Use non-reentrant checkpointing to reduce overhead on PyTorch 2.x
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    # initialize vision and seg modules on LLM
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
        # Align attention backend + dtype + checkpointing for the vision tower as well
        try:
            vt = model.get_vision_tower()
            if hasattr(vt, "config") and hasattr(vt.config, "_attn_implementation"):
                vt.config._attn_implementation = "flash_attention_2"
            # Ensure vision tower uses bf16 and gradient checkpointing when requested
            try:
                vt.to(dtype=torch.bfloat16)
            except Exception:
                pass
            if training_args.gradient_checkpointing and hasattr(vt, "gradient_checkpointing"):
                vt.gradient_checkpointing = True
        except Exception:
            pass
    else:
        rank0_print("No vision tower is initialized.")

    # After initialization, report effective QTS+ scoring layers (post auto-adjustments)
    try:
        qts_eff = model.get_qts_plus_tower()
        if qts_eff is not None and hasattr(qts_eff, 'selector') and hasattr(qts_eff.selector, 'scoring_layers'):
            rank0_print("QTS+ Scoring layers (effective): ", len(qts_eff.selector.scoring_layers))
    except Exception as _e:
        pass

    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    # Do not tune mm_projector when LoRA is enabled; LoRA-only training should affect LM adapters only
    if model_args.tune_mm_mlp_adapter and not training_args.lora_enable:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # model_args.num_new_tokens = 0
    # model.initialize_vision_tokenizer(model_args, tokenizer)

    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        # Freeze all parameters first; PEFT will add trainable LoRA adapters on selected LM modules
        model.requires_grad_(False)

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters only on LM backbone.")
        model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()

    # Apply freeze/train setting for QTS+ scoring layers per config
    qts = model.get_qts_plus_tower() if hasattr(model, 'get_qts_plus_tower') else None
    if qts is not None and hasattr(qts, 'selector') and hasattr(qts.selector, 'scoring_layers'):
        # Resolve default behavior if not explicitly set:
        # - Without LoRA: train scoring layers by default
        # - With LoRA: keep scoring layers frozen by default (LoRA-only on LM)
        if model_args.freeze_qts_scoring_layers is None:
            want_train = not training_args.lora_enable
        else:
            want_train = not model_args.freeze_qts_scoring_layers
        for p in qts.selector.scoring_layers.parameters():
            p.requires_grad = want_train
        rank0_print("QTS+ scoring layers frozen:", not want_train)

    # Calculate number of enabled trainable parameters and total parameters and percentage
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    rank0_print(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param}")

    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    rank0_print("Model max length: ", data_args.max_length)
    rank0_print("Batch size: ", training_args.per_device_train_batch_size)
    max_length=data_args.max_length
    prompt_style = (
        "llava_qwen2"
        if vision_type in {"llava_siglip_vision"}
        else ("llama3" if use_llama else ("internvl2_5" if vision_type in {"internvl2_5_vision", "internvl_vision"} else "qwen2_5_vl"))
    )
    if model_args.vision_tower == "qwen2_5_vl_vision":
        if data_args.dataset_type == "vscq":
            from src.dataset.sharegptvideo_choice_dataset import ShareGPTVideoChoiceDataset
            train_dataset = ShareGPTVideoChoiceDataset(
                base_path=data_args.train_base_path,
                jsonl_path=data_args.train_jsonl_path,
                processor=processor,
                tokenizer=tokenizer,
                prompt_style=prompt_style,
                max_length=max_length,
                local_rank=local_rank,
                train=True,
                question_instruction="Please provide your options (only A or B or C or D) directly. Do not give any other responses.",
                video_max_frames=data_args.video_max_frames,
                video_min_frames=data_args.video_min_frames,
                video_sampling=data_args.video_sampling,
            )
            eval_dataset = ShareGPTVideoChoiceDataset(
                base_path=data_args.val_base_path,
                jsonl_path=data_args.val_jsonl_path,
                processor=processor,
                tokenizer=tokenizer,
                prompt_style=prompt_style,
                max_length=max_length,
                local_rank=local_rank,
                train=False,
                question_instruction="Please give your answer and provide your reasoning process.",
                video_max_frames=data_args.video_max_frames,
                video_min_frames=data_args.video_min_frames,
                video_sampling=data_args.video_sampling,
            )
        elif data_args.dataset_type in {"vqa", "llava-video-178k", "llava_video_178k"}:
            if data_args.dataset_type in {"llava-video-178k", "llava_video_178k"}:
                from src.dataset.llava_video_178k_qa_dataset import LlavaVideo178KQADataset as _QADataset
            else:
                from src.dataset.sharegptvideo_qa_dataset import ShareGPTVideoQADataset as _QADataset

            train_dataset = _QADataset(
                base_path=data_args.train_base_path,
                jsonl_path=data_args.train_jsonl_path,
                processor=processor,
                tokenizer=tokenizer,
                prompt_style=prompt_style,
                max_length=max_length,
                local_rank=local_rank,
                train=True,
                video_max_frames=data_args.video_max_frames,
                video_min_frames=data_args.video_min_frames,
                video_sampling=data_args.video_sampling,
            )
            eval_dataset = _QADataset(
                base_path=data_args.val_base_path,
                jsonl_path=data_args.val_jsonl_path,
                processor=processor,
                tokenizer=tokenizer,
                prompt_style=prompt_style,
                max_length=max_length,
                local_rank=local_rank,
                train=False,
                video_max_frames=data_args.video_max_frames,
                video_min_frames=data_args.video_min_frames,
                video_sampling=data_args.video_sampling,
            )
        else:
            raise ValueError(f"Unknown dataset type {data_args.dataset_type}")
        rank0_print("Dataset type: ", data_args.dataset_type)
    elif model_args.vision_tower in {"internvl2_5_vision", "internvl_vision"}:
        # InternVL datasets use `<IMG_CONTEXT>` placeholders and an image processor.
        vcfg_path = model_args.pretrain_vision_model or model_args.vision_processor
        if data_args.dataset_type == "vqa":
            from src.dataset.sharegptvideo_qa_internvl_dataset import ShareGPTVideoQAInternVLDataset as _Dataset

            train_dataset = _Dataset(
                base_path=data_args.train_base_path,
                jsonl_path=data_args.train_jsonl_path,
                image_processor=processor,
                tokenizer=tokenizer,
                vision_config_path=vcfg_path,
                max_length=max_length,
                local_rank=local_rank,
                train=True,
                video_max_frames=data_args.video_max_frames,
                video_min_frames=data_args.video_min_frames,
                video_sampling=data_args.video_sampling,
            )
            eval_dataset = _Dataset(
                base_path=data_args.val_base_path,
                jsonl_path=data_args.val_jsonl_path,
                image_processor=processor,
                tokenizer=tokenizer,
                vision_config_path=vcfg_path,
                max_length=max_length,
                local_rank=local_rank,
                train=False,
                video_max_frames=data_args.video_max_frames,
                video_min_frames=data_args.video_min_frames,
                video_sampling="middle",
            )
        elif data_args.dataset_type == "vscq":
            from src.dataset.sharegptvideo_choice_internvl_dataset import (
                ShareGPTVideoChoiceInternVLDataset as _Dataset,
            )

            train_dataset = _Dataset(
                base_path=data_args.train_base_path,
                jsonl_path=data_args.train_jsonl_path,
                image_processor=processor,
                tokenizer=tokenizer,
                vision_config_path=vcfg_path,
                max_length=max_length,
                local_rank=local_rank,
                train=True,
                question_instruction="Please provide your options (only A or B or C or D) directly. Do not give any other responses.",
                video_max_frames=data_args.video_max_frames,
                video_min_frames=data_args.video_min_frames,
                video_sampling=data_args.video_sampling,
            )
            eval_dataset = _Dataset(
                base_path=data_args.val_base_path,
                jsonl_path=data_args.val_jsonl_path,
                image_processor=processor,
                tokenizer=tokenizer,
                vision_config_path=vcfg_path,
                max_length=max_length,
                local_rank=local_rank,
                train=False,
                question_instruction="Please give your answer and provide your reasoning process.",
                video_max_frames=data_args.video_max_frames,
                video_min_frames=data_args.video_min_frames,
                video_sampling="middle",
            )
        else:
            raise ValueError(
                "InternVL vision tower currently supports dataset_type in {'vqa', 'vscq'} only, "
                f"got: {data_args.dataset_type}"
            )
        rank0_print("Dataset type: ", data_args.dataset_type)
    elif model_args.vision_tower in {"llava_siglip_vision"}:
        # LLaVA SigLIP vision head reuses the Qwen-style datasets (vision_input contains `pixel_values`).
        if data_args.dataset_type == "vscq":
            from src.dataset.sharegptvideo_choice_dataset import ShareGPTVideoChoiceDataset as _Dataset

            train_dataset = _Dataset(
                base_path=data_args.train_base_path,
                jsonl_path=data_args.train_jsonl_path,
                processor=processor,
                tokenizer=tokenizer,
                prompt_style=prompt_style,
                max_length=max_length,
                local_rank=local_rank,
                train=True,
                question_instruction="Please provide your options (only A or B or C or D) directly. Do not give any other responses.",
                video_max_frames=data_args.video_max_frames,
                video_min_frames=data_args.video_min_frames,
                video_sampling=data_args.video_sampling,
            )
            eval_dataset = _Dataset(
                base_path=data_args.val_base_path,
                jsonl_path=data_args.val_jsonl_path,
                processor=processor,
                tokenizer=tokenizer,
                prompt_style=prompt_style,
                max_length=max_length,
                local_rank=local_rank,
                train=False,
                question_instruction="Please give your answer and provide your reasoning process.",
                video_max_frames=data_args.video_max_frames,
                video_min_frames=data_args.video_min_frames,
                video_sampling=data_args.video_sampling,
            )
        elif data_args.dataset_type in {"vqa", "llava-video-178k", "llava_video_178k"}:
            if data_args.dataset_type in {"llava-video-178k", "llava_video_178k"}:
                from src.dataset.llava_video_178k_qa_dataset import LlavaVideo178KQADataset as _QADataset
            else:
                from src.dataset.sharegptvideo_qa_dataset import ShareGPTVideoQADataset as _QADataset

            train_dataset = _QADataset(
                base_path=data_args.train_base_path,
                jsonl_path=data_args.train_jsonl_path,
                processor=processor,
                tokenizer=tokenizer,
                prompt_style=prompt_style,
                max_length=max_length,
                local_rank=local_rank,
                train=True,
                video_max_frames=data_args.video_max_frames,
                video_min_frames=data_args.video_min_frames,
                video_sampling=data_args.video_sampling,
            )
            eval_dataset = _QADataset(
                base_path=data_args.val_base_path,
                jsonl_path=data_args.val_jsonl_path,
                processor=processor,
                tokenizer=tokenizer,
                prompt_style=prompt_style,
                max_length=max_length,
                local_rank=local_rank,
                train=False,
                video_max_frames=data_args.video_max_frames,
                video_min_frames=data_args.video_min_frames,
                video_sampling=data_args.video_sampling,
            )
        else:
            raise ValueError(f"Unknown dataset type {data_args.dataset_type}")
        rank0_print("Dataset type: ", data_args.dataset_type)
    else:
        raise ValueError(f"Unknown vision tower {model_args.vision_tower}")
    data_collator = DataCollator()

    # -------------------------
    # Token/label sanity checks
    # -------------------------
    # Keep this lightweight: only check one sample and only on rank0.
    if local_rank in (0, -1, None):
        try:
            # InternVL training critically depends on `<IMG_CONTEXT>` being present
            # in the tokenized prompt and being masked in labels.
            if vision_type in {"internvl2_5_vision", "internvl_vision"}:
                sample = None
                for i in range(min(64, len(train_dataset))):
                    s = train_dataset[i]
                    if s is not None and isinstance(s, dict) and "input_ids" in s:
                        sample = s
                        break
                if sample is not None:
                    ids = sample["input_ids"]
                    labs = sample.get("labels", None)
                    ctx_id = int(getattr(model.config, "image_token_id", -1))
                    if ctx_id < 0:
                        raise ValueError("model.config.image_token_id is not set for InternVL.")
                    n_ctx = int((ids == ctx_id).sum().item()) if torch.is_tensor(ids) else 0
                    if n_ctx <= 0:
                        raise ValueError("No `<IMG_CONTEXT>` tokens found in a tokenized dataset sample.")
                    if torch.is_tensor(labs):
                        ok = bool((labs[ids == ctx_id] == -100).all().item())
                        if not ok:
                            raise ValueError("Labels are not masked (-100) at `<IMG_CONTEXT>` positions.")
        except Exception as e:
            raise RuntimeError(f"Tokenizer/dataset sanity check failed: {e}") from e
    
    rank0_print("="*20 + " Training " + "="*20)

    # Prefer fused AdamW if available for better throughput
    try:
        import inspect
        if training_args.optim == "adamw_torch" and "fused" in inspect.signature(torch.optim.AdamW.__init__).parameters:
            training_args.optim = "adamw_torch_fused"
            rank0_print("Using fused AdamW optimizer.")
    except Exception:
        pass
    trainer = QTSplusTrainer(model=model,
                        args=training_args,
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        compute_metrics=compute_metrics,
                        preprocess_logits_for_metrics=preprocess_logits_for_metrics
                      )
    # Attach full processor for saving/decoding convenience
    try:
        # Use vision processor only when it shares the same tokenizer; otherwise prefer the LM tokenizer for decoding.
        trainer.processing_class = processor if getattr(processor, "tokenizer", None) is tokenizer else tokenizer
    except Exception:
        pass
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_state()
    model.config.use_cache = True

    rank0_print("="*20 + " Save model " + "="*20)
    if training_args.lora_enable:
        state_dict_with_lora = model.state_dict()
        torch.save(state_dict_with_lora, os.path.join(training_args.output_dir, 'model_with_lora.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
