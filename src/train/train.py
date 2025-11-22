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

from src.model.language_model import QTSplusQwen2_5_VLTextForCausalLM
from src.model.vision_encoder import Qwen2_5_VLVisionProcessor
from src.model.language_model import Qwen2_5_VLTextForCausalLM
from src.train.qts_plus_trainer import QTSplusTrainer

import os
import torch._dynamo
torch._dynamo.config.suppress_errors = False
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

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
    dataset_type: str = field(default="vscq", metadata={"help": "Type of dataset: choice or qa"})

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
        'lm_head',         # output head
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

    local_rank = training_args.local_rank
    if local_rank == 0:
        wandb.init(project=model_args.wandb_project_name, name=model_args.wandb_run_name)

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

    # Load tokenizer and processor
    processor = Qwen2_5_VLVisionProcessor.from_pretrained(model_args.vision_processor)
    tokenizer = processor.tokenizer

    # Ensure special tokens are well-defined and consistent across tokenizer/model
    # Many decoder-only LMs use EOS as PAD; Qwen configs typically set BOS=PAD=151643 and EOS=151645.
    # Define a PAD token if missing to avoid HF warnings during training/generation.
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.bos_token = "<|endoftext|>"
    tokenizer.padding_side = "right"

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

    model.config.use_cache = False

    # Align model config and generation config to tokenizer to silence PAD/BOS/EOS mismatch warnings
    try:
        pad_id = getattr(tokenizer, "pad_token_id", 151643)
        bos_id = getattr(tokenizer, "bos_token_id", 151643)
        eos_id = getattr(tokenizer, "eos_token_id", 151645)
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

    if model_args.freeze_lm:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()
    # Prefer SDPA attention implementation for speed + memory efficiency
    try:
        if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "sdpa"
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
                vt.config._attn_implementation = "sdpa"
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
    if model_args.vision_tower == "qwen2_5_vl_vision":
        if data_args.dataset_type == "vscq":
            from src.dataset.sharegptvideo_choice_dataset import ShareGPTVideoChoiceDataset
            train_dataset = ShareGPTVideoChoiceDataset(
                base_path=data_args.train_base_path,
                jsonl_path=data_args.train_jsonl_path,
                processor=processor,
                max_length=max_length,
                local_rank=local_rank,
                train=True,
            )
            eval_dataset = ShareGPTVideoChoiceDataset(
                base_path=data_args.val_base_path,
                jsonl_path=data_args.val_jsonl_path,
                processor=processor,
                max_length=max_length,
                local_rank=local_rank,
                train=False,
                question_instruction="Please provide your options and provide additional details.",
            )
        elif data_args.dataset_type == "vqa":
            from src.dataset.sharegptvideo_qa_dataset import ShareGPTVideoQADataset
            train_dataset = ShareGPTVideoQADataset(
                base_path=data_args.train_base_path,
                jsonl_path=data_args.train_jsonl_path,
                processor=processor,
                max_length=max_length,
                local_rank=local_rank,
                train=True,
            )
            eval_dataset = ShareGPTVideoQADataset(
                base_path=data_args.val_base_path,
                jsonl_path=data_args.val_jsonl_path,
                processor=processor,
                max_length=max_length,
                local_rank=local_rank,
                train=False,
            )
        else:
            raise ValueError(f"Unknown dataset type {data_args.dataset_type}")
        rank0_print("Dataset type: ", data_args.dataset_type)
    else:
        raise ValueError(f"Unknown vision tower {model_args.vision_tower}")
    data_collator = DataCollator()
    
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
        trainer.processing_class = processor
    except Exception:
        pass
    
    trainer.train()
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
