# -*- encoding: utf-8 -*-
# @File        :   QTSplusTrainer.py
# @Time        :   2025/09/17 00:57:05
# @Author      :   Siyou
# @Description :

import os
import torch
from transformers import Trainer
from transformers.utils import logging, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing import Optional
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, 
)
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_xpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_mps_available,
    is_torch_hpu_available,
)
from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )
from transformers.training_args import OptimizerNames
from transformers.trainer import _is_peft_model
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, Dict
from torch import nn

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"


class QTSplusTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override signature columns to match our dataset output
        self._signature_columns = ["input_ids", "attention_mask", "labels", "vision_input"]
        # Cache for a single eval example used for quick qualitative checks
        self._eval_example_cache: Optional[Dict[str, Any]] = None
        # Cache for tokenizer/processor resolution
        self._proc_for_decode = None

    def _get_tokenizer_like(self):
        """Return an object with `decode`, `pad_token_id`, `eos_token_id`.

        Priority: self.processing_class -> self.tokenizer -> None. If a
        processor is provided, try its .decode or fall back to .tokenizer.
        """
        if self._proc_for_decode is not None:
            return self._proc_for_decode
        proc = getattr(self, "processing_class", None)
        tok = getattr(self, "tokenizer", None)
        cand = None
        if proc is not None:
            if hasattr(proc, "decode"):
                cand = proc
            elif hasattr(proc, "tokenizer"):
                cand = getattr(proc, "tokenizer")
        if cand is None:
            cand = tok
        self._proc_for_decode = cand
        return cand

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        kwargs = {}
        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            # Support Apex AMP if explicitly requested; raise a clear error if missing
            try:
                from apex import amp  # type: ignore
            except Exception as e:
                raise RuntimeError("Apex AMP requested (use_apex=True) but apex is not installed.") from e
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            # Keep a consistent return type with the non-apex branch
            return loss.detach()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs, qts_loss = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            past_val = outputs[self.args.past_index]
            # Avoid holding autograd graphs/tensors longer than needed
            self._past = past_val.detach() if hasattr(past_val, "detach") else past_val

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes
        # Add QTS+ losses
        loss = (loss + qts_loss["flops_loss"] + qts_loss["kv_loss"] + qts_loss["smooth_loss"])
        return (loss, outputs) if return_outputs else loss

    # ---------------------------
    # Logging and inline eval I/O
    # ---------------------------
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Augment log with console prints and a quick eval sample generation.

        Prints the training loss and runs a tiny generate() on a cached eval
        sample to display: question, predicted output, and ground-truth answer.
        Only runs on local process zero to avoid duplicated output.
        """
        # Call parent logger first to preserve default behavior
        try:
            super().log(logs, *args, **kwargs)
        except TypeError:
            # Fallback for HF versions expecting only a single arg
            super().log(logs)

        if not self.is_local_process_zero():
            return

        step = self.state.global_step
        # loss_val = logs.get("loss", logs.get("train_loss", None))
        # if loss_val is not None:
        #     print(f"[train] step={step} loss={loss_val:.6f}")

        # Try printing one eval example prediction on logging steps
        try:
            if self.eval_dataset is not None and self.model is not None and self._get_tokenizer_like() is not None:
                self._print_eval_example()
        except Exception as e:
            # Be robust: don't interrupt training if qualitative print fails
            logger.warning(f"Eval example print failed at step {step}: {e}")

    def _prepare_eval_example(self) -> Optional[Dict[str, Any]]:
        """Prepare and cache a single eval example (CPU tensors).

        Returns a dict with keys: input_ids, attention_mask, labels,
        question_input_ids, vision_input.
        """
        if self._eval_example_cache is not None:
            return self._eval_example_cache

        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            return None

        # Prefer first valid sample
        sample = None
        for i in range(min(64, len(self.eval_dataset))):
            s = self.eval_dataset[i]
            if s is not None:
                sample = s
                break
        if sample is None:
            return None

        self._eval_example_cache = sample
        return self._eval_example_cache

    @torch.no_grad()
    def _print_eval_example(self) -> None:
        """Run a tiny generation on one cached eval sample and print Q/Pred/GT.

        Keeps overhead minimal and uses max_new_tokens=4 (single choice).
        """
        sample = self._prepare_eval_example()
        if sample is None:
            return

        model = self.accelerator.unwrap_model(self.model)
        model_was_training = model.training
        model.eval()

        # Move necessary tensors to device
        device = self.args.device
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)
        q_ids_raw = sample.get("question_input_ids", None)
        q_ids_raw = q_ids_raw.unsqueeze(0).to(device) if q_ids_raw is not None else None

        # vision_input may be a dict of tensors on CPU
        vision_input = sample.get("vision_input", None)
        if isinstance(vision_input, dict):
            vision_input = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vision_input.items()}

        # Derive prompt (question-only) prefix length
        # Use the first index where labels != -100 within valid attention as the
        # start of the answer. This avoids counting special tokens (e.g. EOS)
        # that are also masked out in labels and would inflate the prompt span.
        valid = (attention_mask[0] == 1)
        ans_pos = torch.nonzero((labels[0] != -100) & valid, as_tuple=False)
        if ans_pos.numel() > 0:
            q_len = int(ans_pos[0].item())
        else:
            # Fallback: no supervised answer tokens found; use all valid tokens
            q_len = int(valid.sum().item())
        prompt_ids = input_ids[:, :q_len]

        # Ground-truth answer tokens: where labels != -100 and attention == 1
        ans_mask = (labels[0] != -100) & valid
        gt_ids = input_ids[0, ans_mask].unsqueeze(0)

        # Generate a short answer (single-choice); pass prompt attention mask
        prompt_attention_mask = attention_mask[:, :q_len]
        tok = self._get_tokenizer_like()
        pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None) or 0
        eos_id = getattr(tok, "eos_token_id", None)
        bos_id = getattr(tok, "bos_token_id", None)
        gen_ids = model.generate(
            vision_input=vision_input,
            input_ids=prompt_ids,
            question_input_ids=q_ids_raw,
            attention_mask=prompt_attention_mask,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            bos_token_id=bos_id,
        )

        # Trim generated to the completion portion (generate() prepends prompt)
        comp_ids = gen_ids[:, prompt_ids.shape[1]:]

        # Decode strings
        tok = self._get_tokenizer_like()
        if tok is None or not hasattr(tok, "decode"):
            print("[eval] Skipping example print: no tokenizer available to decode.")
            if model_was_training:
                model.train()
            return
        q_text = tok.decode(prompt_ids[0], skip_special_tokens=False)
        pred_text = tok.decode(comp_ids[0], skip_special_tokens=False)
        gt_text = tok.decode(gt_ids[0], skip_special_tokens=False)

        print("="*20 +"Example" + "="*20)
        print(f"[*]Q:  \t{q_text}")
        print(f"[*]Pred: \t{pred_text}")
        print(f"[*]GT:   \t{gt_text}")
        if model_was_training:
            model.train()
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

        logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.args.lora_enable:
            state_dict_with_lora = self.model.state_dict()
            torch.save(state_dict_with_lora, os.path.join(self.args.output_dir, 'model_with_lora.bin'))
            
        # Save model/config and processing assets to make checkpoints self-contained
        # 1) Save config and generation config when available
        try:
            cfg = getattr(self.model, "config", None)
            if cfg is not None:
                # Prefer HF API if available
                if hasattr(cfg, "save_pretrained"):
                    cfg.save_pretrained(output_dir)
                elif hasattr(cfg, "to_json_file"):
                    cfg.to_json_file(os.path.join(output_dir, "config.json"))
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")

        try:
            gen_cfg = getattr(self.model, "generation_config", None)
            if gen_cfg is not None:
                if hasattr(gen_cfg, "save_pretrained"):
                    gen_cfg.save_pretrained(output_dir)
                elif hasattr(gen_cfg, "to_json_file"):
                    gen_cfg.to_json_file(os.path.join(output_dir, "generation_config.json"))
        except Exception as e:
            logger.warning(f"Failed to save generation config: {e}")

        # 2) Save processor/tokenizer. Prefer saving the full processor when present.
        proc = getattr(self, "processing_class", None)
        try:
            if proc is not None and hasattr(proc, "save_pretrained"):
                proc.save_pretrained(output_dir)
        except Exception as e:
            logger.warning(f"Failed to save processing_class: {e}")

        tok = getattr(self, "tokenizer", None)
        try:
            # Save tokenizer too if explicitly provided and different from processor
            if tok is not None and hasattr(tok, "save_pretrained") and tok is not proc:
                tok.save_pretrained(output_dir)
        except Exception as e:
            logger.warning(f"Failed to save tokenizer: {e}")

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
