<p>
  <h1>
  <img src="./assets/logo_with_glasses.svg" height=150px align="right"/>
  Seeing the Forest and the Trees: Query-Aware Tokenizer for Long-Video Multimodal Language Models
  </h1>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2511.11910-b31b1b.svg)](https://arxiv.org/abs/2511.11910)
[![PWC](https://img.shields.io/badge/%F0%9F%8C%8E%20Website-Official%20Page-blue)](https://qtsplus.github.io/)
[![PWC](https://img.shields.io/badge/HuggingFace-Model-Green)](https://huggingface.co/collections/AlpachinoNLP/qtsplus)


## 🚀 Introduction
![](./assets/qtsplus.svg)

Despite the recent advances in the video understanding ability of multimodal large language models (MLLMs), long video understanding remains a challenge. One of the main issues is that the number of vision tokens grows linearly with video length, which causes an explosion in attention cost, memory, and latency. To solve this challenge, we present Query-aware Token Selector (QTSplus), a lightweight yet powerful visual token selection module that serves as an information gate between the vision encoder and LLMs.

Given a text query and video tokens, QTSplus dynamically selects the most important visual evidence for the input text query by (i) scoring visual tokens via cross-attention, (ii) predicting an instance-specific retention budget based on the complexity of the query, and (iii) selecting Top-n tokens. Furthermore, a small re-encoder preserves temporal order using absolute time information. Integrated into Qwen2.5-VL, QTSplus compresses the vision stream by up to 89% and reduces end-to-end latency by 28% on long videos.
```bash
QTSplus/
├── README.md, LICENSE, environment.txt
├── assets/                     # logo and figures for the project
├── src/
│   ├── dataset/                # dataset classes & synthesis scripts
│   ├── demo/                   # interactive/demo scripts
│   ├── model/                  # vision towers, tokenizer, projectors, LLM wrappers
│   ├── train/                  # training and fine‑tuning scripts
│   └── utils/                  # misc helpers (vision preproc, dist utils, etc.)
│── preprocess/             # data‑preprocessing utilities
└── verify/                       # small smoke tests for models & pipelines
```
## 🚀 Quick Start

### 1. Download Pretrained Models
| Model    | Download Link|
|----------|----------|
| QTSplus-3B | [HuggingFace](https://huggingface.co/AlpachinoNLP/QTSplus-3B)|
| QTSplus-3B-FT  | [HuggingFace](https://huggingface.co/AlpachinoNLP/QTSplus-3B-FT)|

### 2. Inference Demo

```python
import os
import glob
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import Optional

# Function to build messages for video or image input
def build_messages(video: str | None, images_dir: str | None, prompt: str) -> list[dict]:
    if video:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video, "max_pixels": 360 * 420, "fps": 1.0},
                    {"type": "text", "text": prompt or "Describe this video."},
                ],
            }
        ]
    if images_dir:
        image_list = sorted(glob.glob(os.path.join(images_dir, "*.jpeg")))
        if not image_list:
            image_list = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        return[
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": image_list},
                        {"type": "text", "text": prompt or "What is in these images?"},
                    ],
                }
            ]
    else:
        raise ValueError("Either video path or images directory must be provided.")

# Input Example
question = "What is happening in the video?"
video_path = "path/to/video.mp4"  # Set to None if using images
images_dir = None  

# Load model and processor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    "AlpachinoNLP/QTSplus-3B",
    trust_remote_code=True,
    local_files_only=True,
).to(dtype=dtype, device=device)

model.eval()

processor = AutoProcessor.from_pretrained(
    "AlpachinoNLP/QTSplus-3B", trust_remote_code=True, local_files_only=True
)

# Build messages for the input video or images
messages = build_messages(video_path, images_dir, question)
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

inputs = processor(
    text=[text],
    images=None,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to(dtype=torch.float16, device=device)

# Extract and format the vision input for QTS+ model
pixel_values_videos = inputs.pop('pixel_values_videos', None)
video_grid_thw = inputs.pop('video_grid_thw', None)
inputs.pop('second_per_grid_ts', None)  # Remove unused parameter

# Format vision input as expected by QTS+ model
vision_input = None
if pixel_values_videos is not None and video_grid_thw is not None:
    vision_input = {
        'pixel_values_videos': pixel_values_videos,
        'video_grid_thw': video_grid_thw
    }
print("="*40)
# Build question_input_ids from the textual question only (avoid including system/vision tokens)
question_ids = processor.tokenizer(
    question,
    return_tensors="pt",
    add_special_tokens=False,
).input_ids.to(dtype=torch.long, device=device)

# Inference
generated_ids = model.generate(
    vision_input=vision_input,
    input_ids=inputs.input_ids,
    question_input_ids=question_ids,
    max_new_tokens=256,
)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
# Fallback: if trimming logic yields empty text (common when using inputs_embeds),
# decode the full sequences instead.
output_text = [
    txt if (txt is not None and txt.strip() != "") else processor.decode(ids, skip_special_tokens=True)
    for txt, ids in zip(output_text, generated_ids)
]
print(output_text[0])
print("="*40)
```

## 💿 Data
![](./assets/dataset.svg)

QTSplus is evaluated and trained on long‑video question‑answering data base on ShareGPTVideo.

- ShareGPTVideo(300k Videos + Captions), Qwen3-235B (Text Teacher) 
- QTS-VSCQ1: Based on ShareGPTVideoChoice(preprocess code in https://github.com/QTSplus/QTSplus-Dataset). Qwen3-235B Converts captions into Visual Single-Choice Questions (VSCQ). Generates options and correct answer key.
- QTS-VSCQ2: Filtering with Qwen2.5-VL (Vision Teacher). Teacher answers VSCQ1, Keep only if Teacher Answer == Ground Truth.
- QTS-VQA: Qwen2.5-VL (Vision Teacher) generates free-form QA answers for VSCQ1's questions.

### Example directory layout

For a dataset rooted at `datasets/ShareGPTVideoChoice/train_300k_480p`, a typical layout is:

```bash
datasets/ShareGPTVideoChoice/
├── train_300k_480p/
│   ├── v_000001/           # directory of frames for one video
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── ...
└── metadata/
    └── choice_train.jsonl  # JSONL annotations (one object per example)
```

The exact filenames and directory names are flexible as long as they are consistent with the `vision_id` values and the `train_base_path` / `val_base_path` arguments passed to `src/train/train.py`.

### Preprocessing and placeholder directories

The directory `preprocess/QTSplus-Dataset` is included as a placeholder for dataset preprocessing scripts and artifacts. No raw data is distributed in this repository; please follow the dataset format described above or consult the project website for instructions on obtaining and preprocessing the data used in the paper.

## 🚄 Training
![](./assets/training_process.svg)
### A. Setup

The repository is designed around a conda‑based Python 3.11 environment with a CUDA‑enabled GPU. The commands below are taken directly from `environment.sh` and provide a reproducible setup on recent Linux distributions.

1. **Create and activate the conda environment**

```bash
conda create -n qtsplus python=3.11 -y
conda activate qtsplus
```

2. **Install toolchain and CUDA toolkit**

```bash
conda install conda-forge::gcc=11 conda-forge::gxx=11 -y
conda install nvidia/label/cuda-12.8.1::cuda-toolkit -y
conda install av -c conda-forge -y
```

3. **Install PyTorch with CUDA 12.8 support**

```bash
pip3 install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

4. **Install core Python libraries**

```bash
pip install transformers==4.57.1
DS_BUILD_CUTLASS_OPS=0 DS_BUILD_RAGGED_DEVICE_OPS=0 DS_BUILD_EVOFORMER_ATTN=0 pip install deepspeed
pip install accelerate pandas wandb matplotlib scikit-learn datasets evaluate ftfy sentencepiece bitsandbytes
```

5. **Install FlashAttention (prebuilt wheel)**

```bash
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/flash_attn-2.8.1+cu128torch2.9-cp311-cp311-linux_x86_64.whl
```

This wheel is specific to Linux x86_64, CUDA 12.8, PyTorch 2.9.0 and Python 3.11; if you deviate from this configuration, you will need to install a compatible FlashAttention build instead.

6. **Verify installation**

After installation, you should be able to run:

```bash
python -c "import torch, transformers, deepspeed, accelerate; print(torch.cuda.is_available())"
```

which should print `True` on a correctly configured GPU machine.

### B. Prepare pretrained models

Split Command:
```bash
python -m src.utils.separate_qwen2_5_vl.py --model_path <path_to_model>
```
- For example: Download `Qwen2.5-VL-3B-Instruct` and split it, and place the parts into:
  - `<path_to_model>/Qwen2.5-VL-3B-Instruct-LM`
  - `<path_to_model>/Qwen2.5-VL-3B-Instruct-Vision`

Paths above are the defaults used by the example scripts; you can change them as long as you adjust the corresponding CLI arguments.

Training QTSplus is handled by `src/train/train.py` together with `src/train/qts_plus_trainer.py`. The script is designed to be launched via `accelerate` and optionally `deepspeed`.

### C. Reference training script

The file `script/training_example.sh` contains a concrete configuration for training QTSplus with Qwen2.5‑VL‑3B:

- It assumes:
  - `PROJECT_PATH` points to the root of this repository.
  - Pretrained models are under `pretrained_models/Qwen2.5-VL-3B-Instruct-LM` and `pretrained_models/Qwen2.5-VL-3B-Instruct-Vision`.
  - Datasets follow the structure described above.
  - A valid `config/accelerate_config.yaml` exists (not shipped in the repo; you must create it using `accelerate config`).
- It launches multi‑GPU training with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $PROJECT_PATH/config/accelerate_config.yaml \
  --main_process_port 29502 src/train/train.py \
  --version v0 \
  --pretrain_lm_model  $PROJECT_PATH/pretrained_models/Qwen2.5-VL-3B-Instruct-LM \
  --lm_model_type qwen2_5_vl_causal_lm \
  --vision_tower qwen2_5_vl_vision \
  --pretrain_vision_model $PROJECT_PATH/pretrained_models/Qwen2.5-VL-3B-Instruct-Vision/model.safetensors \
  --vision_processor $PROJECT_PATH/pretrained_models/Qwen2.5-VL-3B-Instruct-Vision \
  --bf16 True \
  --train_base_path datasets/ShareGPTVideoChoice/train_300k_480p \
  --train_jsonl_path $PROJECT_PATH/datasets/ShareGPTVideoChoice/3b/qa/prediction_correct_train.jsonl \
  --val_base_path datasets/ShareGPTVideoChoice/train_300k_480p \
  --val_jsonl_path $PROJECT_PATH/datasets/ShareGPTVideoChoice/3b/qa/prediction_correct_train.jsonl \
  --output_dir $PROJECT_PATH/checkpoint/QTSplus-3B \
  --dataset_type vscq \
  --model_max_length 512 \
  --num_train_epochs 8 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1
  # ... (additional optimization and QTS+ hyperparameters)
```

You should treat this script as a template and adapt:

- `train_base_path` / `val_base_path` and JSONL paths to your local dataset.
- `dataset_type` (`vscq` for multiple‑choice, `vqa` for open‑ended QA).
- Hyperparameters such as `model_max_length`, learning rate, and QTSplus‑specific parameters (`qts_plus_tau_s`, `qts_plus_nmax`, `qts_plus_rho_min`, `qts_plus_rho_max`, etc.).

### D. Training logic

At a high level:

- `train.py` builds:
  - `ModelArguments` (paths and QTSplus configuration).
  - `DataArguments` (dataset paths and type).
  - `TrainingArguments` (standard HuggingFace training configuration plus LoRA options).
- It initializes:
  - The Qwen2.5‑VL tokenizer and vision processor.
  - The base language model (`Qwen2_5_VLTextForCausalLM`) and wraps it with `QTSplusQwen2_5_VLTextForCausalLM`.
  - The QTSplus selector, re‑encoder, and vision tower via `src/model/qts_plus_arch.py`.
- Datasets are instantiated according to `dataset_type` (`vscq` or `vqa`) and wrapped in a custom `DataCollator`.
- Training is driven by `QTSplusTrainer`, which:
  - Computes the standard causal‑LM loss.
  - Adds the auxiliary QTSplus losses (proxy FLOPs, KV‑cache, smoothness) with weights `lambda_t`, `lambda_m`, `lambda_s`.
  - Periodically logs qualitative predictions on evaluation samples.

The training script logs configuration and metrics to `wandb` by default (see `WANDB_API_KEY` in `script/training_example.sh`); you can disable external logging by setting `--report_to none` in `TrainingArguments` if desired.

## 🫡 Acknowledgements

This codebase builds on and depends heavily on several open‑source projects and datasets:

- **Qwen2.5‑VL**: The underlying multimodal large language model providing the vision encoder and text backbone. Our `QTSplusQwen2_5_VLTextForCausalLM` implementation follows the official Qwen2.5‑VL design and APIs.
- **HuggingFace ecosystem**: We use `transformers`, `accelerate`, and the HuggingFace model/processor format for training, saving, and loading models.
- **Deepspeed** and **FlashAttention**: For efficient large‑scale training and memory‑efficient attention kernels.
- **Weights & Biases (wandb)**: For experiment tracking in the reference training script.
- **ShareGPTVideoChoice / ShareGPTVideoQA**: The dataset formats and evaluation protocols implemented in `src/dataset` and `evaluation/eval_sharegpt_video_choice.py` are designed to match these benchmarks.

We are grateful to the authors and maintainers of these components. Please cite their work in addition to our paper when appropriate.

## 🧰 System Hardware requirements

The resource requirements depend strongly on the resolution, video length, and batch size. The points below summarize what is implicitly assumed by the provided scripts.

- **Operating system**
  - The `environment.sh` setup and reference training scripts assume a recent Linux distribution.
  - Other platforms (Windows, macOS) require adapting the CUDA and package installation steps.

- **GPU**
  - A CUDA‑enabled NVIDIA GPU is strongly recommended for both training and inference.
  - The example environment uses `cuda-toolkit` 12.8 and PyTorch 2.9.0 with CUDA 12.8 wheels.
  - The reference training script uses `CUDA_VISIBLE_DEVICES=0,1,2,3`, i.e., 4 GPUs; single‑GPU training is possible in principle but may require reducing `model_max_length`, input resolution, and batch size.

- **CPU and memory**
  - Video preprocessing and dataloading rely on `torchvision` and `av`; a multi‑core CPU and sufficient RAM are recommended to avoid bottlenecks.
  - For large‑scale experiments on long videos, plan for substantial disk space for raw videos/frames and checkpoints (hundreds of GB can be required depending on dataset size).

- **Inference**
  - The demo script `evaluation/demo.py` supports both GPU and CPU devices (`--device cuda:0` or `--device cpu`), but CPU‑only inference will be significantly slower, especially for long videos.

For the exact experimental setup and hardware used in the paper, please refer to the methodology and appendix sections of `res/qtsplus.pdf` and the associated arXiv version.

## ✨ Cite our work
If you find this repo useful, please consider citing: 

```bibtex
@misc{li2025seeingforesttreesqueryaware,
  title        = {Seeing the Forest and the Trees: Query-Aware Tokenizer for Long-Video Multimodal Language Models},
  author       = {Siyou Li and Huanan Wu and Juexi Shao and Yinghao Ma and Yujian Gan and Yihao Luo and Yuwei Wang and Dong Nie and Lu Wang and Wengqing Wu and Le Zhang and Massimo Poesio and Juntao Yu},
  year         = {2025},
  eprint       = {2511.11910},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/2511.11910}
}
```
