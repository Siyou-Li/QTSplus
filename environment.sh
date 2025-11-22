conda create -n qtsplus python=3.11 -y
conda activate qtsplus
conda install conda-forge::gcc=11 conda-forge::gxx=11 -y
conda install nvidia/label/cuda-12.8.1::cuda-toolkit -y
conda install av -c conda-forge -y
# install pytorch with cuda 12.8 support
pip3 install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1
# install prebuilt flash attention wheel
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/flash_attn-2.8.1+cu128torch2.9-cp311-cp311-linux_x86_64.whl
# install deepspeed with cutlass ops disabled to avoid build issues
DS_BUILD_CUTLASS_OPS=0 DS_BUILD_RAGGED_DEVICE_OPS=0 DS_BUILD_EVOFORMER_ATTN=0 pip install deepspeed
# other dependencies
pip install accelerate pandas wandb matplotlib scikit-learn datasets evaluate ftfy sentencepiece bitsandbytes 
