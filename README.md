# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration 
[[Paper](https://arxiv.org/abs/2306.00978)][[Slides](https://www.dropbox.com/scl/fi/dtnp6h6y1mnp7g036axu6/AWQ-slide.pdf?rlkey=ffgh50hxhx8dmsnjiu8kef0ou&dl=0)][[Video](https://youtu.be/3dYLj9vjfA0)]

## Install

1. Clone this repository and navigate to AWQ folder
```
git clone https://github.com/mit-han-lab/llm-awq
cd llm-awq
```

2. Install Package
```
conda create -n awq python=3.9 -y
conda activate awq
pip install --upgrade pip  # enable PEP 660 support
conda install conda-forge::gcc_linux-64=11
conda install conda-forge::gxx_linux-64=11
conda install cuda -c nvidia/label/cuda-12.1
pip install -e .
# /home/tk/anaconda3/envs/awq/lib/python3.9/site-packages/awq_inference_engine-0.0.0-py3.9-linux-x86_64.egg/{awq_inference_engine.cpython-39-x86_64-linux-gnu.so, awq_inference_engine.py}
```

* For **edge devices** like Orin, before running the commands above, please:

    1. Modify [pyproject.toml](pyproject.toml) by commenting out [this line](https://github.com/mit-han-lab/llm-awq/blob/3fce69061682fdd528824e5da3d03a8a8b545f2a/pyproject.toml#L17).
    2. Set [this line](https://github.com/mit-han-lab/llm-awq/blob/3fce69061682fdd528824e5da3d03a8a8b545f2a/pyproject.toml#18) to transformers==4.32.0.
    3. Manually install precompiled PyTorch binaries (>=2.0.0) from [NVIDIA](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).
    4. Set the appropriate Python version for conda environment (e.g., `conda create -n awq python=3.8 -y` for JetPack 5).
  
3. Install efficient W4A16 (4-bit weight, 16-bit activation) CUDA kernel and optimized FP16 kernels (e.g. layernorm, positional encodings).
```
cd awq/kernels
python setup.py install
```

## Usage

We provide several sample script to run AWQ (please refer to `./scripts`). We use Llama3-8B as an example.

1. Perform AWQ search and save search results (we already did it for you):
```bash
python -m awq.entry --model_path /PATH/TO/LLAMA3/llama3-8b \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama3-8b-w4-g128.pt
```

2. Evaluate the AWQ quantized model on WikiText-2 (simulated pseudo quantization)
```bash
python -m awq.entry --model_path /PATH/TO/LLAMA3/llama3-8b \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/llama3-8b-w4-g128.pt \
    --q_backend fake
```

3. Generate real quantized weights (INT4)
```bash
mkdir quant_cache
python -m awq.entry --model_path /PATH/TO/LLAMA3/llama3-8b \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/llama3-8b-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/llama3-8b-w4-g128-awq.pt
```

4. Load and evaluate the real quantized model (now you can see smaller gpu memory usage)
```bash
python -m awq.entry --model_path NousResearch/Meta-Llama-3-8B --w_bit 4 --q_group_size 128 --load_awq  ./data/llama3-8b-w4-g128.pt --q_backend real
```

## Code
* [`pseudo_quantize_tensor`](https://github.com/w32zhong/llm-awq/blob/a5d2d1be5ca6cb4dc542efaaf1005582dc31a047/awq/quantize/quantizer.py#L61)
* [`WQLinear`](https://github.com/w32zhong/llm-awq/blob/a5d2d1be5ca6cb4dc542efaaf1005582dc31a047/awq/quantize/qmodule.py#L78)
