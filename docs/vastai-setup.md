# Vast.AI Setup for Quantized LLM Inference (medical-llm-hallucination-thesis)

This note documents how to set up a Vast.AI instance to run quantized LLM inference with Ollama.


## 1. Instance Setup
- Go to https://vast.ai
- Choose an instance with:
  - Ollama WebUI templat
  - NVIDIA CUDA pre-installed (for GPU support)


## 2. Configure GitHub SSH Access

Run the following commands from an Ubuntu terminal.

`ssh-keygen -t ed25519 -C "[your_email]"`

`eval "$(ssh-agent -s)"`

`ssh-add ~/.ssh/id_ed25519`

`cat ~/.ssh/id_ed25519.pub` 

The last command (above) will write your public key to terminal. Copy this to clipboard.

### Then go to GitHub:

Navigate to...

 - → Settings → SSH and GPG Keys → New SSH Key
 - → Paste your public key

## 3. Clone Repo & Configure Git

`git clone git@github.com:nabilLearns/medical-llm-hallucination-thesis.git`

`cd medical-llm-hallucination-thesis`

### Check out correct branch (optional)
`git checkout [branch_of_interest]`

`git pull`

example branch: `feature/quantize_and_classify`

### Set Git identity (if you want to commit changes to the repo)
`git config --global user.email "[your_email]"`

`git config --global user.name "[github_username]"`


## 4. Python Virtual Environment Setup

<!---python3 -m venv venv
source venv/bin/activate-->

`pip install --upgrade pip`

`pip install -r requirements.txt`


## 5. Enable Full Error Traces from Hydra

`export HYDRA_FULL_ERROR=1`


## 6. Run Experiments


### Test single run

`python run-llm-exp.py model=llama3-med quantization=gguf quantization.level=q2_k max_samples=1000`

Note: You can set `max_samples` to a lower value (e.g., 1, 10) to test the inference pipeline. If you do not specify a value,
the default value (1000) will be used (full run).

### Test multi-run (2 models, 2 quant levels)
```
python run-llm-exp.py --multirun \
  model=gemma3-1b-it,biomistral-7b-med \
  quantization=gguf \
  quantization.level=q8_0,q4_k_m
```

### Full run (multi-model, multi-quant)
```
python run-llm-exp.py --multirun \
  model=biomistral-7b-med,gemma3-1b-it,gemma3-4b-it,gemma3-12b-it,llama3-med,\
qwen2.5-0.5b-it,qwen2.5-1.5b-it,qwen2.5-3b-it,qwen2.5-7b-it-med,qwen2.5-7b-it,qwen2.5-14b-it,qwen2.5-32b-it \
  quantization=gguf \
  quantization.level=q8_0,q6_k,q5_k_m,q4_k_m,q3_k_m,q2_k
```

OR

```
python do_all_exps.py
```

## 7. Notes

- Timeout + CPU safety added in branch: fix/llama3med-q2k-timeout
- Run on A40/A100 instances for speed
- Ollama CLI doesn't let you set max tokens in Python API
- Screenshot for llama3-med Q2_K timeout issue posted in GitHub issue
