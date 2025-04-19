# medical-llm-hallucination-thesis

This repository evaluates the impact of quantization on the ability of a Large Language Model (LLM) to detect hallucination(s) in answers to medical questions. The `run-llm-exp.py` script support inference and benchmarking across multiple models and quantization schemes. The currently supported models and quantization schemes are listed below in the 'Experiment Design' section, but more models/quantization schemes can be introduced by adding config files for what you'd like to add, to the `conf/` folder in the repo.

Example cli usage of `run-llm-exp.py`, with gemma3 (instruction tuned) 1B parameter model, using the q4_k_m GGUF quantized version.
```
python run-llm-exp.py model=gemma3-1b-it quantization=gguf quantization.level=q4_k_m max_samples=10
```

## Features
- [x] Hydra-powered CLI interface via `run-llm-exp.py`
- [x] Support binary hallucination classification across 10 models x 6 quantization levels
- [x] Outputs include:
  - JSON logs of system + model metrics
  - CSV of model predictions, ground truths, extracted model judgements
  - Confusion matrix metrics, abstention rate, inference latency, and GPU usage statistics

## Project Status

- ✅ Most models completed (see Results section)
- ⚠️ Some larger models skipped due to sharded GGUF format incompatibility and time constraints
- 🛠️ Actively running final models to complete benchmarks

Each experiment records a number of metrics: 
* Confusion matrix: Accuracy, precision, recall, f1 score, support
* Abstention rate
* Inference latency statistics (e.g., peak, avg, std_dev, variance)
* GPU usage statistics (e.g., peak, avg, std_dev for GPU load, utilization)

## Experiment Design
**models:** biomistral-7b-med, llama3-med, gemma3-1b-it, gemma3-4b-it, gemma3-12b-it, qwen2.5-0.5b-it, qwen2.5-1.5b-it, qwen2.5-3b-it, qwen2.5-7b-it, ~~qwen2.5-14b-it~~ (we intended to run this but are ignoring for now ; due to its large model size, it required GGUF files to be sharded, and because sharded GGUF files are not yet supported in our pipeline, we are choosing to ignore large models for now due to time contstraints), ~~qwen2.5-32b-it~~ (choosing to ignore for now, for the same reason as the previous model), qwen2.5-7b-it-med

**quantization_schemes (GGUF):** q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m, q2_k

## Results
### Available
biomistral7b (6 quants)

gemma3-12b (6 quants)

gemma3-4b (6 quants)

gemma3-1b (6 quants)

llama3-med (q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m)

qwen2.5-7b (6 quants)
### TBD
qwen2.5-0.5b-it

qwen2.5-1.5b-it

* Results for the medically fine-tuned **BioMistral-7B q8_0 GGUF quant** when tested on the pqa-labelled 1k dataset are available (see `results`, and `plots` folders). They can be reproduced with the command below:

```python run-llm-exp.py --multirun model=biomistral-7b-med quantization=gguf quantization.level=q8_0```

## Next Steps
* Complete runs for all compressed model(s) and write results to a file, changing: (1) extent of quantization, (2) quantization technique, (3) model

## Fixes
* Fixed calculation of FP's, FN's
