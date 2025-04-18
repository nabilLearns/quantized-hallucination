# medical-llm-hallucination-thesis

Example cli usage of `run-llm-exp.py`, with gemma3 (instruction tuned) 1B parameter model, using the q4_k_m GGUF quantized version.
```
python run-llm-exp.py model=gemma3-1b-it quantization=gguf quantization.level=q4_k_m max_samples=10
```

## Current Progress
* Inference pipeline now available via `run-llm-exp.py`.
* Multiple experiments can be scheduled via Hydra from the cli with the `--multirun` option 
* Pipeline supports hallucination classification experiments for 10 models, with 6 GGUF quantization schemes for each (q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m, q2_k)
* Results: Model and GPU performance metrics are recorded to a JSON file. Data on the models responses are recorded in a csv storing the question asked, ground truth answer, hallucinated answer, model response, and the extracted prediction from that response.

Each experiment records a number of metrics: 
* Confusion matrix: Accuracy, precision, recall, f1 score, support
* Abstention rate
* Inference latency statistics (e.g., peak, avg, std_dev, variance)
* GPU usage statistics (e.g., peak, avg, std_dev for GPU load, utilization)


## Results
* Results for the medically fine-tuned **BioMistral-7B q8_0 GGUF quant** when tested on the pqa-labelled 1k dataset are available (see `results`, and `plots` folders). They can be reproduced with the command below:

```python run-llm-exp.py --multirun model=biomistral-7b-med quantization=gguf quantization.level=q8_0```

## Next Steps
* Complete runs for all compressed model(s) and write results to a file, changing: (1) extent of quantization, (2) quantization technique, (3) model

## Fixes
* Fixed calculation of FP's, FN's
