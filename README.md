# medical-llm-hallucination-thesis

Example cli usage of `run-llm-exp.py`, with gemma3 (instruction tuned) 1B parameter model, using the q4_k_m GGUF quantized version.
```
python run-llm-exp.py model=gemma3-1b-it quantization=gguf quantization.level=q4_k_m max_samples=10
```

## Current Progress
* implemented initial inference pipeline (`llm-classify-run.ipynb`)
    * can run **uncompressed Qwen-2.5 7B** model on the MedHallu `pqa_labelled` subset (currently tested on 32 examples: 16 ground-truth answers, 16 hallucinated; note: there are two answers per row; 1 gt and 1 hallucinated)
    * calculates preliminary metrics: accuracy, precision, recall, f1 score, abstention rate
* pipeline now supports hallucination classification experiments for 10 models, with 6 GGUF quantization schemes for each (q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m, q2_k)
## Next Steps
~~fix calculation of FP's, FN's~~
* do a full run with uncompressed model on 1k example subset (`pqa_labelled`) and write results to a file with metadata
* do runs for compressed model(s) and write results to a file, changing: (1) extent of quantization, (2) quantization technique, (3) model
~~* parse the file with results, put together summary data, create chart(s) to show any trends~~
