import subprocess
import time
import pandas
import os
#extra_models = ['qwen2.5-32b-it','qwen2.5-7b-it-med',]
models = [#'biomistral-7b-med',
          
          #'gemma3-1b-it',
          #'gemma3-4b-it',
          
          #'gemma3-12b-it',
          #'llama3-med',
          
          #'qwen2.5-0.5b-it',
          #'qwen2.5-1.5b-it',
          
          #'qwen2.5-3b-it',
          'qwen2.5-7b-it',
          'qwen2.5-14b-it']
gguf_quants = ['q8_0', 'q6_k', 'q5_k_m', 'q4_k_m', 'q3_k_m', 'q2_k']

# example cli input for one run
# python run-llm-exp.py --multirun model=biomistral-7b-med quantization=gguf quantization.level=q5_k_m

exp_runtimes = {m: {q: 0 for q in gguf_quants} for m in models}
for model in models:
    for quant in gguf_quants:
        print(f"\n Running: {model} @ {quant} \n")
        command = [
            'python',
            'run-llm-exp.py',
            '--multirun', 
            f'model={model}',
            'quantization=gguf',
            f'quantization.level={quant}'
        ]
        start_time = time.time()
        subprocess.run(command)
        end_time = time.time()
        duration = end_time - start_time
        exp_runtimes[f'{model}'][f'{quant}'] = duration
        print(f"✅ Finished {model} @ {quant} in {duration:.2f} seconds\n")
        #subprocess.call(['python', 'run-llm-exp.py', 'model', model, 'quantization', 'gguf', 'quantization.level', quant])
exp_runtimes = pd.DataFrame(exp_runtimes)
exp_runtimes.to_csv('exp_runtimes_A40_94.csv')
