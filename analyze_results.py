import os
import json
from pprint import pp
import matplotlib.pyplot as plt
import seaborn as sns

model_families = list(set([r.split('-')[0] for r in os.listdir('results')]))
family_x_idx_pairs=[(r.split('-')[0], idx) for idx, r in enumerate(os.listdir('results'))]

# GOAL: transform models as follows: models={model_family:list_of_directories} -> {model_family:list_of_json_files}
# initialize models
models = {m:[] for m in model_families}
# add families as keys
for (m,i) in family_x_idx_pairs:
    models[m].append(os.listdir('results')[i])
# transform values : path to model folder -> path to json results file
for m in models.keys():
    # models[m]: list of folder names corresponding to the the model family m
    
    # get json files from each folder
    m_files = []
    for idx, folder in enumerate(models[m]):
        subfolder_files = os.listdir(os.path.join('results', models[m][idx])) # [*.csv, *.json]
        for file in subfolder_files:
            if 'json' in file:
                m_files.append(os.path.join('results',models[m][idx],file))
    models[m] = m_files

# get accuracy for all families, models, k-quants
family_model_summaries = {k:None for k in model_families}
for family in model_families:
    model_summaries = {}
    for m in models[family]:
        curr_summary = {}
        with open(m, 'r') as file:
            curr_data = json.load(file)
        exp_name = curr_data['experiment_name'].replace('qwen','Qwen').replace('_ollama_pqa_labeled', '').replace('it_gguf_', '').replace('q','').replace('_k_m', '').replace('_k','').replace('_0','').replace('med_gguf_', '').replace('7b_gguf_', '')
        model = '-'.join(exp_name.split('-')[:-1])
        bits = exp_name.split('-')[-1]
        curr_summary[bits] = curr_data['metrics']['accuracy'] # METRIC
        if model_summaries.get(model, None) == None:
            model_summaries[model] = curr_summary
        else:
            model_summaries[model].update(curr_summary)
        print(f"Model: {model}, Bits: {bits}, Acc: {curr_data['metrics']['accuracy']}") # METRIC
    if family_model_summaries.get(family,None) == None:
        family_model_summaries[family] = model_summaries
    else:
        family_model_summaries[family].update(model_summaries)

# --- 4 families in 1 figure ---
qwen_models = list(family_model_summaries['qwen2.5'].keys()) # 1.5, 0.5, 3, 7
# sort the dictionary of bit-accuracy (key-value) pairs by the key
qwen15 = {bit: family_model_summaries['qwen2.5'][qwen_models[0]][bit] for bit in sorted(family_model_summaries['qwen2.5'][qwen_models[0]])}
qwen05 = {bit: family_model_summaries['qwen2.5'][qwen_models[1]][bit] for bit in sorted(family_model_summaries['qwen2.5'][qwen_models[1]])}
qwen30 = {bit: family_model_summaries['qwen2.5'][qwen_models[2]][bit] for bit in sorted(family_model_summaries['qwen2.5'][qwen_models[2]])}
qwen70 = {bit: family_model_summaries['qwen2.5'][qwen_models[3]][bit] for bit in sorted(family_model_summaries['qwen2.5'][qwen_models[3]])}

gemma_models = list(family_model_summaries['gemma3'].keys()) # 12, 1, 4
gemma12 = {bit: family_model_summaries['gemma3'][gemma_models[0]][bit] for bit in sorted(family_model_summaries['gemma3'][gemma_models[0]])}
gemma1 = {bit: family_model_summaries['gemma3'][gemma_models[1]][bit] for bit in sorted(family_model_summaries['gemma3'][gemma_models[1]])}
gemma4 = {bit: family_model_summaries['gemma3'][gemma_models[2]][bit] for bit in sorted(family_model_summaries['gemma3'][gemma_models[2]])}

biomistral_models = list(family_model_summaries['biomistral'].keys())
biomistral7_med = {bit: family_model_summaries['biomistral'][biomistral_models[0]][bit] for bit in sorted(family_model_summaries['biomistral'][biomistral_models[0]])}

llama3_med_models = list(family_model_summaries['llama3'].keys())
llama3_med = {bit: family_model_summaries['llama3'][llama3_med_models[0]][bit] for bit in sorted(family_model_summaries['llama3'][llama3_med_models[0]])}

fig, ax = plt.subplots(2, 2, figsize=(12,10))
# qwen
ax[0][0].plot(qwen05.keys(), qwen05.values(), '.-.', label='qwen2.5-0.5b')
ax[0][0].plot(qwen15.keys(), qwen15.values(), '.-.', label='qwen2.5-1.5b')
ax[0][0].plot(qwen30.keys(), qwen30.values(), '.-.', label='qwen2.5-3b')
ax[0][0].plot(qwen70.keys(), qwen70.values(), '.-.', label='qwen2.5-7b')#, color='red')
ax[0][0].legend()
ax[0][0].set_title('Qwen2.5')
ax[0][0].set_ylabel('Accuracy [%]')
ax[0][0].set_xlabel('k-quantization level [bits]')
ax[0][0].set_ylim(0, 1)

# gemma
ax[0][1].plot(gemma1.keys(), gemma1.values(), '.-.', label='gemma3-1b')
ax[0][1].plot(gemma4.keys(), gemma4.values(), '.-.', label='gemma3-4b')
ax[0][1].plot(gemma12.keys(), gemma12.values(), '.-.', label='gemma3-12b')
ax[0][1].legend()
ax[0][1].set_title('Gemma-3')
ax[0][1].set_ylabel('Accuracy [%]')
ax[0][1].set_xlabel('k-quantization level [bits]')
ax[0][1].set_ylim(0, 1)

# llama
ax[1][0].plot(llama3_med.keys(), llama3_med.values(), '.-.', label='llama3_med_7B')
ax[1][0].legend()
ax[1][0].set_title('Llama3-med')
ax[1][0].set_ylabel('Accuracy [%]')
ax[1][0].set_xlabel('k-quantization level [bits]')
ax[1][0].set_ylim(0, 1)

# biomistral
ax[1][1].plot(biomistral7_med.keys(), biomistral7_med.values(), '.-.', label='biomistral_med_7B')
ax[1][1].legend()
ax[1][1].set_title('Biomistral-med')
ax[1][1].set_ylabel('Accuracy [%]')
ax[1][1].set_xlabel('k-quantization level [bits]')
ax[1][1].set_ylim(0, 1)
plt.suptitle('K-Quantization Level vs. Accuracy for Different LLM-Examiner Families')
plt.tight_layout()
plt.show()

# --- The three 7B models in 1 figure ---
qwen_models = list(family_model_summaries['qwen2.5'].keys()) # 1.5, 0.5, 3, 7
# sort the dictionary of bit-accuracy (key-value) pairs by the key
qwen15 = {bit: family_model_summaries['qwen2.5'][qwen_models[0]][bit] for bit in sorted(family_model_summaries['qwen2.5'][qwen_models[0]])} # qwen15 := qwen2.5 1.5B parameter model
qwen05 = {bit: family_model_summaries['qwen2.5'][qwen_models[1]][bit] for bit in sorted(family_model_summaries['qwen2.5'][qwen_models[1]])} # qwen05 := qwen2.5 0.5B parameter model
qwen30 = {bit: family_model_summaries['qwen2.5'][qwen_models[2]][bit] for bit in sorted(family_model_summaries['qwen2.5'][qwen_models[2]])}
qwen70 = {bit: family_model_summaries['qwen2.5'][qwen_models[3]][bit] for bit in sorted(family_model_summaries['qwen2.5'][qwen_models[3]])}

gemma_models = list(family_model_summaries['gemma3'].keys()) # 12, 1, 4
gemma12 = {bit: family_model_summaries['gemma3'][gemma_models[0]][bit] for bit in sorted(family_model_summaries['gemma3'][gemma_models[0]])} # gemma12 := gemma3 12B parameter model
gemma1 = {bit: family_model_summaries['gemma3'][gemma_models[1]][bit] for bit in sorted(family_model_summaries['gemma3'][gemma_models[1]])}
gemma4 = {bit: family_model_summaries['gemma3'][gemma_models[2]][bit] for bit in sorted(family_model_summaries['gemma3'][gemma_models[2]])}

biomistral_models = list(family_model_summaries['biomistral'].keys())
biomistral7_med = {bit: family_model_summaries['biomistral'][biomistral_models[0]][bit] for bit in sorted(family_model_summaries['biomistral'][biomistral_models[0]])}

llama3_med_models = list(family_model_summaries['llama3'].keys())
llama3_med = {bit: family_model_summaries['llama3'][llama3_med_models[0]][bit] for bit in sorted(family_model_summaries['llama3'][llama3_med_models[0]])}

fig, ax = plt.subplots(1, 1)#, figsize=(12,10))
# qwen
ax.plot(qwen70.keys(), qwen70.values(), '.-.', label='qwen2.5-7b')
ax.plot(llama3_med.keys(), llama3_med.values(), '.-.', label='llama3_med_7B')
ax.plot(biomistral7_med.keys(), biomistral7_med.values(), '.-.', label='biomistral_med_7B')
ax.legend()
#ax[0][0].set_title('Qwen2.5')
ax.set_ylabel('Accuracy [%]')
ax.set_xlabel('k-quantization level [bits]')
ax.set_ylim(0, 1)
#plt.suptitle('')
plt.title('K-quantization Level vs. Accuracy for 7B parameter LLM-Examiners',pad=15)
plt.tight_layout()
plt.show()