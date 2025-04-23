import os
import json
from pprint import pp
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# original
#model_families = list(set([r.split('-')[0] for r in os.listdir('results')]))

# higher quality code?
def extract_family(dirname: str) -> str:
    """
    Examples
    Input: qwen2.5-1.5b-it_gguf_q3_k_m_ollama_pqa_labeled
    Ouput: qwen2.5

    Input: llama3-med_gguf_q6_k_ollama_pqa_labeled
    Output: llama3
    """
    return dirname.split('-')[0]
    
def all_family_names(results_dir:str) -> list[str]:
    return [extract_family(d) for d in os.listdir(results_dir)]
    
def unique_model_families(results_dir:str) -> list[str]:
    return list(set(all_family_names(results_dir)))
    
model_families = unique_model_families(results_dir='results')

# GOAL: transform models as follows: models={model_family:list_of_directories} -> {model_family:list_of_json_files}

# family_to_rdirlist is an intermediate variable needed to construct family_to_json_list
family_to_rdirlist = {family:[] for family in model_families} # Example model families: 'qwen2.5', 'gemma3'
family_idx_pairs=[(extract_family(dirname), idx) for idx, dirname in enumerate(os.listdir('results'))]
for idx, family in enumerate(family_idx_pairs):
  family_to_rdirlist[family].append(os.listdir('results')[idx])

# family_to_rjsonlist <- family_to_rdirlist
family_to_rjsonlist = {family:[] for family in model_families}    
for family in model_families:
    family_rjsons = [] # family results jsons
    for idx, rdir in enumerate(family_to_rdirlist[family]):
        rdir_files:[str,str] = os.listdir(os.path.join('results', rdir))
        assert len(rdir_files) == 2 # [*.csv, *.json]
        for file in rdir_files:
            if 'json' in file:
                family_rjsons.append(os.path.join('results',rdir,file))
    family_to_rjsonlist[family] = family_rjsons

# --- Metrics ---
def get_results_dict(jsonf:str) -> dict:
    """
    Example: 
    Input: 'results/biomistral-7b_gguf_q6_k_ollama_pqa_labeled/results_biomistral-7b_gguf_q6_k_ollama_pqa_labeled.json'
    Output: {
        'experiment_name': EXPERIMENT_NAME,
        'semi-processed_results': processed_results,
        'metrics': {
            #'gpu_peak_memory_gb': peak_memory_gb,
            'gpu_report': gpu_report,
            'latency_report': latency_report,
            'cm': metrics_dict['cm'].tolist(),
            'accuracy': metrics_dict['accuracy'],
            'precision': metrics_dict['precision'],
            'recall': metrics_dict['recall'], 
            'f1-score': metrics_dict['f1-score'],
            'support': metrics_dict['support'],
            'abstention_rate': metrics_dict['abstention_rate'], 
            'fn_difficulty_counts': metrics_dict['fn_difficulty_counts'].to_dict(),
            'fn_category_counts': metrics_dict['fn_category_counts'].to_dict(),
            'tp_difficulty_counts': metrics_dict['tp_difficulty_counts'].to_dict(),
            'tp_category_counts': metrics_dict['tp_category_counts'].to_dict(),
            'fnr_difficulty': metrics_dict['fnr_difficulty'].to_dict(),
            'fnr_category': metrics_dict['fnr_category'].to_dict(),
            'tpr_difficulty': metrics_dict['tpr_difficulty'].to_dict(),
            'tpr_category': metrics_dict['tpr_category'].to_dict()
        }
    }
    """
    data = None
    try:
        with open(jsonf, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Could not load {jsonf}: {e}")
        raise
    assert data != None 
    return data

def clean_experiment_name(messy_name: str) -> str:
    """
    converts messy_name -> f'{family}-{size}-{k-quant}' format
    
    Example
    Input: biomistral-7b_gguf_q6_k_ollama_pqa_labeled
    Output: biomistral-7b-6
    
    Input: gemma3-1b-it_gguf_q8_0_ollama_pqa_labeled
    Ouput: gemma3-1b-8
    
    Input: qwen2.5-0.5b-it_gguf_q3_k_m_ollama_pqa_labeled
    Output: Qwen2.5-0.5b-3
    """
    return (messy_name.replace('qwen','Qwen')
            .replace('_ollama_pqa_labeled','')
            .replace('it_gguf_','')
            .replace('q','')
            .replace('_k_m', '')
            .replace('_k','')
            .replace('_0','')
            .replace('med_gguf_', '')
            .replace('7b_gguf_', '7b-')
           )
    
def model_name_from_exp_name(exp_name:str) -> str:
    """
    Examples:
    Qwen2.5-0.5b-3 -> Qwen2.5-0.5b, 
    gemma3-12b-8   -> gemma3-12b,
    gemma3-4b-2    -> gemma3-4b,
    """
    return '-'.join(exp_name.split('-')[:-1])

def kquant_from_exp_name(exp_name:str) -> str:
    """Example: 
    Input: gemma3-4b-2
    Output: '2'
    
    Input: biomistral-7b-5
    Ouput: '5'
    """
    return exp_name.split('-')[-1]

def get_family_to_metrics(metric: str = 'accuracy'):
    family_to_metrics = {family: None for family in model_families} # { family : {f'{family}-{param_size}' : { k-quant : metric } } }
    metric:str = 'accuracy'
    for family in model_families:
        model_summaries = {} # {f'{family}-{param_size}' : { k-quant : metric } }
        for jsonf in family_to_jsonlist[family]:
            data = get_results_dict(jsonf)
            exp_name = clean_experiment_name(data['experiment_name'])
            model = model_name_from_exp_name(exp_name) # e.g., Qwen2.5-0.5b, Qwen2.5-7b, ...
            kquant = kquant_from_exp_name(exp_name)
            
            kquant_to_metric = {} #{2: None, 3: None, 4: None, 5: None, 6: None, 8: None}
            kquant_to_metric[kquant] = data['metrics'][metric]
            
            if model_summaries.get(model,None) == None:
                model_summaries[model] = kquant_to_metric
            else:
                model_summaries[model].update(kquant_to_metric)
                
        family_to_metrics[family] = model_summaries
    return family_to_metrics

family_to_metrics = get_family_to_metrics('accuracy')


# --- Create Plots [Still in the works 🚧] ---
def plot_4_families(metric: str = 'accuracy'):
    pass
def plot_7b_models():
    pass

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