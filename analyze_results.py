import os
import json
from pprint import pp
import matplotlib.pyplot as plt
from adjustText import adjust_text # new pip requirement

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
for (family, idx) in family_idx_pairs:
    family_to_rdirlist[family].append(os.listdir('results')[idx])

#pp(family_to_rdirlist);quit()

# family_to_rjsonlist <- family_to_rdirlist
family_to_rjsonlist = {family:[] for family in model_families}    
for family in model_families:
    family_rjsons = [] # family results jsons
    for idx, rdir in enumerate(family_to_rdirlist[family]):
        rdir_files = os.listdir(os.path.join('results', rdir))
        #print(rdir_files)
        #assert len(rdir_files) == 2 # [*.csv, *.json]
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

def get_family_to_metrics(metric: str | list[str,str] = 'accuracy', metric_type: str = 'model'):
    family_to_metrics = {family: None for family in model_families} # { family : {f'{family}-{param_size}' : { k-quant : metric } } }
    #metric:str = 'accuracy'
    for family in model_families:
        model_summaries = {} # {f'{family}-{param_size}' : { k-quant : metric } }
        for jsonf in family_to_rjsonlist[family]:
            data = get_results_dict(jsonf)
            exp_name = clean_experiment_name(data['experiment_name'])
            model = model_name_from_exp_name(exp_name) # e.g., Qwen2.5-0.5b, Qwen2.5-7b, ...
            kquant = kquant_from_exp_name(exp_name)
            
            kquant_to_metric = {} #{2: None, 3: None, 4: None, 5: None, 6: None, 8: None}
            #kquant_to_metric[kquant] = data['metrics'][metric]

            if metric_type == 'model':
                kquant_to_metric[kquant] = data['metrics'][metric]
            elif metric_type == 'system':
                kquant_to_metric[kquant] = data['metrics'][metric[0]][metric[1]] # e.g, data['metrics']['gpu_report']['peak_gpu_usage']
            
            if model_summaries.get(model,None) == None:
                model_summaries[model] = kquant_to_metric
            else:
                model_summaries[model].update(kquant_to_metric)
                
        family_to_metrics[family] = model_summaries
    return family_to_metrics


# --- Create Plots [Still in the works 🚧] ---
def plot_metric_for_family(family_name: str, metric_name:str, ax):
    """
    For the provided family_name (e.g., Qwen2.5) and metric_name (e.g., accuracy), generate a plot as follows:
    x axis: goes from 8-bit to 2-bit quantization, so as a reader moves their eyes along the x-axis they see y-vals corresponding to a greater extent of quantization
    y axis: [metric_name]
    """
    family_to_metrics = get_family_to_metrics(metric_name)
    model_names = sorted(family_to_metrics[family_name].keys())
    for model in model_names:
        k_to_val = family_to_metrics[family_name][model]
        x_vals = ['8', '6', '5', '4', '3', '2']
        y_vals = [k_to_val.get(k, None) for k in x_vals]
        ax.plot(x_vals, y_vals, '.-.', label=model)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f'{k}-bit' for k in x_vals])
        #sorted_k = sorted([int(k) for k in k_to_val.keys()])
        #ax.plot(sorted_k[::-1], [k_to_val[str(k)] for k in sorted_k[::-1]], '.-.', label=model) # ax.plot(sorted(k_to_val), [k_to_val[k] for k in sorted(k_to_val)], '.-.', label=model)
    ax.set_title(f'{family_name}')
    ax.set_xlabel('Quantization level [bits] \n (right = more compression)')  #ax.set_xlabel('k-quantization level [bits]')
    rename_model_metric = {'accuracy': 'Accuracy (%)',
                           'precision': 'Precision',
                           'recall': 'Recall',
                           'f1-score': 'F1-Score',
                           'abstention_rate': 'Abstention Rate'}
    ax.set_ylabel(f'{rename_model_metric[metric_name]}')
    ax.set_ylim(0,1)
    ax.legend()

def plot_system_metric_for_family(family_name: str, metric_name: list[str,str], ax):
    family_to_metrics = get_family_to_metrics(metric_name, metric_type='system')
    model_names = sorted(family_to_metrics[family_name].keys())
    for model in model_names:
        k_to_val = family_to_metrics[family_name][model]
        # plot x-axis from 8-bit to 2-bit quantization
        x_vals = ['8', '6', '5', '4', '3', '2']
        y_vals = [k_to_val.get(k, None) for k in x_vals]
        ax.plot(x_vals, y_vals, '.-.', label=model)
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f'{k}-bit' for k in x_vals])
        #sorted_k = sorted([int(k) for k in k_to_val.keys()], reverse=True)
        #ax.plot(sorted_k, [k_to_val[str(k)] for k in sorted_k], '.-.', label=model) # ax.plot(sorted(k_to_val), [k_to_val[k] for k in sorted(k_to_val)], '.-.', label=model)
    ax.set_title(f'{family_name}')
    ax.set_xlabel('Quantization level [bits] \n (right = more compression)') #ax.set_xlabel('k-quantization level [bits]')
    rename_sys_metric = {'peak_mem_util': 'Peak memory utilization (%)',
                         'avg_load': 'Average GPU Load (%)',
                         'avg_latency': 'Avg. Latency (s)',
                         'throughput_prompts_per_sec': 'Throughput (prompts/s)'}
    ax.set_ylabel(f'{rename_sys_metric[metric_name[1]]}')
    if rename_sys_metric[metric_name[1]] == 'Throughput (prompts/s)':
        #print([k_to_val[k] for k in sorted(k_to_val)])
        ax.set_ylim(0,max(1,3+max([k_to_val[k] for k in sorted(k_to_val)])))
    else:
        #print(metric_name[1])
        ax.set_ylim(0,1)
    ax.legend()
    
def create_all_families_comparison_plot(family_names: list[str], metric_name: str|list[str,str], metric_type: str | list[str,str] = 'model'):
    fig, ax = plt.subplots(2, 2, figsize=(8,8))
    if metric_type == 'model':
        plot_metric_for_family(family_names[0], metric_name, ax[0,0])
        plot_metric_for_family(family_names[1], metric_name, ax[0,1])
        plot_metric_for_family(family_names[2], metric_name, ax[1,0])
        plot_metric_for_family(family_names[3], metric_name, ax[1,1])
        
    elif metric_type == 'system':
        plot_system_metric_for_family(family_names[0], metric_name, ax[0,0])
        plot_system_metric_for_family(family_names[1], metric_name, ax[0,1])
        plot_system_metric_for_family(family_names[2], metric_name, ax[1,0])
        plot_system_metric_for_family(family_names[3], metric_name, ax[1,1])

    rename_metric = {'accuracy': 'Accuracy (%)',
                     'precision': 'Precision',
                     'recall': 'Recall',
                     'f1-score': 'F1-Score',
                     'abstention_rate': 'Abstentation Rate',
                     'peak_mem_util': 'Peak Memory Utilization (%)',
                     'avg_load': 'Average GPU Load (%)',
                     'avg_latency': 'Avg. Latency (s)',
                     'throughput_prompts_per_sec': 'Throughput (prompts/s)'}
    if metric_type == 'system':
        metric_name = metric_name[1]
    plt.suptitle(f'{rename_metric[metric_name]} vs. Quantization Across Families')
    plt.tight_layout() #plt.subplots_adjust(wspace=0.3, hspace=0.3, right=0.95)
    plt.savefig(f'plots/kquant_vs_{metric_name}.png')

# plot accuracy vs. peak gpu usage
#print(f"Peak Memory Used: {df['metrics']['gpu_report']['peak_mem_util'] * df['metrics']['gpu_report']['mem_total'] / 1000:.2f} GB")
def plot_accuracy_vs_peak_mem():
    acc_mem_points = []

    for family in model_families:
        for jsonf in family_to_rjsonlist[family]:
            try:
                data = get_results_dict(jsonf)
                acc = data['metrics']['accuracy']
                #latency = data['metrics']['latency_report']['avg_latency']
                peak_mem = data['metrics']['gpu_report']['peak_mem_util'] * data['metrics']['gpu_report']['mem_total'] / 1000
                exp_name = clean_experiment_name(data['experiment_name'])
                model = model_name_from_exp_name(exp_name)
                k = kquant_from_exp_name(exp_name)
                label=f'{model}-{k}'
                acc_mem_points.append({
                    'peak_mem': peak_mem,
                    'accuracy': acc,
                    'label': label,
                    'family': family,
                    'dominated': False
                })
            except Exception as e:
                print(f"Skipping {jsonf}: {e}")

    plt.figure(figsize=(10,6))
    texts = [] # for adjusttext library
    
    # find pareto set
    for i, p1 in enumerate(acc_mem_points):
        for j, p2 in enumerate(acc_mem_points):
            if (p2["accuracy"] >= p1["accuracy"] and p2["peak_mem"] <= p1["peak_mem"] and (p2["accuracy"] > p1["accuracy"] or p2["peak_mem"] < p1["peak_mem"])):
                acc_mem_points[i]["dominated"] = True
                break
                
    family_colors = {
        'qwen2.5': 'red',
        'gemma3': 'green',
        'biomistral': 'blue',
        'llama3': 'orange'
    }

    # plot points, one point at a time
    for point in acc_mem_points:
        color = family_colors[point["family"]]        
        size = 50 if not point["dominated"] else 20
        edge_color = 'black' if not point["dominated"] else 'none'

        plt.scatter(point['peak_mem'], point['accuracy'],
                    s=size,
                    color=color,
                    edgecolors=edge_color,
                    linewidths=1,
                    label=point["family"] if not point["dominated"] else "",
                    alpha=0.8)
        
        if not point["dominated"]:
            texts.append(plt.text(point["peak_mem"], point["accuracy"], point["label"],
                                 fontsize=8,
                                 weight='bold',
                                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', lw=0.5)))

    adjust_text(texts,
                only_move={'text': 'xy'},
                arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
    
    plt.xlabel('Peak Memory Usage [GB]', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs. Peak Memory Usage for Quantized LLMs', fontsize=14)
    plt.grid(True)
    
    # Custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=fam,
                          markerfacecolor=color, markersize=10)
               for fam, color in family_colors.items()]
    handles.append(plt.Line2D([0], [0], marker='o', color='black', label='Pareto-optimal',
                              markerfacecolor='none', markersize=10, linestyle='None', markeredgewidth=1.5))
    
    plt.legend(handles=handles, title='Model Family')
    plt.tight_layout()
    plt.savefig('plots/clean_accuracy_vs_peak_mem.png', dpi=300)
    

def plot_accuracy_vs_latency():
    acc_latency_points = []

    for family in model_families:
        for jsonf in family_to_rjsonlist[family]:
            try:
                data = get_results_dict(jsonf)
                acc = data['metrics']['accuracy']
                latency = data['metrics']['latency_report']['avg_latency']
                exp_name = clean_experiment_name(data['experiment_name'])
                model = model_name_from_exp_name(exp_name)
                k = kquant_from_exp_name(exp_name)
                label=f'{model}-{k}'
                acc_latency_points.append({
                    'latency': latency,
                    'accuracy': acc,
                    'label': label,
                    'family': family,
                    'dominated': False
                })
            except Exception as e:
                print(f"Skipping {jsonf}: {e}")

    plt.figure(figsize=(10,6))
    texts = [] # for adjusttext library
    
    # find pareto set
    for i, p1 in enumerate(acc_latency_points):
        for j, p2 in enumerate(acc_latency_points):
            if (p2["accuracy"] >= p1["accuracy"] and p2["latency"] <= p1["latency"] and (p2["accuracy"] > p1["accuracy"] or p2["latency"] < p1["latency"])):
                acc_latency_points[i]["dominated"] = True
                break
                
    family_colors = {
        'qwen2.5': 'red',
        'gemma3': 'green',
        'biomistral': 'blue',
        'llama3': 'orange'
    }

    # plot points, one point at a time
    for point in acc_latency_points:
        color = family_colors[point["family"]]
        
        # pt size
        #plot_pt_size_by_model_size = False
        #if plot_pt_size_by_model_size == True:
        #    model_size_map = {'0.5b': 30, '1b': 35, '1.5b': 40, '3b': 50, '4b': 60, '7b': 70, '12b': 80}
        #    model_size = point["label"].split('-')[1]
        #    try:
        #        size = model_size_map[model_size]
        #    except:
        #        size = model_size_map['7b']
        
        size = 50 if not point["dominated"] else 20
        
        edge_color = 'black' if not point["dominated"] else 'none'

        plt.scatter(point['latency'], point['accuracy'],
                    s=size,
                    color=color,
                    edgecolors=edge_color,
                    linewidths=1,
                    label=point["family"] if not point["dominated"] else "",
                    alpha=0.8)
        
        if not point["dominated"]:
            texts.append(plt.text(point["latency"], point["accuracy"], point["label"],
                                 fontsize=8,
                                 weight='bold',
                                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', lw=0.5)))


    adjust_text(texts,
                only_move={'text': 'xy'},
                arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
    
    plt.xlabel('Average Latency (seconds)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs. Latency for Quantized LLMs', fontsize=14)
    plt.grid(True)
    
    # Custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=fam,
                          markerfacecolor=color, markersize=10)
               for fam, color in family_colors.items()]
    handles.append(plt.Line2D([0], [0], marker='o', color='black', label='Pareto-optimal',
                              markerfacecolor='none', markersize=10, linestyle='None', markeredgewidth=1.5))
    
    plt.legend(handles=handles, title='Model Family')
    plt.tight_layout()
    plt.savefig('plots/clean_accuracy_vs_latency.png', dpi=300)

def gen_plots():
    # k-quantization vs. model performanace
    create_all_families_comparison_plot(['qwen2.5', 'gemma3', 'llama3', 'biomistral'], 'accuracy')
    create_all_families_comparison_plot(['qwen2.5', 'gemma3', 'llama3', 'biomistral'], 'precision')
    create_all_families_comparison_plot(['qwen2.5', 'gemma3', 'llama3', 'biomistral'], 'recall')
    create_all_families_comparison_plot(['qwen2.5', 'gemma3', 'llama3', 'biomistral'], 'f1-score')
    create_all_families_comparison_plot(['qwen2.5', 'gemma3', 'llama3', 'biomistral'], 'abstention_rate')
    create_all_families_comparison_plot(['qwen2.5', 'gemma3', 'llama3', 'biomistral'], 'accuracy')
    #logging.info("Performance plots have been generated and saved.")
    print("Performance plots have been generated and saved.")

    
    # k-quantization vs. system performance
    """
    peak_mem_util: feasible deployment | will the model fit on our hardware?
    avg_gpu_load: GPU energy efficiency | how efficiently is the GPU being used during inference? Can I scale this model up to many inferences without wasting GPU cycles? I'm concerned about Energy usage, cooling, cost-per-inference
    std_dev_gpu_load: Real-time systems may have hard requirements on latency ; can I expect stable, consistent resource usage? unexpected spikes?
    avg_latency: UX | key metric for user experience ; can we use this in a live-application? will this model meet our response time SLAs?
    throughput: sclability under load 
    """
    
    # peak mem util
    create_all_families_comparison_plot(family_names=['qwen2.5', 'gemma3', 'llama3', 'biomistral'],
                                        metric_name=['gpu_report', 'peak_mem_util'],
                                        metric_type='system')

    # avg_gpu_load
    create_all_families_comparison_plot(family_names=['qwen2.5', 'gemma3', 'llama3', 'biomistral'],
                                        metric_name=['gpu_report', 'avg_load'],
                                        metric_type='system')
    # avg_latency
    create_all_families_comparison_plot(family_names=['qwen2.5', 'gemma3', 'llama3', 'biomistral'],
                                        metric_name=['latency_report', 'avg_latency'],
                                        metric_type='system')
    # throughput
    create_all_families_comparison_plot(family_names=['qwen2.5', 'gemma3', 'llama3', 'biomistral'],
                                        metric_name=['latency_report', 'throughput_prompts_per_sec'],
                                        metric_type='system')
    print("System plots have been generated and saved.")
    
    plot_accuracy_vs_latency()
    plot_accuracy_vs_peak_mem()
    print("Tradeoff plots have been generated and saved.")
    #logging.info("System plots have been generated and saved.")
    
if __name__ == "__main__":
    gen_plots()