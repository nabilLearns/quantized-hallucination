import os
import pandas as pd
import torch as t
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPTQConfig, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay #,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import gc # garbage collector interface
import pprint
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm # inference progress bar
import math           # math.ceil is used for computing # batches (inference)
from dataclasses import dataclass # for HallucinationMetrics class ; not used atm
#from llama_cpp import Llama # for gguf file ; not used atm
import yaml # for cfgs
import logging
from abc import ABC, abstractmethod
import json # final results for a run will be written to a json file

# script specific (not used in the jupyter notebook version)
from omegaconf import DictConfig, OmegaConf
import hydra
import subprocess

# --- Configuration ---
#MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
#DATASET_NAME = 'UTAustin-AIHealth/MedHallu'
#DATASET_CONFIG = 'pqa_labeled'
#BATCH_SIZE = 8
#MAX_SAMPLES = 1000
#QUANTIZATION_MODE = 'gguf' #'8bit_bnb' #'uncompressed'
#EXPERIMENT_NAME = f'{MODEL_NAME.lower()}_{QUANTIZATION_MODE.lower()}_{DATASET_CONFIG.lower()}'

# --- Setup logging ---
log = logging.getLogger(__name__)

# --- Hydra replaces the python logger so the logging setup below doesn't do anything (?) ---
#logging.basicConfig(
#    filename='example.log',
#    encoding='utf-8',
#    format='%(asctime)s %(message)s',
#    datefmt='%m/%d/%Y %I:%M:%S %p',
#    level=logging.INFO)
#print("Logger setup completed.")
#log.info("Logger setup completed.")
# --------------------------------

# --- Helper Classes/Functions ---
class AbstractModelLoader(ABC):
    @abstractmethod
    def _build_tokenizer(self, cfg):
        pass
    @abstractmethod
    def _build_hf_model(self, cfg):
        pass
    @abstractmethod
    def _build_gguf_model(self, cfg):
        pass
    @abstractmethod
    def load(self):
        pass

class ModelLoader(AbstractModelLoader):
    def __init__(self, model_family: str, quantization_method: str, quantization_level: Optional[str]=None, device:str="auto"):
        self.model_family = model_family
        self.quantization_method = quantization_method
        self.quantization_level = quantization_level
        self.device = device
        self.model_configs = {
            'qwen': {
                'base_hf_id': 'Qwen/Qwen2.5-7B-Instruct',
                'gguf_hf_id': 'bartowski/Qwen2.5-7B-Instruct-GGUF',
                'gguf_files': {
                    'Q8_0':   'Qwen2.5-7B-Instruct-Q8_0.gguf',
                    'Q6_K_L': 'Qwen2.5-7B-Instruct-Q6_K_L.gguf',
                    'Q6_K': 'Qwen2.5-7B-Instruct-Q6_K.gguf',
                    'Q5_K_S': 'Qwen2.5-7B-Instruct-Q5_K_S.gguf',
                    'Q4_K_M': 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
                    'Q3_K_XL': 'Qwen2.5-7B-Instruct-Q3_K_XL.gguf',
                    'Q2_K': 'Qwen2.5-7B-Instruct-Q2_K.gguf'
                },
                'tokenizer_kwargs': {'padding_side': 'left'},
            },
            'llama3-med': {
                'base_hf_id': 'm42-health/Llama3-Med42-8B',
                'gguf_hf_id': 'mradermacher/Llama3-Med42-8B-GGUF',
                'gguf_files': {
                    'Q8_0': 'Llama3-Med42-8B.Q8_0.gguf',
                    'Q6_K': 'Llama3-Med42-8B.Q6_K.gguf',
                    'Q5_K_S': 'lama3-Med42-8B.Q5_K_S.gguf',
                    'Q5_K_M': 'Llama3-Med42-8B.Q5_K_M.gguf',
                    'IQ4_XS': 'Llama3-Med42-8B.IQ4_XS.gguf',
                    'Q4_K_S': 'Llama3-Med42-8B.Q4_K_S.gguf',
                    'Q4_K_M': 'Llama3-Med42-8B.Q4_K_M.gguf',
                    # add llama levels
                },
                'tokenizer_kwargs': {}
            }
            # We can add more families later inshaAllah (・ω<)
        }
        self.config = {}
        try:
            self.config = self.model_configs[self.model_family]
        except:
            log.error(f"Invalid model family. We only support the following families: {str(list(self.model_configs.keys()))}")
    
    def _get_tokenizer_kwargs():
        # default
        tokenizer_kwargs = {
            'trust_remote_code': True,
            **self.config.get('tokenizer_kwargs',{})
        }
        if 'gguf' in self.quantization_method:
            try:
                tokenizer_kwargs['gguf_file'] = self.config['gguf_files'][f'{self.quantization_level}']
            except:
                log.error(f"Tokenizer Kwargs: Was not able to extract {self.model_family} {self.quantization_level} gguf file from config.")
        return tokenizer_kwargs

    def _get_model_kwargs():
        # default
        model_kwargs = {
            'quantization_config': None,
            'device_map': "auto",
            'attn_implementation': "eager", # can use 'flash-attention-2' if user nvidia ampere or newer
            'torch_dtype': t.float16, # depends on gpu being used ; 
            'trust_remote_code': True,
            #**self.config.get('model_kwargs',{})
        }
        if 'gguf' in self.quantization_method:
            try:
                model_kwargs['gguf_file'] = self.config['gguf_files'][f'{self.quantization_level}']
            except:
                log.error(f"Model Kwargs: Was not able to extract {self.model_family} {self.quantization_level} gguf file from config.")        
        elif 'uncompressed' in self.quantization_method:
            log.info("Loading uncompressed model.")
        # add if statements to modify model_kwargs['quantization_config'] if we add more quantization methods
        # if self.quantization_methods == .... and self.quantization_level == ...
        # model_kwargs['quantization_config'] = quantization_config
        
        return model_kwargs
            
    def _build_tokenizer(self):
        tokenizer_path = self.config['base_hf_id']; log.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer_kwargs = _get_tokenizer_kwargs()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
        
    def _build_hf_model(self):
        model_kwargs = self._get_model_kwargs()
        model = AutoModelForCausalML(self.config['base_hf_id'], **model_kwargs)
        log.info(f"Successfully loaded {self.config['base_hf_id']} model.")
        return model
        
    def _build_gguf_model(self):
        model_kwargs = self._get_model_kwargs()
        model = AutoModelForCausalML(self.config['gguf_hf_id'], **model_kwargs)
        log.info(f"Successfully loaded {self.config['gguf_hf_id']} model.")
        return model

    def _download_gguf_to_local_machine(self) -> str:
        try:
            subprocess.run(["huggingface-cli", "download", f"{self.config['gguf_hf_id']}", "--include", f"{self.config['gguf_files'][f'{self.quantization_level}']}"])
            return 'Sucess!'
        except:
            log.error("Was not able to download gguf model / gguf files from hugging face.")
            return 'Failed.'
                
    def build_model_and_tokenizer(self):
        model, tokenizer = None, None
        print(f"Loading model: {self.model_family}]") # \nHyperparams: {model_kwargs}")
        if 'gguf' in self.quantization_method:
            status = self._download_gguf_to_local_machine()
            if status == 'Failed.':
                pass
            elif status == 'Success!':
                tokenizer = self._build_tokenizer()
                model = self._build_gguf_model()
                log.info("GGUF Model loaded successfully!")
        else:
            tokenizer = self._build_tokenizer()
            model = self._build_hf_model()
            log.info("Model loaded successfully!")
        return model, tokenizer

class DatasetLoader():
    def __init__(self, MAX_SAMPLES=1000, DATASET_NAME='UTAustin-AIHealth/MedHallu', DATASET_SPLIT='pqa_labelled'):
        self.max_samples = MAX_SAMPLES
        self.dataset_name = DATASET_NAME
        self.dataset_split = DATASET_SPLIT
        
    def build_dataset(self):
        log.info(f'Loading {self.dataset_name} dataset ({self.dataset_split})...')
        ds = load_dataset(self.dataset_name, self.dataset_split)
        dataset = ds['train'] # use train split which has 1k labelled samples
        if self.max_samples is not None:
            print(f'Limiting dataset to {self.max_samples} samples for testing.')
            dataset = dataset.select(range(self.max_samples)) # for N rows, there are 2*N answers to classify (1 gt, 1 hallucinated) 
        return dataset

def clean_gpu():
    if 'model' in globals():
        print("Deleting existing global 'model' variable.")
        del globals()['model']
    if 'classifier' in globals():
        print("Deleting existing bloal 'classifier' variable.")
        del globals()['classifier']
    gc.collect()
    t.cuda.empty_cache()

# fcn group 1 -> combine into a class?
def format_prompt_chatml(knowledge: str, question: str, answer: str, prompt_style="original", sys_prompt_style="original") -> List[dict]:
    """Put together world knowledge, a medical question, and a medical answer together in a prompt according to requested prompt_style"""
    
    few_shot_not_sure_user_content = f"""
    World Knowledge: [Example Knowledge Snippet]
    Question: [Example Question]
    Answer: [Example Factual Answer]
    Your Judgement: 0
    
    World Knowledge: [Example Knowledge Snippet 2]
    Question: [Example Question 2]
    Answer: [Example Hallucinated Answer]
    Your Judgement: 1
    
    World Knowledge: [Example Knowledge Snippet 3 - where answer might be ambiguous or knowledge insufficient]
    Question: [Example Question 3]
    Answer: [Example Ambiguous Answer or Answer unrelated to Knowledge]
    Your Judgement: 2
    
    --- Now your turn ---
    World Knowledge: {knowledge}
    Question: {question}
    Answer: {answer}
    
    Return just '0' (factual), '1' (hallucinated), or '2' (not sure).
    Your Judgement:"""

    original_user_content = f""""
    World Knowledge: {knowledge}
    Question: {question}
    Answer: {answer}

    Return just an integer value, '0' if the answer is factual, and '1' if the answer is hallucinated. No letter or word, just the integer value.
    
    Your Judgement:"""

    user_prompt = None
    if prompt_style == "few_shot_not_sure":
        user_prompt = few_shot_not_sure_user_content
    elif prompt_style == "original":
        user_prompt = original_user_content

    sys_prompt = None
    with open('prompts.yaml', 'r') as f:
        prompts_file = yaml.safe_load(f)
    try:
        sys_prompt = prompts_file['system_prompts'][sys_prompt_style]
    except:
        sys_prompt = prompts_file['system_prompts']['original']
        
    messages = [
        {"role": "system", "content": sys_prompt.strip()}, # remove leading/trailing whitespaces with .strip()
        {"role": "user", "content": user_prompt.strip()} # original meaning, from the MedHallu paper
    ]
    return messages
def create_prompts(dataset):
    all_prompts = [] # all string prompts
    #all_ground_truths = [] # corresponding labels for each prompt (0: truth, 1: hallucinated)
    
    print("Preparing prompts")
    for i, row in enumerate(dataset):
        knowledge = row["Knowledge"]
        question = row["Question"]
        hallucinated_answer = row["Hallucinated Answer"]
        ground_truth_answer = row["Ground Truth"]
    
        # create prompts for hallucinated and ground truth answers
        prompt_hallucinated = format_prompt_chatml(knowledge, question, hallucinated_answer)
        prompt_truth = format_prompt_chatml(knowledge, question, ground_truth_answer)
    
        all_prompts.append(prompt_hallucinated)
        #all_ground_truths.append(1)
        all_prompts.append(prompt_truth)
        #all_ground_truths.append(0)
    print("Prompts are prepared.")
    log.info("Prompts are prepared.")
    return all_prompts
# TODO: measure inference, throughput, TTS?
def classify_med_answers(prompts, BATCH_SIZE=8):
    print(f"Starting batch inference on {len(prompts)} prompts...")
    log.info(f"Starting batch inference on {len(prompts)} prompts...")
    outputs = []
    num_batches=math.ceil(len(prompts) / BATCH_SIZE)
    with t.no_grad():
        # show inference progress bar with tqdm
        for batch_idx in tqdm(range(num_batches), desc="Classifying Batches", unit="batch"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE,len(all_prompts))
            batch_prompts = prompts[start_idx:end_idx]
            batch_output = classifier(batch_prompts,
                                      #batch_size=BATCH_SIZE # redundant
                                      max_new_tokens=3, # should this be a hyperparam we include in a cfg file?
                                      pad_token_id=tokenizer.pad_token_id,
                                      eos_token_id=tokenizer.eos_token_id,
                                      do_sample=False,
                                      repetition_penalty=1.2) 
            outputs.extend(batch_output)
    
        # all at once
        #outputs = classifier(
        #    all_prompts,
        #    max_new_tokens=3,
        #    batch_size=BATCH_SIZE,
        #    pad_token_id=tokenizer.pad_token_id,
        #    eos_token_id=tokenizer.eos_token_id,
        #    do_sample=False,
        #    repetition_penalty=1.2
        #)
    log.info("Inference complete.")
    print("Inference complete.")
    return outputs

# fcn group 2 -> combine into a class?
def extract_prediction(generated_text):
    """Extract the '0' or '1' from generated text, in case model does not listen to instructions and adds other tokens"""
    text = generated_text.strip()
    text_start = text[-10:]
    #print("text start: ", text_start)
    if '0' in text_start:
        return 0
    elif '1' in text_start:
        return 1
    elif '2' in text_start:
        return 2
    else:
        #print(f"Could not parse '0' or '1' from model output: {text}")
        return -1
        
def extract_all_binary_predictions(outputs) -> Tuple[List[int], list]: # e
    """
    for output in outputs:
        1. filter out original prompt, # b/c model output *includes the input prompt* as well as new generated text
        2. extract binary prediction, and
        3. append to lists for predictions and raw model outputs
    """
    
    predictions = []
    raw_outputs = []
    log.info("Processing Results.")
    print("Processing Results.")
    for i, output in enumerate(outputs):
        model_response = None
        try:
            full_chat = output[0]['generated_text'] # this INCLUDES the prompt ; we only want newly generated text
            assistant_response_dict = full_chat[-1] # full_chat[0]: system, full_chat[1]: user, full_chat[2]: assistant
            #print(assistant_response_dict)
            model_response = None
            if assistant_response_dict['role'] == 'assistant':
                model_response = assistant_response_dict['content']
        except:
            pass
        
        pred = extract_prediction(model_response)
        predictions.append(pred)
        raw_outputs.append(model_response) # store the raw '!!!!!' or '0' or '1'
    log.info("Results Processed.")
    print("Results Processed.")
    return predictions, raw_outputs

# fcn group 3 -> combine into a class?
def get_hallucination_info(info_type: str, dataset): # how do I type hint a hugging face (hf) dataset?
    """
    construct lists for the type and difficulty of hallucinated answers in all_prompts
    all_ground_truths:        [    1,     0,        1,    0, ...]
    hallucination_difficulty: ["easy", None, "medium", None, ...]
    """
    hallucination_difficulty = [None]*dataset.num_rows*2
    hallucination_category = [None]*dataset.num_rows*2
    for idx, val in enumerate(all_ground_truths):
        if val == 1:
            hallucination_difficulty[idx] = dataset['Difficulty Level'][idx//2]
            hallucination_category[idx] = dataset['Category of Hallucination'][idx//2]

    if info_type == "difficulty":
        return hallucination_difficulty
    elif info_type == "category":
        return hallucination_category

def process_results(dataset, predictions) -> dict: # TODO: rename fcn to something less ambiguous（＾ｖ＾）
    """
    # --- Assemble final results ---
    # example
    # all_ground_truths:              [    1,     0,     1,     0]
    # raw_gt_pred_pairs:             [(1,0), (0,-1), (1,1), (0,0)]
    # raw_hallucination_difficulty: ["hard",   None,"easy",  None]
    # raw_hallucination_category:  [   "A",    None,   "B",  None]
    
    # valid_idxs: [0,2,3] (explicit) ; [1, 0, 1, 1] (boolean mask) ; we'll go with the explicit version
    # gt_pred_pairs:         [(1,0),  (1,1),   (0,0)]
    # valid_hal_difficulty: ["hard", "easy",   None]
    # valid_hal_category:  [    "A",   "B",   None]
    """
    
    all_ground_truths = [1,0] * dataset.num_rows # in prepare_prompts(), we alternate between adding prompts with hallucination and gt answers
    raw_gt_pred_pairs = t.tensor(list(zip(all_ground_truths, predictions))) # tensor allows for easy selection from a list of desired idxs
    raw_hallucination_difficulty = get_hallucination_info("difficulty", dataset) # can't convert to tensor b/c has strings
    raw_hallucination_category = get_hallucination_info("category", dataset)
    
    valid_idxs = [idx for idx, (gt,pred) in enumerate(raw_gt_pred_pairs) if pred != -1] # explicit [0, 2, 3, 4, 7, 110]
    gt_pred_pairs = raw_gt_pred_pairs[valid_idxs] #filter_invalid_pairs(raw_gt_pred_pairs)
    valid_hal_difficulty = [raw_hallucination_difficulty[valid_idx] for valid_idx in valid_idxs]
    valid_hal_category = [raw_hallucination_category[valid_idx] for valid_idx in valid_idxs]
    valid_df = pd.DataFrame({
        'gt': [gt for (gt,pred) in gt_pred_pairs],
        'predictions': [pred for (gt,pred) in gt_pred_pairs],
        'difficulty': valid_hal_difficulty,
        'category': valid_hal_category
    })

    processed_results = {
        'all_ground_truths': all_ground_truths, 
        'raw_gt_pred_pairs': raw_gt_pred_pairs,
        'raw_hallucination_difficulty': raw_hallucination_difficulty,
        'valid_idxs': valid_idxs,
        'gt_pred_pairs': gt_pred_pairs,
        'valid_hal_difficulty': valid_hal_difficulty,
        'valid_hal_category': valid_hal_category,
        'valid_df': valid_df
    }
    
    return processed_results

def compute_confusion_matrix_vals(gt_pred_pairs: list[tuple[int,int]]):
    """compute and return TP, FP, TN, FN from a list of (ground truth, prediction) pairs"""
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # used for analysis later
    TP_idxs = []
    FP_idxs = []
    TN_idxs = []
    FN_idxs = []
    
    for idx, (gt,pred) in enumerate(gt_pred_pairs):
        if pred == 1 and gt == 1:
            TP += 1
            TP_idxs.append(idx)
        elif pred == 1 and gt == 0:
            FP += 1
            FP_idxs.append(idx)
        elif pred == 0 and gt == 0:
            TN += 1
            TN_idxs.append(idx)
        elif pred == 0 and gt == 1:
            FN += 1
            FN_idxs.append(idx)
    return TP, FP, TN, FN, TP_idxs, FP_idxs, TN_idxs, FN_idxs

def calculate_metrics(raw_gt_pred_pairs, gt_pred_pairs, valid_df):
    """
    computes and returns the following 9 metrics:
    (1) confusion matrix, (2) accuracy, (3) precision, (4) recall, (5) f1-score, (6) support, 
    (7) abstention rate,
    (8) false negative rate (fnr) and (9) true positive rate (tpr) broken down by hallucinated answer category/difficulty
    """
    cm = None
    accuracy = None
    precision = None
    recall = None
    f1 = None
    support = None
    abstention_rate = None
    fn_difficulty_counts = None
    tp_difficulty_counts = None
    fn_category_counts = None
    tp_category_counts = None
    fnr_difficulty = None
    fnr_category = None
    tpr_difficulty = None
    tpr_category = None
    
    if len(gt_pred_pairs) > 0:
        TP, FP, TN, FN, TP_idxs, FP_idxs, TN_idxs, FN_idxs = compute_confusion_matrix_vals(gt_pred_pairs)
        cm = np.array([[TN, FP], [FN, TP]])
        #cm = confusion_matrix(valid_gts, valid_preds, labels=[0,1])
    
        # what proportion of answers did the LLM NOT classify?
        total_valid_preds = len(gt_pred_pairs)
        invalid_count = len(raw_gt_pred_pairs) - total_valid_preds
        abstention_rate = invalid_count / len(raw_gt_pred_pairs)
    
        # what does tpr, fpr look like for different hallucination categories, difficulties?
        # get counts
        fn_difficulty_counts = valid_df.iloc[FN_idxs]['difficulty'].value_counts() # measure of error; type II error (minimize!!)
        tp_difficulty_counts = valid_df.iloc[TP_idxs]['difficulty'].value_counts() # success measure; sensitivity (a.k.a recall) (maximize!!)
        fn_category_counts = valid_df.iloc[FN_idxs]['category'].value_counts()
        tp_category_counts = valid_df.iloc[TP_idxs]['category'].value_counts()
    
        # plot fnr
        fnr_difficulty = fn_difficulty_counts / (fn_difficulty_counts + tp_difficulty_counts)
        fnr_category = fn_category_counts / (fn_category_counts + tp_category_counts) # todo: modify to account to NaNs
    
        # plot tpr
        tpr_difficulty = tp_difficulty_counts / (tp_difficulty_counts + fn_difficulty_counts)
        tpr_category = tp_category_counts / (tp_category_counts.add(fn_category_counts, fill_value=0))
        
        # overall metrics: accuracy, precision, recall, f1 score
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        valid_gts = [gt for (gt,pred) in gt_pred_pairs]
        valid_preds = [pred for (gt,pred) in gt_pred_pairs]
        precision, recall, f1, support = precision_recall_fscore_support(
            valid_gts, valid_preds, average='binary', pos_label=1, zero_division=0
        )
    else:
        log.warning("No valid predictions were made, skipping metric calculations.")
        print("No valid predictions were made, skipping metric calculations.")
        
    metrics_dict = {
        'cm': cm,
        'accuracy': accuracy,
        'recall': recall,
        'f1-score': f1,
        'support': support,
        'abstention_rate': abstention_rate,
        'fn_difficulty_counts': fn_difficulty_counts,
        'fn_category_counts': fn_category_counts,
        'tp_difficulty_counts': tp_difficulty_counts,
        'tp_category_counts': tp_category_counts,
        'fnr_difficulty': fnr_difficulty,
        'fnr_category': fnr_category,
        'tpr_difficulty': tpr_difficulty,
        'tpr_category': tpr_category, 
    }
    return metrics_dict

def write_results_to_json(EXPERIMENT_NAME, results: dict = None):
    assert results != None
    out_file = f"results_{EXPERIMENT_NAME}.json"
    print(f"Saving results to {out_file}.")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    print("Save complete.")
    
def write_results_to_txt(results: dict = None):
    assert results != None
    pprint.pp((results_df))
    with open('results.txt', 'w') as results_file:
        results_file.write(str(results_df))

# master fcn
@hydra.main(version_base=None, config_path="conf/", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    log.info(f"Hydra output directory: {os.getcwd()}")
    log.info(f"Configuration we're using:\n {OmegaConf.to_yaml(cfg)}")
    log.info("Starting experiment run...")
    
    # --- Configuration --- # get values from hydra cfg object    
    MODEL_NAME = cfg.model.name
    DATASET_NAME = cfg.dataset.name
    DATASET_SPLIT = cfg.dataset.split
    QUANTIZATION_METHOD = cfg.quantization.method
    QUANTIZATION_LEVEL = cfg.quantization.level
    
    EXPERIMENT_NAME = f'{MODEL_NAME}_{QUANTIZATION_METHOD}_{QUANTIZATION_LEVEL}_{DATASET_SPLIT}'

    log.info(f"EXPERIMENT NAME: {EXPERIMENT_NAME}")
    print(f"EXPERIMENT NAME: {EXPERIMENT_NAME}")

    # --- Load Dataset, Model, Tokenizer ---
    dataset = DatasetLoader().build_dataset(); assert dataset != None
    clean_gpu()

   # model_family: str, quantization_method: str, quantization_level: Optional[str]=None, device:str="auto"
    
    model, tokenizer = ModelLoader.build_model_and_tokenizer(
        model_family=MODEL_NAME,
        quantization_method=QUANTIZATION_METHOD,
        quantization_level=QUANTIZATION_LEVEL,
    )
    assert model != None and tokenizer != None
    
    # --- Construct HF Pipeline ---
    classifier = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    # --- Create Prompts ---
    all_prompts = create_prompts(dataset)

    # --- Clear memory stats *before* inference, for acc. logging of GPU usage *during* inference ---
    gc.collect()
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats()

    # --- Inference ---
    outputs = classify_med_answers(all_prompts, BATCH_SIZE)

    # --- Log Peak GPU Memory Usage --- note: re-factor into function?
    peak_memory_bytes = t.cuda.max_memory_allocated()
    peak_memory_gb = peak_memory_bytes / (2**30) # 2^30B in one GB
    log.info(f"Peak GPU Memory Allocated: {peak_memory_gb} GB")
    print(f"Peak GPU Memory Allocated: {peak_memory_gb} GB")

    predictions, raw_outputs = extract_all_binary_predictions(outputs)

    # --- Compute Metrics (fnr, tpr, acc, prec, rec, f1, abstention rate) ---
    processed_results = process_results(dataset, predictions)

    metrics_dict = calculate_metrics(processed_results['raw_gt_pred_pairs'],
                                     processed_results['gt_pred_pairs'],
                                     processed_results['valid_df'])
    log.info(metrics_dict)
    
    # --- Create Plots ---
    rename_hallucination_categories = {
        'Incomplete Information': 'Inc. Inf.',
        'Mechanism and Pathway Misattribution': 'M&PM',
        'Misinterpretation of #Question#': 'MisinterpretQ',
        'Methodological and Evidence Fabrication': 'M&E Fab.'
    }
    
    ## false negative rate (fnr) (misses)
    ### plot1
    metrics_dict['fnr_difficulty'].plot(title="False Negative (miss) rate, organized by hallucination difficulty", kind='bar')
    ### plot2
    metrics_dict['fnr_category'].plot(title="False Negative (miss) rate organized by category",kind='bar')
    ### combined plot
    plt.bar(metrics_dict['fnr_difficulty'].index, metrics_dict['fnr_difficulty'])
    plt.bar([rename_hallucination_categories[x] for x in metrics_dict['fnr_category'].index], metrics_dict['fnr_category'])
    plt.xticks(rotation=30)
    plt.legend(['difficulty', 'category'])
    plt.title("False Negative (miss) rate of hallucinated answers organized by difficulty, category")
    plt.ylabel('False Negative Rate')

    ## true positive rate (tpr) (successes)
    ### plot 3
    plt.bar(metrics_dict['tpr_difficulty'].index, metrics_dict['tpr_difficulty'])
    plt.title('True Positive (hit) rate for hallucinated answers organized by difficulty')
    ### plot 4
    plt.bar([rename_hallucination_cat[x] for x in metrics_dict['tpr_category'].index], metrics_dict['tpr_category'])
    plt.title('True Positive (hit) rate for hallucinated answer classification, organized by category')
    plt.ylabel('true positive rate')
    ### combined tpr plot
    plt.bar(metrics_dict['tpr_difficulty'].index, metrics_dict['tpr_difficulty'])
    plt.bar([rename_hallucination_cat[x] for x in metrics_dict['tpr_category'].index], metrics_dict['tpr_category'])
    plt.title('True Positive (hit) rate for hallucation answer classification, organized by difficulty, category')
    plt.legend(['difficulty', 'category'])
    plt.ylabel('true positive rate')
    plt.xticks(rotation=30)

    # --- Save Results ---
    #EXPERIMENT_NAME = EXPERIMENT_NAME.split('/')[-1]

    results = {
        'experiment_name': EXPERIMENT_NAME.split('/')[-1],
        'semi-processed_results': processed_results,
        'metrics': {
            'gpu_peak_memory_gb': peak_memory_gb,
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
    write_results_to_json(EXPERIMENT_NAME, results)

if __name__ == "__main__":
    run_experiment()