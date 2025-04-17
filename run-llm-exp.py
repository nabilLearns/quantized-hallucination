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
import ollama # another runtime for inference with quantized gguf models ; this seems to be the best candidate based on experience
import yaml # for cfgs
import logging
from abc import ABC, abstractmethod
import json # final results for a run will be written to a json file
from time import time

# script specific (not used in the jupyter notebook version)
from omegaconf import DictConfig, OmegaConf
import hydra
import subprocess
from huggingface_hub import hf_hub_download
from multiprocessing import Pool
import pickle


# --- Setup logging ---
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING) # so we don't get spammed with [httpx] logs during ollama inference

# --- Helper Classes/Functions ---
class AbstractModelLoader(ABC):
    @abstractmethod
    def _get_tokenizer_kwargs(self):
        pass
    @abstractmethod
    def _build_tokenizer(self):
        pass
    @abstractmethod
    def _get_model_kwargs(self):
        pass
    @abstractmethod
    def _build_hf_model(self):
        pass
    @abstractmethod
    def _download_gguf_to_local_machine(self):
        pass
    @abstractmethod
    def _build_gguf_model(self):
        pass
    @abstractmethod
    def build_model_and_tokenizer(self):
        pass

class ModelLoader(AbstractModelLoader):
    def __init__(self,
                 model_family: str,
                 quantization_method: str,
                 quantization_level: Optional[str]=None,
                 base_hf_id: Optional[str]=None,
                 gguf_hf_id: Optional[str]=None,
                 gguf_fpath: Optional[str]=None,
                 model_specific_tokenizer_kwargs: Optional[Dict[str, str|bool]]=None,
                 device:str="auto"):

        self.model_family = model_family
        self.quantization_method = quantization_method
        self.quantization_level = quantization_level
        self.device = device

        # get these from the config file
        self.base_hf_id = base_hf_id
        self.gguf_hf_id = gguf_hf_id
        self.gguf_fpath = gguf_fpath # selected by hydra at cli
        self.model_specific_tokenizer_kwargs = model_specific_tokenizer_kwargs
        #self.tokenizer = _build_tokenizer()
        #self.model = _build_model()
        #self.model_kwargs = model_kwargs
    
    def _get_tokenizer_kwargs(self):
        """Get kwargs from config"""
        # default
        tokenizer_kwargs = {
            'trust_remote_code': True,
            **self.model_specific_tokenizer_kwargs
        }
        #log.info(tokenizer_kwargs)
        if 'gguf' in self.quantization_method:
            try:
                tokenizer_kwargs['gguf_file'] = self.gguf_fpath #self.config['gguf_files'][f'{self.quantization_level}']
            except:
                log.error(f"Tokenizer Kwargs: Was not able to extract {self.model_family} {self.quantization_level} gguf file from config.")
        # should this just be a self variable?
        return tokenizer_kwargs

    def _get_model_kwargs(self):
        # default
        model_kwargs = {
            'quantization_config': None,
            'device_map': "auto",
            'attn_implementation': "eager", # can use 'flash-attention-2' if user nvidia ampere or newer
            'torch_dtype': t.float16, # depends on gpu being used ; 
            'trust_remote_code': True,
            #**self.model_kwargs
        }
        if 'gguf' in self.quantization_method:
            try:
                model_kwargs['gguf_file'] = self.gguf_fpath #self.config['gguf_files'][f'{self.quantization_level}']
            except:
                log.error(f"Model Kwargs: Was not able to extract {self.model_family} {self.quantization_level} gguf file from config.")        
        elif 'uncompressed' in self.quantization_method:
            log.info("Loading uncompressed model.")
        # should this just be a self variable?
        return model_kwargs
            
    def _build_tokenizer(self):
        tokenizer_path = self.base_hf_id #self.config['base_hf_id']; 
        log.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer_kwargs = self._get_tokenizer_kwargs()
        log.info(tokenizer_kwargs)

        # Runtime == HF
        if self.quantization_method == 'gguf':
            try:
                log.info("Attempting to build gguf tokenizer.")
                tokenizer = AutoTokenizer.from_pretrained(self.gguf_hf_id, **tokenizer_kwargs)
                log.info("Done building gguf tokenizer.")
                return tokenizer
            except:
                log.info("Failed to build gguf tokenizer. Using tokenizer for base model instead.")
                tokenizer_kwargs.pop('gguf_file')
                #pass
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)

        log.info("TOKENIZER BUILT!")
        log.info(type(tokenizer))
        #self.tokenizer = tokenizer
        return tokenizer
        
    def _build_hf_model(self):
        model_kwargs = self._get_model_kwargs()
        model = AutoModelForCausalLM.from_pretrained(self.base_hf_id, **model_kwargs) #(self.config['base_hf_id'], **model_kwargs)
        log.info(f"Successfully loaded {self.base_hf_id} model.")
        log.info(type(model))
        #self.model = model
        return model

    def _download_gguf_to_local_machine(self) -> str:
        try:
            log.info(f"DOWNLOADING {self.gguf_hf_id} MODEL(S) FROM HUGGING FACE")
            hf_hub_download(repo_id=f'{self.gguf_hf_id}',filename=f'{self.gguf_fpath}')
            return 'Success!'
        except Exception as e:
            log.error("Was not able to download gguf model / gguf files from hugging face.")
            log.info(e)
            return 'Failed.'

    def _build_gguf_model(self):
        model_kwargs = self._get_model_kwargs()
        model = AutoModelForCausalLM.from_pretrained(self.gguf_hf_id, **model_kwargs)
        log.info(f"Successfully loaded {self.gguf_hf_id} model.")
        return model
        
    def build_model_and_tokenizer(self):
        model, tokenizer = None, None
        print(f"Loading model: {self.model_family}]") # \nHyperparams: {model_kwargs}")
        if 'gguf' in self.quantization_method:
            status = self._download_gguf_to_local_machine()
            log.info(f"GGUF DOWNLOAD STATUS: {status}")
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

class DatasetLoader:
    def __init__(self, MAX_SAMPLES=1000, DATASET_NAME='UTAustin-AIHealth/MedHallu', DATASET_SPLIT='pqa_labeled'):
        self.max_samples = MAX_SAMPLES
        self.dataset_name = DATASET_NAME
        self.dataset_split = DATASET_SPLIT
        
    def build_dataset(self):
        log.info(f'Loading {self.dataset_name} dataset ({self.dataset_split})...')
        ds = load_dataset(self.dataset_name, self.dataset_split)
        dataset = ds['train'] # use train split which has 1k labelled samples
        if self.max_samples is not None:
            log.info(f'Limiting dataset to {self.max_samples} samples for testing.')
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
    
    log.info("Preparing prompts")
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
    log.info("Prompts are prepared.")
    return all_prompts


def ollama_classify(prompts, model, experiment_name, BATCH_SIZE, MAX_NEW_TOKENS) -> tuple[list,list]:
    log.info(f"Starting Ollama Hallucination Detection for {model}.")
    outputs = []
    latencies = []
    for i, prompt in enumerate(prompts):
        start = time()
        response = ollama.chat(
            model=model,
            messages=prompt
        )
        duration = time() - start
        if i % 10 == 0:
            log.info(f"Inference for {i}th prompt took {duration:.2f}s | Response: {response['message']['content'].strip()}")
        latencies.append(duration)
        outputs.append(response['message']['content'].strip())
    log.info("Inference Done!")
    return outputs, latencies

# this is for hugging_face runtime ; deprecated for now
# TODO: measure inference, throughput, TTS?
def classify_med_answers(prompts, classifier, tokenizer, experiment_name: str, BATCH_SIZE=8, MAX_NEW_TOKENS=3) -> list:
    log.info(f"Starting batch inference on {len(prompts)} prompts...")
    log.info(type(prompts))
    outputs = []
    num_batches=math.ceil(len(prompts) / BATCH_SIZE)
    with t.no_grad():
        # show inference progress bar with tqdm
        for batch_idx in tqdm(range(num_batches), desc=f"Classifying Batches: {experiment_name}", unit="batch"):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]
            batch_output = classifier(batch_prompts,
                                      #batch_size=BATCH_SIZE # redundant
                                      max_new_tokens=MAX_NEW_TOKENS,
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
    return outputs

# fcn group 2 -> combine into a class?
def extract_prediction(generated_text):
    """Extract the '0' or '1' from generated text, in case model does not listen to instructions and adds other tokens"""
    #log.info(generated_text)
    #log.info(f"{[type(x) for x in generated_text]}")
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
        
def extract_all_binary_predictions(outputs, runtime='ollama') -> Tuple[List[int], list]:
    """
    for output in outputs:
        1. filter out original prompt, # b/c model output *includes the input prompt* as well as new generated text
        2. extract binary prediction, and
        3. append to lists for predictions and raw model outputs
    """
    
    predictions = []
    raw_outputs = []
    log.info("Extracting Predictions From Model Responses.")

    if runtime=='ollama':
        for i, model_response in enumerate(outputs):
            raw_outputs.append(model_response)
            pred = extract_prediction(model_response)
            predictions.append(pred)
    else:
        # runtime = HF ; qwen2.5 7B output
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

    
    log.info("Predictions Extracted.")
    return predictions, raw_outputs

# fcn group 3 -> combine into a class?
def get_hallucination_info(info_type: str, dataset, all_ground_truths: list[int]): # how do I type hint a hugging face (hf) dataset?
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

# TODO: rename fcn to something less ambiguous（＾ｖ＾）
def process_results(dataset, predictions: list[int]) -> dict:
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
    raw_hallucination_difficulty = get_hallucination_info("difficulty", dataset, all_ground_truths) # can't convert to tensor b/c has strings
    raw_hallucination_category = get_hallucination_info("category", dataset, all_ground_truths)
    
    valid_idxs = [idx for idx, (gt,pred) in enumerate(raw_gt_pred_pairs) if pred != -1] # explicit [0, 2, 3, 4, 7, 110]
    gt_pred_pairs = raw_gt_pred_pairs[valid_idxs] #filter_invalid_pairs(raw_gt_pred_pairs)
    valid_hal_difficulty = [raw_hallucination_difficulty[valid_idx] for valid_idx in valid_idxs]
    valid_hal_category = [raw_hallucination_category[valid_idx] for valid_idx in valid_idxs]

    # include the ground truth a
    
    valid_df = pd.DataFrame({
        'gt': [gt for (gt,pred) in gt_pred_pairs],
        'predictions': [pred for (gt,pred) in gt_pred_pairs],
        'difficulty': valid_hal_difficulty,
        'category': valid_hal_category
    })

    processed_results = {
        'all_ground_truths': all_ground_truths, 
        'raw_gt_pred_pairs': raw_gt_pred_pairs, # tensor
        'raw_hallucination_difficulty': raw_hallucination_difficulty,
        'valid_idxs': valid_idxs,
        'gt_pred_pairs': gt_pred_pairs, # tensor
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

def calculate_metrics(raw_gt_pred_pairs: list[tuple[int, int]], gt_pred_pairs: list[tuple[int, int]], valid_df):
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

        # ensure all difficulties are represented in the difficulty dicts, categories in the category dicts
        # prevent NaNs from showing up in the calculations for fnr or tpr below
        difficulty_keys = ['hard', 'medium', 'easy']
        category_keys = ['Incomplete Information',
                         'Mechanism and Pathway Misattribution',
                         'Misinterpretation of #Question#',
                         'Methodological and Evidence Fabrication']
        # ensures that all categories are represnted, but still does not prevent NaNs that could be produced by arithmetic done below
        # using df.fillna() to address that
        for k in difficulty_keys:
            if k not in fn_difficulty_counts.keys():
                fn_difficulty_counts[k] = 0
            elif k not in tp_difficulty_counts.keys():
                tp_difficulty_counts[k] = 0
                
        for k in category_keys:
            if k not in fn_category_counts.keys():
                fn_category_counts[k] = 0
            elif k not in tp_category_counts.keys():
                tp_category_counts[k] = 0
    
        # plot fnr
        fnr_difficulty = (fn_difficulty_counts / (fn_difficulty_counts + tp_difficulty_counts)).fillna(0) # nan vals can't be serialized to JSON..
        fnr_category = (fn_category_counts / (fn_category_counts + tp_category_counts)).fillna(0) # so we replace with 0 here, for later
    
        # plot tpr
        tpr_difficulty = (tp_difficulty_counts / (tp_difficulty_counts + fn_difficulty_counts)).fillna(0)
        tpr_category = (tp_category_counts / (tp_category_counts.add(fn_category_counts, fill_value=0))).fillna(0)

        #log.info(type(fnr_difficulty))
        #log.info(fnr_difficulty)
        
        # overall metrics: accuracy, precision, recall, f1 score
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        valid_gts = [gt for (gt,pred) in gt_pred_pairs]
        valid_preds = [pred for (gt,pred) in gt_pred_pairs]
        precision, recall, f1, support = precision_recall_fscore_support(
            valid_gts, valid_preds, average='binary', pos_label=1, zero_division=0
        )
    else:
        log.warning("No valid predictions were made, skipping metric calculations.")
        
    metrics_dict = {
        'cm': cm,
        'accuracy': accuracy,
        'precision': precision,
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

def create_and_save_plots(experiment_name: str, metrics_dict: dict, output_dir:str=None) -> None:
    """
    Creates Plots for LLM Hallucination Detection FNR/TPR broken down by difficulty/category
    Plots are saved, as per the desired directory structure indicated below:

    medical-llm-hallucination-thesis/
      ...
      plots/
        qwen_uncompressed_pqa_labeled/
          confusion_matrix_qwen_uncompressed_pqa_labeled.png
          perf_by_category_qwen_uncompressed_pqa_labeled.png
          perf_by_difficulty_qwen_uncompressed_pqa_labeled.png
    """
    # --- Create output folder if it does not exist ---
    if output_dir == None:
        output_dir = os.path.join(os.getcwd(), 'plots', f'{experiment_name}')
    os.makedirs((output_dir),exist_ok=True)
    
    # --- Plot 1: FNR/TPR by Diffictuly ---
    try:
        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5)) # 1 row, by 2 cols, width=12,height=5
        fig1.suptitle('Performance by Hallucination Difficulty')
    
        # FNR Difficulty
        # display fn_difficulty_counts on easy, med, hard bars
        fnr_diff = metrics_dict['fnr_difficulty'].sort_index()
        fn_dcounts = metrics_dict['fn_difficulty_counts'].sort_index()
        fnr_bars = axes1[0].bar(fnr_diff.index, fnr_diff.values, color='salmon')
        axes1[0].bar_label(fnr_bars, labels=[f'n={c}' for c in fn_dcounts], fontsize=10, padding=3)
        #axes1[0].bar_label(fn_difficulty_counts)
        axes1[0].set_title('False Negative Rate (Miss Rate)', pad=15)
        axes1[0].set_ylabel('Rate')
        axes1[0].set_xlabel('Difficulty')
        axes1[0].set_ylim(0, 1) # Standard scale for rates
    
        # TPR Difficulty
        # display tp_difficulty_counts on easy, med, hard bars
        tpr_diff = metrics_dict['tpr_difficulty'].sort_index()
        tp_dcounts = metrics_dict['tp_difficulty_counts'].sort_index()
        tpr_bars = axes1[1].bar(tpr_diff.index, tpr_diff.values, color='lightblue')
        axes1[1].bar_label(tpr_bars, labels=[f'n={c}' for c in tp_dcounts], fontsize=10, padding=3)
        axes1[1].set_title('True Positive Rate (Recall)', pad=15)
        axes1[1].set_ylabel('Rate')
        axes1[1].set_xlabel('Difficulty')
        axes1[1].set_ylim(0, 1)
    
        plt.tight_layout()
        save_path1 = os.path.join(output_dir, f"perf_by_difficulty_{experiment_name}.png")    
        plt.savefig(save_path1)
        print(f"Saved plot: {save_path1}")
        plt.close(fig1) # Close the figure
    except Exception as e:
        print(f"Failed to create/save difficulty plot: {e}")

    # Used to avoid overcrowding of xlabels in plots
    rename_hallucination_categories = {
        'Incomplete Information': 'Inc. Inf.',
        'Mechanism and Pathway Misattribution': 'M&PM',
        'Misinterpretation of #Question#': 'MisinterpretQ',
        'Methodological and Evidence Fabrication': 'M&E Fab.'
    }
    
    # --- Plot 2: FNR/TPR by Category ---
    try:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5)) # Wider for category labels
        fig2.suptitle('Performance by Hallucination Category')
    
        # FNR Category
        # display fn_category_counts on each bar
        fnr_cat = metrics_dict['fnr_category'].sort_index()
        fn_ccounts = metrics_dict['fn_category_counts'].sort_index()
        fnr_bars = axes2[0].bar([rename_hallucination_categories[x] for x in fnr_cat.index], fnr_cat.values, color='salmon')
        axes2[0].bar_label(fnr_bars, labels=[f'n={c}' for c in fn_ccounts], fontsize=10, padding=3)
        axes2[0].set_title('False Negative Rate (Miss Rate)', pad=15) # in case fnr=1 for one of the categories; prevent overlap
        axes2[0].set_ylabel('Rate')
        axes2[0].set_xlabel('Category')
        axes2[0].tick_params(axis='x', rotation=30)
        axes2[0].set_ylim(0, 1)
    
        # TPR Category
        # display tp_category_counts on each bar
        tpr_cat = metrics_dict['tpr_category'].sort_index()
        tp_ccounts = metrics_dict['tp_category_counts'].sort_index()
        tpr_bars = axes2[1].bar([rename_hallucination_categories[x] for x in tpr_cat.index], tpr_cat.values, color='lightblue')
        axes2[1].bar_label(tpr_bars, labels=[f'n={c}' for c in tp_ccounts], fontsize=10, padding=3)
        axes2[1].set_title('True Positive Rate (Recall)', pad=15)
        axes2[1].set_ylabel('Rate')
        axes2[1].set_xlabel('Category')
        axes2[1].tick_params(axis='x', rotation=30)
        axes2[1].set_ylim(0, 1)
    
        plt.tight_layout()
        save_path2 = os.path.join(output_dir, f"perf_by_category_{experiment_name}.png")        
        plt.savefig(save_path2)
        print(f"Saved plot: {save_path2}")
        plt.close(fig2)
    except Exception as e:
        print(f"Failed to create/save category plot: {e}")

    
    # --- Plot 3: Confusion Matrix ---
    try:
        cm = metrics_dict.get('cm')
        if cm is not None:
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]) # Assuming labels are 0 (Factual), 1 (Hallucinated)
            disp.plot(ax=ax3, cmap=plt.cm.Blues)
            ax3.set_title('Confusion Matrix')
            # Improve labels if needed
            ax3.set_xlabel("Predicted Label\n(0: Factual, 1: Hallucination)")
            ax3.set_ylabel("True Label\n(0: Factual, 1: Hallucination)")
    
            save_path3 = os.path.join(output_dir, f"confusion_matrix_{experiment_name}.png")
            plt.tight_layout()        
            plt.savefig(save_path3)
            print(f"Saved plot: {save_path3}")
            plt.close(fig3)
        else:
            print("Confusion matrix data not found, skipping plot.")
    except Exception as e:
        print(f"Failed to create/save confusion matrix plot: {e}")

def generate_latency_report(latencies: list):
    latencies = np.array(latencies)
    latency_mean = latencies.mean()
    latency_std_dev = latencies.std()
    latency_var = latencies.var()
    total_exp_inf_time_sec = latencies.sum()
    throughput_prompts_per_sec =  latencies.shape[0] / total_exp_inf_time_sec
    latency_report = {
        'latencies': latencies,
        'latency_mean': latency_mean,
        'latency_std_dev': latency_std_dev,
        'latency_var': latency_var,
        'total_exp_inference_time_sec': total_exp_inf_time_sec,
        'throughput_prompts_per_sec': throughput_prompts_per_sec
    }
    return latency_report

# TODO: add (better) exception handling
def write_results_to_json(EXPERIMENT_NAME, results: dict = None, output_dir:str = None):
    assert EXPERIMENT_NAME != None; assert results != None
    
    if output_dir == None:
        # make results folder, if it does not exist
        output_dir = os.path.join(os.getcwd(), 'results', f'{EXPERIMENT_NAME}')
    os.makedirs(output_dir, exist_ok=True)
    
    out_file = os.path.join(output_dir, f"results_{EXPERIMENT_NAME}.json")
    print(f"Saving results to {out_file}.")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4) # default: 0 ; value given to anything that can't be serialized i.e. nan

    log.info("Save complete.")
    
def write_results_to_txt(EXPERIMENT_NAME, results: dict = None, output_dir:str = None):
    """not used atm"""
    assert EXPERIMENT_NAME != None; assert results != None
    
    if output_dir == None:
        # make results folder, if it does not exist
        output_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    pprint.pp((results))
    out_file = os.path.join(output_dir, f"results_{EXPERIMENT_NAME}.txt")
    with open(out_file, 'w') as results_file:
        results_file.write(str(results))

# master fcn
@hydra.main(version_base=None, config_path="conf/", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    #log.info(f"Configuration we're using:\n {OmegaConf.to_yaml(cfg)}")
    log.info("####################################")
    log.info("Starting experiment run...")
    log.info("####################################")
    log.info(f"Hydra output directory: {os.getcwd()}")
    log.info(f"RUNTIME: {cfg.runtime.name}")
    # --- Configuration --- # cfg values are selected by hydra after launched from cli   
    DATASET_NAME = cfg.dataset.name
    # pass to DatasetLoader
    DATASET_SPLIT = cfg.dataset.split
    MAX_SAMPLES = cfg.max_samples
    
    # pass to ModelLoader ; change/modify now that we're using ollama for inference
    RUNTIME = cfg.runtime.name
    MODEL_NAME = cfg.model.name
    QUANTIZATION_METHOD = cfg.quantization.method # e.g., gguf, uncompressed
    QUANTIZATION_LEVEL = cfg.quantization.level # e.g., q8_0, q4_k_m, etc.
    
    # needed for ollama runtime
    if RUNTIME == 'ollama':
        OLLAMA_FILE = cfg.model.ollama_files[QUANTIZATION_LEVEL] # e.g., hf.co/mradermacher/Llama3-Med42-8B-GGUF:Q8_0
        log.info(f"MODEL_NAME: {MODEL_NAME}")
        log.info(f"QUANTIZATION_METHOD: {QUANTIZATION_METHOD}")
        log.info(f"QUANTIZATION_LEVEL: {QUANTIZATION_LEVEL}")
        log.info(f"OLLAMA_FILE: {OLLAMA_FILE}")
    
    # needed for HF runtime, not for ollama
    elif RUNTIME == 'hugging_face':
        base_hf_id = cfg.model.base_hf_id
        gguf_hf_id = cfg.model.gguf_hf_id
        gguf_fpath = cfg.model.gguf_files.get(QUANTIZATION_LEVEL, None)
        model_specific_tokenizer_kwargs = cfg.model.tokenizer_kwargs
    
    # pass to classify_med_answers
    BATCH_SIZE = cfg.batch_size
    MAX_NEW_TOKENS = cfg.max_new_tokens

    # if we're running an experiment for a certain gguf k-quant..
    # relevant info on the k-quant should be in the model config
    if 'gguf' in QUANTIZATION_METHOD:
        if RUNTIME == 'hugging_face':
            assert QUANTIZATION_LEVEL in cfg.model.gguf_files.keys()
        elif RUNTIME == 'ollama':
            assert QUANTIZATION_LEVEL in cfg.model.ollama_files.keys()
        
    EXPERIMENT_NAME = f'{MODEL_NAME}_{QUANTIZATION_METHOD}_{QUANTIZATION_LEVEL}_{RUNTIME}_{DATASET_SPLIT}'
    log.info(f"EXPERIMENT NAME: {EXPERIMENT_NAME}")
 
    # --- Load Dataset ---
    dataset = DatasetLoader(MAX_SAMPLES=MAX_SAMPLES, DATASET_SPLIT=DATASET_SPLIT).build_dataset(); assert dataset != None
    log.info("Dataset ready.")
    clean_gpu()
    # --- Create Prompts ---
    all_prompts = create_prompts(dataset)

    # --- Load Model, Tokenizer ---
    # Inference with HuggingFace runtime (AutoTokenizer.from_pretrained(...), AutoModelForCausalLM.from_pretrained(...))
    if RUNTIME == 'hugging_face':
        model, tokenizer = ModelLoader(model_family=MODEL_NAME, 
                                       quantization_method=QUANTIZATION_METHOD,
                                       quantization_level=QUANTIZATION_LEVEL,
                                       base_hf_id=base_hf_id,
                                       gguf_hf_id=gguf_hf_id,
                                       gguf_fpath=gguf_fpath,
                                       model_specific_tokenizer_kwargs=model_specific_tokenizer_kwargs).build_model_and_tokenizer()
        assert model != None
        assert tokenizer != None
        # --- Construct HF Pipeline ---
        classifier = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )
        log.info("SUCCESS! MODEL AND TOKENIZER LOADED")
    
    # Inference with llama-cpp-python runtime ; dropping due to bad latency (~TTFT) results when testing on jnb ; could not load weights to GPU, and CPU inference latency was ~130[s] per prompt (ノ-_-)ノ ミ ┴┴
    
    # --- Clear memory stats *before* inference, for acc. logging of GPU usage *during* inference ---
    gc.collect()
    try:
        t.cuda.empty_cache()
        t.cuda.reset_peak_memory_stats()
    except:
        pass # perhaps device = 'cpu'

    # --- Inference --- todo: add more inference related metrics like latency, throughput?
    # Hugging Face Runtime
    outputs = None
    if RUNTIME == 'hugging_face':
        outputs = classify_med_answers(prompts=all_prompts,
                                       classifier=classifier,
                                       tokenizer=tokenizer,
                                       experiment_name=EXPERIMENT_NAME,
                                       BATCH_SIZE=BATCH_SIZE,
                                       MAX_NEW_TOKENS=MAX_NEW_TOKENS)

    elif RUNTIME == 'ollama':
        log.info(f"Downloading {OLLAMA_FILE} from HF")
        ollama.pull(OLLAMA_FILE)
        log.info(f"Download Complete!")
        outputs, latencies = ollama_classify(prompts=all_prompts,
                                             model=OLLAMA_FILE,
                                             experiment_name=EXPERIMENT_NAME,
                                             BATCH_SIZE=BATCH_SIZE,
                                             MAX_NEW_TOKENS=MAX_NEW_TOKENS)
    
    # --- Log Peak GPU Memory Usage --- note: re-factor into function?
    # Hardware Performance
    peak_memory_gb = None
    try:
        peak_memory_bytes = t.cuda.max_memory_allocated() # not relevant for CPU
        peak_memory_gb = peak_memory_bytes / (2**30) # 2^30B in one GB
        log.info(f"Peak GPU Memory Allocated: {peak_memory_gb} GB")
    except:
        pass

    # --- Extract Prediction ---
    predictions, raw_outputs = extract_all_binary_predictions(outputs)

    # --- Compute Model Performance Metrics (fnr, tpr, acc, prec, rec, f1, abstention rate) ---
    log.info(" --- Computing Model Performance Metrics --- ")
    processed_results = process_results(dataset, predictions)
    log.info("Processed Results")
    log.info(pprint.pp(processed_results))
    
    metrics_dict = calculate_metrics(processed_results['raw_gt_pred_pairs'],
                                     processed_results['gt_pred_pairs'],
                                     processed_results['valid_df'])
    log.info("Metrics Dict")
    log.info(pprint.pp(metrics_dict))
    
    # --- Create Plots ---
    log.info("--- Creating Plots ---")
    create_and_save_plots(EXPERIMENT_NAME, metrics_dict)

    # --- Save Results ---
    log.info("--- Saving Final Results to JSON ---")

    latency_report = None
    if RUNTIME == 'ollama':
        latency_report = generate_latency_report(latencies)
        for k,v in latency_report.items():
            if type(v) == np.ndarray:
                latency_report[k] = latency_report[k].tolist()
        #results['metrics']['inference_latencies'] = latencies.tolist()
        #results['metrics'].update(generate_latency_report(latencies))
    
    log.info("Preparing results into a serializable format")
    #log.info({k:type(v) for k,v in processed_results.items()})
    pop_keys = []
    for k,v in processed_results.items():
        if type(v) == t.Tensor:
            processed_results[k] = processed_results[k].tolist()
        elif type(v) == pd.DataFrame:
            output_dir = os.path.join(os.getcwd(), 'results', f'{EXPERIMENT_NAME}')
            os.makedirs(output_dir, exist_ok=True)
            output_pth = os.path.join(output_dir, f'{k}_{EXPERIMENT_NAME}.csv')
            processed_results[k].to_csv(output_pth)
            #pickle.dump(processed_results[k], open(output_pth, 'wb'))
            #processed_results[k].to_pickle(f'{k}_{EXPERIMENT_NAME}.pkl # tried pickle ; but resulting file could not be opened
            pop_keys.append(k)
        elif type(v) == np.ndarray:
            processed_results[k] = processed_results[k].tolist()
    keys_popped = [processed_results.pop(k) for k in pop_keys]
    #log.info({k:type(v) for k,v in processed_results.items()})
    
    results = {
        'experiment_name': EXPERIMENT_NAME,
        'semi-processed_results': processed_results,
        'metrics': {
            'gpu_peak_memory_gb': peak_memory_gb,
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
    
    #log.info(pprint.pp(results))
    
    write_results_to_json(EXPERIMENT_NAME, results)
    log.info("Results Saved! Experiment is Complete.")

if __name__ == "__main__":
    run_experiment()