# Code from https://github.com/ericwtodd/function_vectors
# MIT License - Copyright (c) 2023 Eric Todd

import numpy as np
import pandas as pd
from typing import *



def create_fewshot_primer(prompt_data) -> str:
    """Creates the primer string for GPT in-context learning
    
    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information

    Returns:
    prompt: the constructed ICL prompt primer as a string
    """       
    prompt = ''
    prompt += prompt_data['prefixes']['instructions'] + prompt_data['instructions'] + prompt_data['separators']['instructions']
    
    for example in prompt_data['examples']:
        
        prompt += prompt_data['prefixes']['input'] + example['input'] + prompt_data['separators']['input']
        prompt += prompt_data['prefixes']['output'] + example['output'] + prompt_data['separators']['output']
        
    return prompt
    
def create_prompt(prompt_data, sentence=None) -> str:
    """Creates a prompt using the specified sentence for GPT in-context learning
    
    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    sentence: a query string (sentence/word) to include in the ICL prompt

    Returns:
    prompt: the constructed ICL prompt as a string
    """
    if sentence is None and prompt_data['query_target'] is not None:
        sentence = prompt_data['query_target']['input']

    if isinstance(sentence, list):
        sentence = sentence[0]

    prompt_init = create_fewshot_primer(prompt_data)    
    prompt = prompt_init + prompt_data['prefixes']['input'] + sentence + prompt_data['separators']['input']
    prompt += prompt_data['prefixes']['output']
    
    return prompt   

# Partial primer & prompt functions
def create_partial_fewshot_primer(prompt_data, include = np.arange(8)) -> str:
    """Creates the primer string for GPT in-context learning, filtering to include a subset of specified priming strings
    
    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    include: an iterable of ints indicating which examples to include in the ICL prompt
    
    Returns:
    prompt: the constructed ICL prompt primer as a string
    """
    prompt = ''
    prompt += prompt_data['prefixes']['instructions'] + prompt_data['instructions'] + prompt_data['separators']['instructions']

    # Grab each priming example in the specified order.
    for i in include:
        example = prompt_data['examples'][i]
        prompt += prompt_data['prefixes']['input'] + example['input'] + prompt_data['separators']['input']
        prompt += prompt_data['prefixes']['output'] + example['output'] + prompt_data['separators']['output']
        
    return prompt

def create_partial_prompt(prompt_data, sentence=None, include=np.arange(8)) -> str:
    """Creates a prompt using the specified sentence and partial list of in-context primer sentences
    
    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    sentence: a query string (sentence /word) to include in the ICl prompt
    include: an iterable of ints indicating which examples to include in the ICL prompt
    
    Returns:
    prompt: the prompt as a string
    """
    if sentence is None and prompt_data['query_target'] is not None:
        sentence = prompt_data['query_target']['input']
    if isinstance(sentence, list):
        sentence = sentence[0]
        
    prompt_init = create_partial_fewshot_primer(prompt_data, include)
    
    prompt = prompt_init + prompt_data['prefixes']['input'] + sentence + prompt_data['separators']['input']
    prompt += prompt_data['prefixes']['output']
    
    return prompt

def word_pairs_to_prompt_data(word_pairs : dict,
                              instructions: str = "",
                              prefixes: dict = {"input":"Q:", "output":"A:","instructions":""},
                              separators: dict = {"input":"\n", "output":"\n\n", "instructions":""},
                              query_target_pair: dict = None, prepend_bos_token=False,
                              shuffle_labels=False, prepend_space=True) -> dict:
    """Takes a dataset of word pairs, and constructs a prompt_data dict with additional information to construct an ICL prompt.
    Parameters:
    word_pairs: dict of the form {'word1':['a', 'b', ...], 'word2':['c', 'd', ...]}
    instructions: prefix instructions for an ICL prompt
    prefixes: dict of ICL prefixes that are prepended to inputs, outputs and instructions
    separators: dict of ICL separators that are appended to inputs, outputs and instructions
    query_target_pair: dict with a single input-output pair acting as the query for the prompt
    prepend_bos_token: whether or not to prepend a BOS token to the prompt
    shuffle_labels: whether to shuffle the ICL labels
    prepend_space: whether to prepend a space to every input and output token

    Returns: 
    prompt_data: dict containing ICL prompt examples, and template information
    """
    prompt_data = {}
    prompt_data['instructions'] = instructions
    prompt_data['separators'] = separators
    if prepend_bos_token:
        prefixes = {k:(v if k !='instructions' else '<|endoftext|>' + v) for (k,v) in prefixes.items()}
    prompt_data['prefixes'] = prefixes

    if query_target_pair is not None:
        query_target_pair = {k:(v[0] if isinstance(v, list) else v) for k,v in query_target_pair.items()}
    prompt_data['query_target'] = query_target_pair
        
    if shuffle_labels:
        randomized_pairs = [np.random.permutation(x).tolist() if i==1 else x for (i,x) in enumerate(list(word_pairs.values()))] # shuffle labels only
        if prepend_space:
            prompt_data['examples'] = [{'input':' ' + str(w1), 'output':' ' + str(w2)} for (w1,w2) in list(zip(*randomized_pairs))]
            prompt_data['query_target'] = {k:' ' + str(v) for k,v in query_target_pair.items()} if query_target_pair is not None else None
        else:
            prompt_data['examples'] = [{'input':w1, 'output':w2} for (w1,w2) in list(zip(*randomized_pairs))]
    else:    
        if prepend_space:
            prompt_data['examples'] = [{'input':' ' + str(w1), 'output':' ' + str(w2)} for (w1,w2) in list(zip(*word_pairs.values()))]
            prompt_data['query_target'] = {k:' ' + str(v) for k,v in query_target_pair.items()} if query_target_pair is not None else None
        else:
            prompt_data['examples'] = [{'input':w1, 'output':w2} for (w1,w2) in list(zip(*word_pairs.values()))]
    
    return prompt_data


# DATASET UTILS
class ICLDataset:
    """
    A simple dataset class containing input-output pairs, used for ICL prompt construction.
    """
    def __init__(self, dataset):    
        if isinstance(dataset, str):
            self.raw_data = pd.read_json(dataset)
        elif isinstance(dataset, dict):
            self.raw_data = pd.DataFrame(dataset)
        self.raw_data = self.raw_data[['input', 'output']]

    def __getitem__(self,i):       
        if isinstance(i, int):
            return self.raw_data.iloc[i].to_dict()
        elif isinstance(i, slice):
            return self.raw_data.iloc[i].to_dict(orient='list')
        elif isinstance(i, list) or isinstance(i, np.ndarray):            
            return self.raw_data.iloc[i].to_dict(orient='list')
        elif isinstance(i, str):
            if i not in self.raw_data.columns:
                raise KeyError(f"Column '{i}' not in the dataset. Current columns in the dataset: {self.raw_data.columns.to_list()}")
            else:
                return self.raw_data[i].to_list()
        else:
            raise ValueError(f"{i} is not a valid index type. Expected one of: [int, list, np.ndarray, slice, str]")

    def __len__(self):
        return len(self.raw_data)
    
    def __repr__(self):
        s = "ICLDataset" + "({\n\tfeatures: " + f"{self.raw_data.columns.to_list()},\n\tnum_rows: {self.__len__()}" + "\n})"
        return s
