import random
from typing import Literal, Dict
from functools import partial
import re

from frozendict import frozendict
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizer
from absl import flags, logging, app

import experiment.flags
from experiment.data_staging_utils import serialize_prompt_to_file, get_prompts, PromptCategory

logging.set_verbosity(logging.INFO)
flags = flags.FLAGS

def generate_dataset(data_path: str=None, split: str="train"):
    with open(f"{data_path}/{split}/{split}.txt", encoding="utf-8") as file:
        for index, line in enumerate(file):
            components = line.strip().split("\t\t")
            if len(components) < 3:
                logging.info(f"{index}: {line}")
                continue
            logging.info(f"{index}: {components[2].split()}")
            yield {"inputs": components[0].strip(), "target": components[1].strip(), "meta": components[2].strip()}

def filter_length(example, encoder_seq_length: int, decoder_seq_length: int, tokenizer: PreTrainedTokenizer, get_statistic: bool=True):
    input_token_length = len(tokenizer(example["inputs"]).input_ids)
    output_token_length = len(tokenizer(example["target"]).input_ids)
    if get_statistic:
        global output_length
        global input_length
        output_length.append(input_token_length)
        input_length.append(output_token_length)
    return (input_token_length <= encoder_seq_length and output_token_length <= decoder_seq_length)

def add_prompt(example, prompts: frozendict, tokenizer: PreTrainedTokenizer, prompt_offset: int=3,):
    prompts_keys = list(prompts.keys())

    token_length = len(tokenizer(example["inputs"]).input_ids)
    
    input_prompts = []
    token_ids = []
    while (
        token_length + len(token_ids) < flags.encoder_seq_length-prompt_offset
        and len(prompts_keys) > 0
    ):
        key = prompts_keys.pop(random.randrange(len(prompts_keys)))
        input_prompts += random.choices(prompts[key])
        prompt = "; ".join(input_prompts)
        token_ids = tokenizer(prompt).input_ids
    
    # logging.info(f"loading example with {token_length} example tokens and {len(token_ids)} instruction tokens")
    if input_prompts:
        example["inputs"] = f"[instructions] {'; '.join(input_prompts)}\t{example['inputs']}"
    
    return example

def break_task(examples): # task_dependencies: frozendict):
    """
        This function is to breakdown tasks and form new example structure, 
        Notice: only 1 example one time, must be used with DATASETS.MAP(batched=True,batch_size=1)
    """
    input_components = examples["inputs"][0].split("\t")
    output_components = examples["target"][0].split("\t")
    
    components = {}
    examples["inputs"] = []
    examples["target"] = []
    meta = examples["meta"][0]
    examples["meta"] = []
    
    for category in PromptCategory:
        for component in output_components + input_components:
            if f"[{category}]" in component:
                components[category] = component
        assert category in components, f"[{category}]\n{output_components + input_components=}\n{meta=}"
    
    # We break original task into multiple sub tasks,
    # Concretely:
    #   - Next actions prediction based on symbolized actions
    #   - State tracking
    #   - Last user actions prediction 
    #   - Last system actions prediction
    conversation: str = components[PromptCategory.conversation]
    history: str = components[PromptCategory.history]

    conversation_turns = re.findall(r'((?:\[user\]|\[system\]).*?)(?=$| \[user\]| \[system\])', conversation)
    history_turns = re.findall(r'((?:\w+, )*\w+)(?:;|\Z)', history)
    assert len(conversation_turns) == len(history_turns), f"{conversation_turns=}\n{history_turns=}\n{history=}"
    # 1. State tracking
    _question = f"based on {PromptCategory.conversation} and {PromptCategory.params}, give the current states of the conversation"
    _input = [components[PromptCategory.conversation], components[PromptCategory.params]]
    random.shuffle(_input)
    examples["inputs"].append('\t'.join([_question] + _input))
    examples["target"].append(components[PromptCategory.states])
    examples["meta"].append(meta + "\ttask:state_tracking")
    
    # 2. Last user actions prediction
    _question = f"based on {PromptCategory.useracts}, {PromptCategory.params} and utterances, give the current user actions"
    _input = [conversation_turns[-1], components[PromptCategory.params], components[PromptCategory.useracts]]
    examples["inputs"].append('\t'.join([_question] + _input))
    examples["target"].append(history_turns[-1])
    examples["meta"].append(meta + "\ttask:user_actions_tracking")
    
    # 3. Last system actions prediction
    if len(conversation_turns) > 2:
        _question = f"based on {PromptCategory.sysacts}, {PromptCategory.params} and utterances, give the current system actions"
        _input = [conversation_turns[-2], components[PromptCategory.params], components[PromptCategory.sysacts]]
        examples["inputs"].append('\t'.join([_question] + _input))
        examples["target"].append(history_turns[-2])
        examples["meta"].append(meta + "\ttask:system_actions_tracking")
    
    # 4. Next actions prediction
    _question = f"based on {PromptCategory.params}, {PromptCategory.useracts}, {PromptCategory.sysacts}, {PromptCategory.dependencies}, {PromptCategory.targetacts} and {PromptCategory.history}, give the current user actions"
    _input = [components[PromptCategory.params], components[PromptCategory.useracts], 
              components[PromptCategory.sysacts], components[PromptCategory.dependencies], 
              components[PromptCategory.targetacts], components[PromptCategory.history]]
    examples["inputs"].append('\t'.join([_question] + _input))
    examples["target"].append(components[PromptCategory.nextacts])
    examples["meta"].append(meta + "\ttask:next_actions_prediction")
    
    return examples

DATA_STAGE = Literal["raw","add_prompt","break_task",]
def get_dataset(split: str=None, 
                data_path: str=None, 
                shuffle: bool=True, 
                dataset_stage: [DATA_STAGE]=["break_task", "add_prompt"], 
                to_tokens: bool=True, return_indermediate=True):
    cache = flags.num_workers == 1
    if return_indermediate:
        dict_intermediate_ds = {}
        
    ds = Dataset.from_generator(
        generate_dataset,
        gen_kwargs={"data_path": data_path, "split": split},
        cache_dir="./cache"
    )
    
    if return_indermediate:
        dict_intermediate_ds["initial"] = ds
    
    
    if "break_task" in dataset_stage:
        if cache:
            ds = ds.map(break_task, num_proc=1, batched=True, batch_size=1, cache_file_name="./cache/break_task")
        else:
            ds = ds.map(break_task, num_proc=flags.num_workers, batched=True, batch_size=1)
        if return_indermediate:
            dict_intermediate_ds["break_task"] = ds
     
    length_before = len(ds)
    logging.info(f"before filter: dataset length = {length_before}")
    
    tokenizer = AutoTokenizer.from_pretrained(flags.model_path)
    tokenizer.model_max_length = flags.encoder_seq_length
    
    # Call tokenizer before any mapping to make it hashable 
    # https://github.com/huggingface/datasets/issues/3638
    tokenizer("Some", "test", truncation=True) 
    
    parameters = {"tokenizer": tokenizer, "encoder_seq_length": flags.encoder_seq_length, "decoder_seq_length": flags.decoder_seq_length, "get_statistic": flags.get_statistic}
    if cache:
        ds_filtered = ds.filter(partial(filter_length, **parameters), num_proc=1, cache_file_name="./cache/filter")
    else:
        ds_filtered = ds.filter(partial(filter_length, **parameters), num_proc=flags.num_workers)
    logging.info(f"{ds}")
    length_after = len(ds_filtered)
    logging.info(f"after filter: dataset length = {length_after}")
    logging.info(f"keep proportion: {length_after / length_before * 100}%")

    if "add_prompt" in dataset_stage:
        serialize_prompt_to_file(data_path, flags.prompt_file)
        prompts = get_prompts(data_path, flags.prompt_file)
        prompts = frozendict(prompts)
        parameters = {"tokenizer": tokenizer, "prompts": prompts}
        if cache:
            ds_filtered = ds_filtered.map(partial(add_prompt, **parameters), num_proc=1, cache_file_name="./cache/add_prompt")
        else:
            ds_filtered = ds_filtered.map(partial(add_prompt, **parameters), num_proc=flags.num_workers)
        if return_indermediate:
            dict_intermediate_ds["add_prompt"] = ds_filtered
            
    if to_tokens:
        parameters = {"tokenizer": tokenizer}
        if cache:
            ds_filtered = ds_filtered.map(partial(preprocess_function, **parameters), num_proc=1, cache_file_name="./cache/to_tokens")
        else:
            ds_filtered = ds_filtered.map(partial(preprocess_function, **parameters), num_proc=flags.num_workers)
        if return_indermediate:
            dict_intermediate_ds["to_tokens"] = ds_filtered

    if shuffle:
        ds_filtered = ds_filtered.shuffle()

    if return_indermediate:
        return ds_filtered, dict_intermediate_ds
    else:
        return ds_filtered, None

def preprocess_function(example, tokenizer: PreTrainedTokenizer):
    model_inputs = tokenizer(example["inputs"], max_length=flags.encoder_seq_length, padding="max_length", truncation=True)
    labels = tokenizer(text_target=example["target"], max_length=flags.decoder_seq_length, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    

def get_merged_dataset(split: str=None, data_path: str=None, shuffle: bool=True, dataset_stage: DATA_STAGE=["break_task"], to_tokens: bool=True) -> Dataset:
    versions = ["v0", "v1", "v2", "v3", "v4", "v5"]
    ds_list = []
    for version in versions:
        ds = get_dataset(split,f"{data_path}{version}/",shuffle,dataset_stage,to_tokens, return_indermediate=False)
        ds_list.append(ds)
    dataset = concatenate_datasets(ds_list)
    return dataset
    
def main(_):
    if flags.get_statistic:
        from matplotlib import pyplot as plt
        import pickle
        import os
        
        if (not os.path.exists("./input_length.pickle") or not os.path.exists("./output_length.pickle")):
            global input_length 
            input_length = []
            global output_length 
            output_length = []
            _ = get_dataset("train","../data/processed/v0/")
        
            with open('./input_length.pickle', 'wb') as fp:
                input_length
                pickle.dump(input_length, fp)
            
            with open('./output_length.pickle', 'wb') as fp:
                output_length
                pickle.dump(output_length, fp)
                
        with open("./input_length.pickle", "rb") as f:
            input_length = pickle.load(f)
        with open("./output_length.pickle", "rb") as f:
            output_length = pickle.load(f)
        fig, ax = plt.subplots(ncols=2,figsize =(10, 7))
        logging.info(len(output_length))
        logging.info(len([output for output in output_length if output < 300]))

        ax[0].hist(input_length,) #bins = bins)

        ax[1].hist(output_length,) #bins = bins)

        plt.savefig('./data_length.png')
    
        logging.info(max(input_length))
        logging.info(min(input_length))

        logging.info(max(output_length))
        logging.info(min(output_length))
    else:
        dataset, intermediate_values = get_dataset("train","../data/processed/v0/",dataset_stage=["task_break"],to_tokens=False)

        for i in intermediate_values["break_task"].shuffle():
            print(i)
            input()
            
if __name__=="__main__":
    app.run(main)