import random
from typing import Literal, Dict
from collections import defaultdict
from functools import partial

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizer
from absl import flags, logging, app

import experiment.flags
from experiment.generate_prompt import serialize_prompt_to_file, get_prompts

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

def add_prompt(example, prompts: Dict, tokenizer: PreTrainedTokenizer, prompt_offset: int=3,):
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
        example["inputs"] = f"[instructions] {'; '.join(input_prompts)} {example['inputs']}"
    
    return example

DATA_STAGE = Literal["raw","add_prompt"]
def get_dataset(split: str=None, data_path: str=None, shuffle: bool=True, dataset_stage: DATA_STAGE="add_prompt"):

    ds = Dataset.from_generator(
        generate_dataset,
        gen_kwargs={"data_path": data_path, "split": split},
        cache_dir="./"
    )
    
    length_before = len(ds)
    logging.info(f"before filter: dataset length = {length_before}")
    
    tokenizer = AutoTokenizer.from_pretrained(flags.model_path)
    tokenizer.model_max_length = flags.encoder_seq_length
    
    parameters = {"tokenizer": tokenizer, "encoder_seq_length": flags.encoder_seq_length, "decoder_seq_length": flags.decoder_seq_length, "get_statistic": flags.get_statistic}
    ds_filtered = ds.filter(partial(filter_length, **parameters), batched=False, num_proc=flags.num_workers)
    logging.info(f"{ds}")
    length_after = len(ds_filtered)
    logging.info(f"after filter: dataset length = {length_after}")
    logging.info(f"keep proportion: {length_after / length_before * 100}%")

    if dataset_stage == "add_prompt":
        serialize_prompt_to_file(data_path, flags.prompt_file)
        prompts = get_prompts(data_path, flags.prompt_file)
        
        parameters = {"tokenizer": tokenizer, "prompts": prompts}
        ds_filtered = ds_filtered.map(partial(add_prompt, **parameters), batched=False, num_proc=flags.num_workers)

    if shuffle:
        ds_filtered = ds_filtered.shuffle()

    return ds_filtered

def get_merged_dataset(split: str=None, data_path: str=None, shuffle: bool=True, dataset_stage: DATA_STAGE="add_prompt") -> Dataset:
    versions = ["v0", "v1", "v2", "v3", "v4", "v5"]
    ds_list = []
    for version in versions:
        ds = get_dataset(split,f"{data_path}{version}/",shuffle,dataset_stage)
        ds_list.append(ds)
    return concatenate_datasets(ds_list)

def filter_length(example, encoder_seq_length: int, decoder_seq_length: int, tokenizer: PreTrainedTokenizer, get_statistic: bool=True):
    input_token_length = len(tokenizer(example["inputs"]).input_ids)
    output_token_length = len(tokenizer(example["target"]).input_ids)
    if get_statistic:
        global output_length
        global input_length
        output_length.append(input_token_length)
        input_length.append(output_token_length)
    return (input_token_length <= encoder_seq_length and output_token_length <= decoder_seq_length)
    
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
        _ = get_dataset("train","../data/processed/v0/")

if __name__=="__main__":
    app.run(main)