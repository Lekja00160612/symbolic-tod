import random
from typing import Literal, Dict
from collections import defaultdict
from functools import partial

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from absl import flags, logging

import experiment.flags
from experiment.generate_prompt import serialize_prompt_to_file, get_prompts

flags = flags.FLAGS

def generate_dataset(data_path: str=None, split: str="train"):
    with open(f"{data_path}{split}.txt", encoding="utf-8") as file:
        for index, line in enumerate(file):
            logging.info(f"loading example {index}")
            yield {"inputs": line.strip().split("\t\t")[0]}

def add_prompt(example, prompts: Dict, tokenizer: PreTrainedTokenizer, prompt_offset: int=3):
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

    if dataset_stage == "add_prompt":
        serialize_prompt_to_file(data_path, flags.prompt_file)
        prompts = get_prompts(data_path, flags.prompt_file)
        
        tokenizer = AutoTokenizer.from_pretrained(flags.model_path)
        tokenizer.model_max_length = flags.encoder_seq_length
        
        parameters = {"tokenizer": tokenizer, "prompts": prompts}
        ds = ds.map(partial(add_prompt, **parameters), batched=False)

    if shuffle:
        ds = ds.shuffle()
    return ds

    # dataloader = DataLoader(ds.with_format("torch"), num_workers=flags.num_workers)
    # return dataloader
