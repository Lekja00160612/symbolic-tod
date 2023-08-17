from functools import partial
from absl import flags, app

import torch
from experiment.dataloader import get_dataset, get_merged_dataset
# from datasets import disable_caching
from transformers import TrainingArguments, Trainer, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizer
from datasets import Dataset
# from torch.utils.data import DataLoader

flags = flags.FLAGS

def train(dataset: Dataset):
    training_args = TrainingArguments(
        output_dir="./output_dir/",
        learning_rate=1e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        max_steps=500,
        # 128 * 2 * 8 = 2048
        # 173491 * 6 /2048 ~= 500
        per_device_train_batch_size=int(8 / 2 / 2),
        gradient_accumulation_steps=int(128 * 2 *2),

        save_steps=40,
        # logging step
        logging_steps=1,

        # precision_setting
        # bf16=True,
        # fp16=True,
        optim="adamw_torch_fused",

        dataloader_num_workers=flags.num_workers # 4 cpu cores
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(flags.model_path, device_map="auto",) # torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(flags.model_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model, max_length=2048)
    def preprocess_function(examples, tokenizer: PreTrainedTokenizer):
        model_inputs = tokenizer(examples["inputs"], max_length=flags.encoder_seq_length, padding="max_length", truncation=True)
        labels = tokenizer(text_target=examples["target"], max_length=flags.decoder_seq_length, padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    dataset = dataset.map(partial(preprocess_function, tokenizer=tokenizer), batched=True, num_proc=flags.num_workers).shuffle()

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=dataset,
        
        # eval_dataset=get_dataset("validation", flags.data_path),
        # compute_metrics=compute_metrics
    )

    trainer.train()

def main(_):
    # data_loader = get_dataset("train", flags.data_path + "v0")
    dataset = get_merged_dataset("train", flags.data_path, False)
    # data_loader = DataLoader(dataset,)
    # for i in data_loader:
    #     print(i)
    #     input()
    train(dataset=dataset)
   

if __name__=="__main__":
    # disable_caching()
    app.run(main)