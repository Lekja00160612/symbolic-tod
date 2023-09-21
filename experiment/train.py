from functools import partial
from absl import flags, app

import torch
from experiment.dataloader import get_merged_dataset
# from datasets import disable_caching
from transformers import TrainingArguments, Trainer, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
# from torch.utils.data import DataLoader

flags = flags.FLAGS

def train(dataset: Dataset, experiment: str=flags.experiment_name):
    training_args = TrainingArguments(
        output_dir=f"./{flags.output_dir}/{experiment}",
        learning_rate=2e-5,
        num_train_epochs=2,
        weight_decay=0.01,
        max_steps=3000,
        # 128 * 2 * 8 = 2048
        # 173491 * 6 /2048 ~= 500
        # 6M9 / 2048 ~= 1904 
        per_device_train_batch_size=int(8 ),
        gradient_accumulation_steps=int(128 ),

        save_steps=200,
        # logging step
        logging_steps=1,

        # precision_setting
        bf16=True,
        # fp16=True,
        optim="adamw_torch_fused",

        dataloader_num_workers=flags.num_workers # 4 cpu cores
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(flags.model_path, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(flags.model_path)
    tokenizer.model_max_length = flags.encoder_seq_length
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model, max_length=1024)

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
    dataset = get_merged_dataset("train", flags.data_path, shuffle=True, to_tokens=True, dataset_stage="break_task")
    train(dataset=dataset, experiment="task_break")
   

if __name__=="__main__":
    # disable_caching()
    app.run(main)