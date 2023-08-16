from absl import flags, app

from experiment.dataloader import get_dataset
from datasets import disable_caching
from transformers import TrainingArguments, Trainer, AutoModelForSeq2SeqLM

flags = flags.FLAGS


def compute_metrics(eval_pred):
    predictions, labels = eval_pred



def train():

    training_args = TrainingArguments(
        output_dir="./output_dir/",
        learning_rate=1e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        max_steps=5,
        # 16 * 2 * 8 = 256
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,

        save_steps=10,
        # logging step
        logging_steps=1,

        # precision_setting
        # bf16=True,
        fp16=True,
        optim="adamw_torch_fused",

        dataloader_num_workers=4 # 4 cpu cores
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(flags.model_path)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=get_dataset("train", flags.data_path),
        eval_dataset=get_dataset("validation", flags.data_path),
        compute_metrics=compute_metrics
    )

    trainer.train()

def main(_):
    data_loader = get_dataset("train", flags.data_path)
    for i in data_loader:
        print(i)
        input()

if __name__=="__main__":
    disable_caching()
    app.run(main)