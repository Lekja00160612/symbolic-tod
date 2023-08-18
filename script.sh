conda create -n symbolic_tod python=3.11 -y
conda activate symbolic_tod

cd symbolic-tod && pip install -r requirements.txt && cd .. # step 0
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git # step 1 get dataset


cd ./dstc8-schema-guided-dialogue/ && pip install -r ./sgd_x/requirements.txt && python3 -m sgd_x.generate_sgdx_dialogues  && cd .. 
# step 2 generate sgd-x
\cp -r ./symbolic-tod/dstc8-changed-code/generate_sgdx_dialogues.py ./dstc8-schema-guided-dialogue/sgd_x/generate_sgdx_dialogues.py
\cp -r ./symbolic-tod/dstc8-changed-code/utils.py ./dstc8-schema-guided-dialogue/sgd_x/utils.py

mkdir ./data/ && mkdir ./data/sgd/ && mkdir ./data/processed/ && mkdir ./data/sgd/v0/ \
&& mkdir ./data/sgd/v1/ && mkdir ./data/sgd/v2/ && mkdir ./data/sgd/v3/ && mkdir ./data/sgd/v4/ \
&& mkdir ./data/sgd/v5/ && mkdir ./data/sgd/v0/train/ && mkdir ./data/sgd/v0/dev/ && mkdir ./data/sgd/v0/test  
mkdir ./data/ && mkdir ./data/sgd/ && mkdir ./data/processed/ 
cp -R ./dstc8-schema-guided-dialogue/sgd_x/data/v1/ ./data/sgd/
cp -R ./dstc8-schema-guided-dialogue/sgd_x/data/v2/ ./data/sgd/
cp -R ./dstc8-schema-guided-dialogue/sgd_x/data/v3/ ./data/sgd/
cp -R ./dstc8-schema-guided-dialogue/sgd_x/data/v4/ ./data/sgd/
cp -R ./dstc8-schema-guided-dialogue/sgd_x/data/v5/ ./data/sgd/
cp -R ./dstc8-schema-guided-dialogue/train/ ./data/sgd/v0/
cp -R ./dstc8-schema-guided-dialogue/dev/   ./data/sgd/v0/
cp -R ./dstc8-schema-guided-dialogue/test/  ./data/sgd/v0/

for i in {0..5}
do
    for split in train dev test
    do
        python3 ./symbolic-tod/addAction2Schema.py \
        --sgd_folder ./data/sgd/v$i/$split \
        --schema_file ./data/sgd/v$i/$split/schema.json \
        --out_schema_file ./data/processed/v$i/$split/schema.json \
        --log_folder ./data/processed/v$i/$split/schema_log/ \
        --alsologtostderr
    done
done

for i in {0..5}
do
    for split in train dev test
    do
        python3 ./symbolic-tod/symbolic.py \
        --sgd_file="./data/sgd/v$i/$split" \
        --schema_file="./data/processed/v$i/$split/schema.json" \
        --output_file="./data/processed/v$i/$split/$split.txt" \
        --log_folder="./data/processed/v$i/$split/data_log/" \
        --delimiter== \
        --symbolize_level=action_value \
        --level=dst \
        --data_format=full_desc \
        --multiple_choice=1a \
        --alsologtostderr
    done
done

### Testing


python -m experiment.evaluate --dstc8_data_dir ../data/processed/v0/ --eval_set dev

python -m experiment.dataloader --prompt_file=./prompt.txt --num_workers=4 --encoder_seq_length=2048 --decoder_seq_length=300 --get_statistic=False

python3 -m experiment.train