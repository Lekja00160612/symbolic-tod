conda create -n symbolic-tod python=3.11 -y
conda activate symbolic-tod

cd symbolic-tod && pip install -r requirements.txt && cd .. # step 0
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git # step 1 get dataset

# step 2 generate sgd-x
cp -r ./symbolic-tod/dstc8-changed-code/generate_sgdx_dialogues.py ./dstc8-schema-guided-dialogue/sgd_x/generate_sgdx_dialogues.py && cp -r ./symbolic-tod/dstc8-changed-code/utils.py ./dstc8-schema-guided-dialogue/sgd_x/utils.py

cd ./dstc8-schema-guided-dialogue/ && pip install -r ./sgd_x/requirements.txt && python3 -m sgd_x.generate_sgdx_dialogues  && cd .. 


# step 3 copy data into folder
mkdir ./data/ && mkdir ./data/sgd/ && mkdir ./data/processed/ && mkdir ./data/sgd/v0/ \
&& mkdir ./data/sgd/v1/ && mkdir ./data/sgd/v2/ && mkdir ./data/sgd/v3/ && mkdir ./data/sgd/v4/ \
&& mkdir ./data/sgd/v5/ && mkdir ./data/sgd/v0/train/ && mkdir ./data/sgd/v0/dev/ && mkdir ./data/sgd/v0/test  
cp -R ./dstc8-schema-guided-dialogue/sgd_x/data/v1/ ./data/sgd/
cp -R ./dstc8-schema-guided-dialogue/sgd_x/data/v2/ ./data/sgd/
cp -R ./dstc8-schema-guided-dialogue/sgd_x/data/v3/ ./data/sgd/
cp -R ./dstc8-schema-guided-dialogue/sgd_x/data/v4/ ./data/sgd/
cp -R ./dstc8-schema-guided-dialogue/sgd_x/data/v5/ ./data/sgd/

cp -R ./dstc8-schema-guided-dialogue/train/ ./data/sgd/v0/ \
&& cp -R ./dstc8-schema-guided-dialogue/dev/   ./data/sgd/v0/ \
&& cp -R ./dstc8-schema-guided-dialogue/test/  ./data/sgd/v0/

# step 4, add possible actions to schema, multi process or not by eliminate ending `&`
for i in {0..5}
do
    for split in train dev test
    do
        python3 ./symbolic-tod/addAction2Schema.py \
        --sgd_folder ./data/sgd/v$i/$split \
        --schema_file ./data/sgd/v$i/$split/schema.json \
        --out_schema_file ./data/processed/v$i/$split/schema.json \
        --log_folder ./data/processed/v$i/$split/schema_log/ \
        --alsologtostderr &
    done
done

# step 5, generate final usable txt file, further processed by hf data mapping
# Notice data is in raw form, any experiment or further process will be done in hf datasets
for i in {0..5}
do
    for split in train
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
        --symbolize_percent=0.3 \
        --alsologtostderr &
    done

    for split in dev test
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
        --symbolize_percent=0.0 \
        --alsologtostderr &
    done
done

### Testing
python -m experiment.evaluate --dstc8_data_dir ../data/processed/v0/ --eval_set dev

python -m experiment.dataloader --prompt_file=./prompt.txt --num_workers=20 --encoder_seq_length=1048 --decoder_seq_length=100 --get_statistic=False

python3 -m experiment.train --experiment_name=task_break --encoder_seq_length=1048 --decoder_seq_length=100 