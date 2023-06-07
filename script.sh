git clone https://github.com/google-research/task-oriented-dialogue.git
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git #
git clone https://github.com/budzianowski/multiwoz.git

git clone https://github.com/google-research/google-research.git
cp -R ./google-research/schema_guided_dst/ ./schema_guided_data
rm -rf google-research

cd ./dstc8-schema-guided-dialogue/ && python3 -m sgd_x.generate_sgdx_dialogues

python3 -m schema_guided_data.evaluate \
--dstc8_data_dir dstc8-schema-guided-dialogue \
--prediction_dir eval_data/dev \
--eval_set dev \
--output_metric_file eval_result/dev.json

#
python3 addAction2Schema.py \
--sgd_folder dstc8-schema-guided-dialogue/train \
--schema_file dstc8-schema-guided-dialogue/train/schema.json \
--out_schema_file schema.json