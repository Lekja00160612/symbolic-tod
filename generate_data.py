"""
    Run this file to generate data
"""
import os
import subprocess

### Generate sgd_x dataset
# process = subprocess.run(
#     ["cd ./dstc8-schema-guided-dialogue/ && python3 -m sgd_x.generate_sgdx_dialogues"],
#     check=True,
#     capture_output=True,
#     shell=True,
# )
# print(process)

### Take all version of sgd (include sgd_x) and generate dataset
for version in ["v0", "v1", "v2", "v3", "v4", "v5"]:
    for sub in ["train", "dev", "test"]:
        SGD_ROOT = (
            f"dstc8-schema-guided-dialogue/sgd_x/data/{version}/{sub}/"
            if version != "v0"
            else f"dstc8-schema-guided-dialogue/{sub}/"
        )
        SCHEMA_FILE = SGD_ROOT + "schema.json"
        num_file = len(os.listdir(SGD_ROOT))
        offset = 1
        for i in range(offset, num_file):
            SGD_FILE = f"{SGD_ROOT}/dialogues_{i:03d}.json"
            OUTPUT_FILE = f"schema_guided_data/{version}/{sub}/dialogues_{i:03d}_.json"
            print(f"generating {OUTPUT_FILE}")
            process = subprocess.run(
                [
                    "python3 task-oriented-dialogue/state_tracking/d3st/create_sgd_schemaless_data.py"
                    " --delimiter=="
                    " --level=dst_intent"
                    " --level=dst_intent"
                    " --data_format=full_desc"
                    " --multiple_choice=1a"
                    f" --sgd_file={SGD_FILE}"
                    f" --schema_file={SCHEMA_FILE}"
                    f" --output_file={OUTPUT_FILE}"
                ],
                check=True,
                capture_output=True,
                shell=True,
            )


# for version in ["MultiWOZ_2.2"]:
#     for sub in ["train", "dev", "test"]:
#         MULTIWOZ_ROOT = f"multiwoz/data/{version}/{sub}/"
#         SCHEMA_FILE = f"multiwoz/data/{version}/schema.json"
#         OUTPUT_DIR = f"schema_guided_data/{version}/{sub}/"
#         print(f"generating {version} {OUTPUT_DIR}")
#         process = subprocess.check_output(
#             [
#                 "cd ./task-oriented-dialogue/ && python3 -m state_tracking.d3st.create_multiwoz_schemaless_data"
#                 " --delimiter=="
#                 " --description_type=full_desc_with_domain"
#                 " --multiple_choice=1a"
#                 " --multiwoz_version=2.2"
#                 f" --multiwoz_dir={MULTIWOZ_ROOT}"
#                 f" --schema_file={SCHEMA_FILE}"
#                 f" --output_dir={OUTPUT_DIR}"
#             ],
#             # check=True,
#             # capture_output=True,
#             shell=True,
#         )
