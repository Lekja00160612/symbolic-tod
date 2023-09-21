""" This file define all flags for main script """
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "google/flan-t5-small", "The huggingface hub path of the model.")
flags.DEFINE_string("data_path", "../data/processed/", "Path where data locate.")
flags.DEFINE_string("prompt_file", "prompt.txt", "File to load/save prompt, must be put in data_path")

flags.DEFINE_boolean("get_statistic", False, "Whether to get statistic for data exploration")
flags.DEFINE_boolean("joint_acc_across_turn", True, "Whether to allow multiple turn join acc")
flags.DEFINE_boolean("use_fuzzy", True, "Use fuzzy string matching for non-categorical slot evaluate")
flags.DEFINE_integer("encoder_seq_length", 1024, "maximum encoder input length")
flags.DEFINE_integer("decoder_seq_length", 100, "maximum decoder input length")
flags.DEFINE_integer("num_workers", 4, "how much processes to execute data processing script, greatly depends on CPU core number")

flags.DEFINE_string("experiment_name", "base_experiment", "Name of the experiment")
flags.DEFINE_string("output_dir", "output_dir", "Directory to save trained model")
flags.DEFINE_string("dstc8_data_dir", None, "dstc directory to get both data and schema")
flags.DEFINE_string("eval_set", None, "Test or Dev set")