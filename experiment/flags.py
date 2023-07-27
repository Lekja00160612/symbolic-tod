""" This file define all flags for main script """
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "google/flan-t5-small", "The huggingface hub path of the model.")
flags.DEFINE_string("data_path", "./data/", "Path where data locate.")
flags.DEFINE_string("prompt_file", "prompt.txt", "File to load/save prompt, must be put in data_path")
flags.DEFINE_integer("encoder_seq_length", 2048, "maximum encoder input length")
flags.DEFINE_integer("decoder_seq_length", 256, "maximum decoder input length")
flags.DEFINE_integer("num_workers", 4, "how much processes to execute data processing script, greatly depends on CPU core number")