""" This file define all flags for main script """
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("sgd_file", None, "The SGD json file path.")
flags.DEFINE_string("schema_file", None, "Schema file path.")
flags.DEFINE_string("output_file", None, "Output file path.")
flags.DEFINE_string("log_folder", None, "Log folder path.")
flags.DEFINE_string(
    "delimiter",
    "=",
    "Delimiter to separate slot/intent IDs from their descriptions or values.",
)
flags.DEFINE_enum(
    "level",
    "dst",
    ["dst"],
    (
        "Which level of information should be "
        "generated: "
        "dst: Only generate slots for DST"
    ),
)
flags.DEFINE_enum(
    "data_format",
    "full_desc",
    ["full_desc", "item_name"],
    (
        "Format of the schemaless data:"
        "full_desc: Use full language description as the item "
        "description; "
        "item_name: Use item name as the item description."
    ),
)
flags.DEFINE_bool("lowercase", True, "If True, lowercase everything.")
flags.DEFINE_bool("check_multi_domain_data", False, "If True, lowercase everything.")
flags.DEFINE_bool(
    "randomize_items", True, "If True, randomize the order of schema items."
)
flags.DEFINE_enum(
    "multiple_choice",
    "none",
    ("none", "a", "1a"),
    "Whether to use multiple choice prompting for categorical slots."
    "none: Don't use multiple choice prompting. "
    'a: Use the prompt "1: ... a) b) c)." '
    '1a: Use the prompt "1: ... 1a) 1b) 1c)."',
)
flags.DEFINE_float(
    "data_percent", 0.0, "If not 0, the percentage of data to be generated."
)
flags.DEFINE_bool(
    "uniform_domain_distribution",
    False,
    "When data_percent > 0 make sure domains are (close-to) uniform distribution.",
)

# ablation study flag
flags.DEFINE_enum(
    "symbolize_level",
    "none",
    ("none", "value", "action", "action_value", "action_id_value"),
    "If True, s1=request {slot_id}, instead of {slot_name}."  # TODO: add more description
    "none: ",
)

flags.DEFINE_enum(
    "param_symbolize_level",
    "non-categorical",
    ("non-categorical", "all"),
    "Whether to turn normal non-categorical value into symbol."
    "non-categorical: Symbolize only non-categorical slot value."
    "all: Symbolize both categorical and non-categorical slot value.",
)
flags.DEFINE_bool(
    "sort_id",
    True,
    "Whether to sort <output> position based on id position from the input",
)