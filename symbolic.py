# pytype: skip-file
r"""Create text format SGD data for generative models.

This is symbolic version, which generates the data format:

[tasks]
[tags]
[params] p0=[slot0 desc] p1=[slot1 desc]... \
[useracts] u0=[useraction0] u1=[useraction1]... u4=out of doamin \
[sysacts] s0=[systemaction0] s1=[systemaction1]... s5=query data base s6=out of domain \
[dependencies] u2,u3->s5; u2,s3->s6 \
[target actions] s5 \
[constraints] user request pi -> system inform pi; target action depend on pi -> system request pi \
[conversation] [user] utterance [system] utterance... \t

[states] p_i=[value_i] p_j=[value_j]... \
[history] u_i u_k; s_j;... \
[next action] s3 s4 s1\
"""
import collections
import copy
from dataclasses import field, dataclass
import json
import os
import pathlib
import random
import string
from typing import Any, Dict, List, Tuple
from absl import app
from absl import flags
from absl import logging

from actionTemplate import ActionTemplate

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_string("sgd_file", None, "The SGD json file path.")
flags.DEFINE_string("schema_file", None, "Schema file path.")
flags.DEFINE_string("output_file", None, "Output file path.")
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
flags.DEFINE_bool(
    "randomize_items", True, "If True, randomize the order of schema items."
)
flags.DEFINE_enum(
    "symbolize_level",
    "none",
    ("none", "value", "action", "action_value", "action_id_value"),
    "If True, s1=request {slot_id}, instead of {slot_name}.",  # TODO: add more description
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
flags.DEFINE_enum(
    "param_symbolize_level",
    "non-categorical",
    ("non-categorical", "non-categorical", "all"),
    "Whether to turn normal non-categorical value into symbol."
    "non-categorical: Symbolize only non-categorical slot value."
    "all: Symbolize both categorical and non-categorical slot value.",
)
flags.DEFINE_bool(
    "sort_id",
    True,
    "Whether to sort <output> position based on id position from the input",
)


@dataclass
class SchemaInfo:
    """Schema information"""

    possible_user_actions: Dict[str, str] = field(default_factory=dict)
    possible_system_actions: Dict[str, str] = field(default_factory=dict)
    possible_values: Dict[str, str] = field(default_factory=dict)
    is_categorical: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)
    dependencies: Dict[str, str] = field(default_factory=dict)


@dataclass
class TurnInfo:
    """Information extracted from dialog turns."""

    out_next_actions: str = "[nextacts]"
    out_history: str = "[history]"
    out_states: str = "[states]"

    in_params: str = "[params]"
    in_user_actions: str = "[useracts]"
    in_system_actions: str = "[sysacts]"
    in_dependencies: str = "[dependencies]"
    in_target_actions: str = "[targetacts]"  # plural for forward compatible
    in_constraints: str = "[constraints]"
    in_conversations: str = "[conversation]"

    user_turn: bool = False
    turn_domain: str = ""
    dialogue_id: str = ""
    turn_id: str = ""
    frame_id: str = ""


def _symbolize() -> str:
    """
    return string  (b24gh), -> decide which slots to change (categorical, time, number_of_people)
    """
    symbolized_text_list = []
    num_parts = random.choice([1])
    for part in range(num_parts):
        part_text_list = []
        num_integers = random.choice([2, 3, 4, 5])
        for integer in range(num_integers):
            part_text_list.append(str(random.randint(0, 9)))

        num_characters = random.choice([2, 3, 4, 5])
        for character in range(num_characters):
            part_text_list.append(random.choice(string.ascii_letters))
        random.shuffle(part_text_list)
        symbolized_text_list.append("".join(part_text_list))
    return " ".join(symbolized_text_list)


def _merge_domain_slot(domain: str, slot_name: str):
    return f"{domain}-{slot_name}"


def load_schema() -> Tuple[collections.OrderedDict, SchemaInfo]:
    """Loads schema items and descriptions.

    Returns:
      A tuple, including an ordered dictionary whose keys are slot names and
      values are placeholder values, and a dictionary whose keys are slot names
      and values are descriptions.
    """
    # We need to preserve state orders since we hope the model learns to generate
    # states in consistent order.
    # TODO(yuancao): We might need to do this for intents/actions as well (in case
    # multiple intents/actions turn out to be a problem).
    # TODO(jeffreyzhao): Clean up how we store schema information by using a
    # dataclass.
    slots = collections.OrderedDict()
    item_desc = SchemaInfo()
    with open(FLAGS.schema_file, "r", encoding="utf-8") as sm_file:
        for schema in json.load(sm_file):
            domain = (
                schema["service_name"].lower()
                if FLAGS.lowercase
                else schema["service_name"]
            )
            slots.update(
                {
                    _merge_domain_slot(domain, slot["name"]): ""
                    for slot in schema["slots"]
                }
            )
            item_desc.params.update(
                {
                    _merge_domain_slot(domain, slot["name"]): slot["description"]
                    for slot in schema["slots"]
                }
            )

            for slot in schema["slots"]:
                name = _merge_domain_slot(domain, slot["name"])
                is_cat = slot["is_categorical"]
                poss_vals = slot["possible_values"]

                # If this is a categorical slot but the possible value are all numeric,
                # consider this as a noncat slot.
                if is_cat and all(v.isdigit() for v in poss_vals):
                    poss_vals = []
                    is_cat = False

                item_desc.is_categorical[name] = is_cat
                item_desc.possible_values[name] = poss_vals

            item_desc.possible_system_actions.update(
                {
                    _merge_domain_slot(domain, f"system_{action_name}"): action_desc
                    for action_name, action_desc in schema[
                        "possible_system_actions"
                    ].items()
                }
            )

            item_desc.possible_user_actions.update(
                {
                    _merge_domain_slot(domain, f"user_{action_name}"): action_desc
                    for action_name, action_desc in schema[
                        "possible_user_actions"
                    ].items()
                }
            )

            item_desc.dependencies.update(
                {
                    _merge_domain_slot(
                        domain,
                        f"system_{ActionTemplate.QUERY_NAME.format(intent_name=intent['name'].lower() if FLAGS.lowercase else intent['name'])}",
                    ): intent["required_slots"]
                    for intent in schema["intents"]
                }
            )
    return slots, item_desc


def _process_user_turn(
    frame: Dict[str, Dict[str, Any]],
    turn_info: TurnInfo,
    cumu_slots: collections.OrderedDict,
    item_desc: SchemaInfo,
    state_dict: Dict[str, List[str]],
) -> Dict[str, int]:
    """Updates turn_info and cumu_slots based on user turn input.

    Args:
      state: A dictionary containing state info.
      turn_info: A TurnInfo object accmulating essential info from each turn.
      cumu_slots: An OrderedDict containing cmumulative slot information.
      domain: A string, domain (service) of the turn.
      item_desc: A dictionary of items and their descriptions.
      state_dict: A dictionary of states from the current turn.

    Returns:
      A dictionary that maps slot descriptions to ids.
    """
    state = frame["state"]
    domain = frame["service"].lower() if FLAGS.lowercase else frame["service"]

    slot_values = state["slot_values"]
    domain_slot_values = {}
    for slot, value in slot_values.items():
        domain_slot_values[_merge_domain_slot(domain, slot)] = value
    slot_values = domain_slot_values

    # Order of slots is preserved. Meanwhile new values of the same
    # slots will overwrite existing ones.
    for slot, value in slot_values.items():
        if slot not in cumu_slots:
            raise ValueError(f"Unknown slot: {slot}.")
        # cumu_slots.update({slot: " | ".join(value)})

    def _handle_params():
        param_name_to_id = {}
        params = list(item_desc.params.keys())
        if FLAGS.randomize_items:
            random.shuffle(params)
        # In multi-domain turn case, desc_prefix already contains desc from the
        # previous domain.
        param_id = 0  # len(state_dict["slot_desc"])
        for param in params:
            if domain not in param.split("-")[0]:
                continue

            param_name = param.split("-")[1]
            if FLAGS.data_format == "full_desc":
                desc = item_desc.params[param]
            elif FLAGS.data_format == "item_name":
                desc = param_name

            # If we are generating with multiple choice, append this prompt.
            if FLAGS.multiple_choice != "none" and item_desc.is_categorical[param]:
                possible_values = item_desc.possible_values[param]
                if FLAGS.randomize_items:
                    random.shuffle(possible_values)
                assert len(possible_values) < len(string.ascii_lowercase)
                letters = list(string.ascii_lowercase)

                possible_values_pieces = []
                for letter, value in zip(letters, possible_values):
                    if FLAGS.multiple_choice == "1a":
                        possible_values_pieces.append(f"{param_id}{letter}) {value}")
                    elif FLAGS.multiple_choice == "a":
                        possible_values_pieces.append(f"{letter}) {value}")
                desc = _add_string(desc, " ".join(possible_values_pieces))
            _id = f"p{param_id}"
            desc = _add_string(_id, desc, delimiter=FLAGS.delimiter)
            turn_info.in_params = _add_string(turn_info.in_params, desc)
            # Description prefix to be included in each turn.
            desc_to_id[param] = _id
            param_name_to_id[param_name] = _id
            param_id += 1
        return param_name_to_id

    def _handle_acts(speaker=None, param_name_to_id: Dict = None):
        # Handle actions
        if speaker == "user":
            acts = list(item_desc.possible_user_actions.keys())
            act_desc = item_desc.possible_user_actions
            id_prefix = "u"
        elif speaker == "system":
            acts = list(item_desc.possible_system_actions.keys())
            act_desc = item_desc.possible_system_actions
            id_prefix = "s"
        else:
            logging.fatal(f"Unavailble speaker: {speaker}")

        if FLAGS.randomize_items:
            random.shuffle(acts)

        act_id = 0
        for act in acts:
            if domain not in act.split("-")[0]:
                continue

            if FLAGS.data_format == "full_desc":
                desc = act_desc[act]
                if "action" in str(FLAGS.symbolize_level):
                    logging.debug(f"{param_name_to_id=}")
                    for param_name in param_name_to_id:
                        desc = (
                            desc
                            if param_name not in desc
                            else desc.replace(param_name, param_name_to_id[param_name])
                        )

            elif FLAGS.data_format == "item_name":
                desc = act.split("-")[1]
            _id = f"{id_prefix}{act_id}"
            desc = _add_string(_id, desc, delimiter=FLAGS.delimiter)

            if speaker == "user":
                turn_info.in_user_actions = _add_string(turn_info.in_user_actions, desc)
            elif speaker == "system":
                turn_info.in_system_actions = _add_string(
                    turn_info.in_system_actions, desc
                )

            desc_to_id[act] = _id
            act_id += 1
        if speaker == "user":
            pass
        elif speaker == "system":
            for dependency in item_desc.dependencies.keys():
                if domain not in dependency.split("-")[0]:
                    continue
                if dependency in desc_to_id:
                    continue
                # else:
                #     logging.fatal(f"{dependency}")
                desc = f"{_id}={ActionTemplate.SYSTEM_QUERY.format(intent_name=dependency.split('-')[1])}"
                _id = f"s{act_id}"
                turn_info.in_system_actions = _add_string(
                    turn_info.in_system_actions, desc
                )
                desc_to_id[dependency] = _id
                act_id += 1

    def _handle_dependencies():
        depend_desc = []
        dependencies = list(item_desc.dependencies.keys())
        if FLAGS.randomize_items:
            random.shuffle(dependencies)
        for dependency in dependencies:
            if domain not in dependency.split("-")[0]:
                continue

            depend_variables_id = []
            for required_slot in item_desc.dependencies[dependency]:
                try:
                    potential_actions = ["user_inform", "system_offer"]
                    actions = [
                        f"{domain}-{potential_action}_{required_slot}"
                        for potential_action in potential_actions
                    ]
                    for action in actions:
                        _id = desc_to_id.get(action, None)
                        if _id is not None:
                            break
                    else:
                        raise Exception
                    depend_variables_id.append(_id)
                except Exception:
                    logging.error(f"{actions} id does not exist")
            desc = f"{', '.join(depend_variables_id)} -> {desc_to_id[dependency]}"
            depend_desc.append(desc)
        turn_info.in_dependencies = _add_string(
            turn_info.in_dependencies, "; ".join(depend_desc)
        )

    def _handle_target_acts():
        intent = state["active_intent"]
        # No active intent
        if intent == "NONE":
            turn_info.in_target_actions = _add_string(
                turn_info.in_target_actions, "none"
            )
            return
        intent = intent.lower() if FLAGS.lowercase else intent
        intent = _merge_domain_slot(
            domain, f"system_{ActionTemplate.QUERY_NAME.format(intent_name=intent)}"
        )
        try:
            _id = desc_to_id[intent]
        except Exception:
            logging.error(f"{intent} not found id")
        turn_info.in_target_actions = _add_string(turn_info.in_target_actions, _id)

    def _handle_constraints():
        constraints = [
            "user request p_i -> system inform p_i",
            "target action depend on p_i -> system request p_i",
        ]
        if FLAGS.randomize_items:
            random.shuffle(constraints)
        turn_info.in_constraints = _add_string(
            turn_info.in_constraints, "; ".join(constraints)
        )

    def _handle_states_and_conversation():
        desc = []
        for slot, slot_values in state["slot_values"].items():
            slot = _merge_domain_slot(domain, slot)
            slot_value = slot_values[0]
            slot_value = slot_value.lower() if FLAGS.lowercase else slot_value
            try:
                _id = desc_to_id[slot]
            except Exception:
                logging.error(f"{slot} not found id")
            desc.append((_id, slot_value, item_desc.is_categorical[slot]))
        if FLAGS.sort_id is True:
            desc.sort(key=lambda x: x[0])
        conversation = copy.deepcopy(state_dict["conversation"])
        if "value" in str(FLAGS.symbolize_level):
            for index, param in enumerate(desc):
                param_id = param[0]
                param_value = param[1]
                param_is_categorical = param[2]
                if (
                    param_is_categorical
                    and FLAGS.param_symbolize_level == "non-categorical"
                ):
                    continue
                symbolize_value = _symbolize()
                desc[index] = (param_id, symbolize_value)
                logging.debug(f"{param[1]} is replaced with {symbolize_value}")

                def _replace_symbolic(turn_utterance):
                    speaker = turn_utterance[0]
                    utterance = turn_utterance[1]
                    if param_value in utterance:
                        logging.debug(f"{utterance}")
                        utterance = utterance.replace(param_value, symbolize_value)
                        logging.debug(f"{utterance}")
                    return [speaker, utterance]

                conversation = list(map(_replace_symbolic, conversation))
        conversation = list(map(lambda x: f"[{x[0]}] {x[1]}", conversation))
        desc = map(lambda x: f"{x[0]}={x[1]}", desc)
        turn_info.in_conversations = _add_string(
            turn_info.in_conversations, " ".join(conversation)
        )
        turn_info.out_states = _add_string(turn_info.out_states, " ".join(desc))

    def _handle_history():
        actions = []
        for action in frame["actions"]:
            act = action["act"].lower() if FLAGS.lowercase else action["act"]
            slot = action["slot"].lower() if FLAGS.lowercase else action["slot"]
            if slot == "":
                actions.append(_merge_domain_slot(domain, f"user_{act}"))
            elif slot == "intent":
                # only use when the action is related to intent
                value = action["values"][0].lower() if slot == "intent" else None
                actions.append(_merge_domain_slot(domain, f"user_{act}_{value}"))
            else:
                actions.append(_merge_domain_slot(domain, f"user_{act}_{slot}"))
        state_dict["history"].append(actions)

        desc_list = []
        for actions in state_dict["history"]:
            action_ids = []
            for action in actions:
                action = action.lower() if FLAGS.lowercase else action
                try:
                    _id = desc_to_id[action]
                    action_ids.append(_id)
                except Exception:
                    logging.error(f"{action} not found")
            desc = " ".join(action_ids)
            desc_list.append(desc)
        logging.debug(
            f"{len(state_dict['history'])}, {turn_info.out_history} {'; '.join(desc_list)}"
        )
        turn_info.out_history = _add_string(turn_info.out_history, "; ".join(desc_list))

    def _handle_next_actions():
        # will be handle during system turn
        pass

    # Clean up.
    desc_to_id = {}
    example_turn_info = TurnInfo()

    turn_info.in_constraints = example_turn_info.in_constraints
    turn_info.in_dependencies = example_turn_info.in_dependencies
    turn_info.in_params = example_turn_info.in_params
    turn_info.in_system_actions = example_turn_info.in_system_actions
    turn_info.in_user_actions = example_turn_info.in_user_actions
    turn_info.in_target_actions = example_turn_info.in_target_actions
    turn_info.out_history = example_turn_info.out_history
    turn_info.out_next_actions = example_turn_info.out_next_actions
    turn_info.out_states = example_turn_info.out_states
    turn_info.in_conversations = example_turn_info.in_conversations

    param_name_to_id = _handle_params()
    _handle_acts("user", param_name_to_id)
    _handle_acts("system", param_name_to_id)
    _handle_dependencies()
    _handle_target_acts()
    _handle_constraints()

    _handle_states_and_conversation()
    _handle_history()
    _handle_next_actions()
    return desc_to_id, turn_info


def _process_agent_turn(
    frame: Dict[str, Dict[str, Any]],
    turn_info: TurnInfo,
    cumu_slots: collections.OrderedDict,
    item_desc: SchemaInfo,
    state_dict: Dict[str, List[str]],
    desc_to_id: Dict[str, str],
) -> None:
    """Updates turn_info based on the system actions.

    Args:
      actions: A list of strings for system actions.
      turn_info: A Turninfo object accmulating essential info from each turn.
      domain: A string, domain (service) of the current turn.
      desc_to_slot_id: A dictionary that maps descriptions to slot ids.
    """
    domain = frame["service"].lower() if FLAGS.lowercase else frame["service"]

    def _handle_next_action_and_history():
        eliminate_acts = []
        next_actions = []
        if frame.get("service_call", None):
            eliminate_acts.append("offer")
            method_name = frame["service_call"]["method"]
            method_name = method_name.lower() if FLAGS.lowercase else method_name
            query_action = _merge_domain_slot(
                domain,
                f"system_{ActionTemplate.QUERY_NAME.format(intent_name=method_name)}",
            )
            next_actions.append(query_action)

        for action in frame["actions"]:
            act = action["act"].lower() if FLAGS.lowercase else action["act"]
            # eliminate
            if act in eliminate_acts:
                continue
            slot = action["slot"].lower() if FLAGS.lowercase else action["slot"]
            if slot == "":
                next_actions.append(_merge_domain_slot(domain, f"system_{act}"))
            elif slot == "intent":
                # only use when the action is related to intent
                value = action["values"][0].lower() if slot == "intent" else None
                next_actions.append(_merge_domain_slot(domain, f"system_{act}_{value}"))
            else:
                next_actions.append(_merge_domain_slot(domain, f"system_{act}_{slot}"))
        state_dict["history"].append(next_actions)

        for action in next_actions:
            try:
                turn_info.out_next_actions = _add_string(
                    turn_info.out_next_actions, desc_to_id[action]
                )
            except Exception:
                # print(desc_to_id)
                logging.error(f"Cannot resolve {action} to id")

    _handle_next_action_and_history()

    return desc_to_id, turn_info


def _add_string(lhs: str, rhs: str, delimiter: str = " "):
    return delimiter.join([str(lhs), str(rhs)])


def process_turn(
    turn: Dict[str, Any],
    turn_info: TurnInfo,
    cumu_slots: collections.OrderedDict,
    item_desc: SchemaInfo,
    turn_id: int,
    desc_to_id: Dict[str, str],
    state_dict: Dict[str, str],
) -> str:
    """Collects information from a single turn.

    Args:
      turn: A dictionary containing original turn structure.
      turn_info: A dictionary accmulating essential info from each turn.
      cumu_slots: An OrderedDict containing cumumulative slot information.
      item_desc: A dictionary of scheam items and their descriptions.
      prefix: A string of the schema item description prefix.
      turn_id: Integer index of turn in dialogue.

    Returns:
      Prefix string (item descriptions) from the current turn and per-frame
      TurnInfo objects.
    """
    speaker = turn["speaker"].lower()
    turn_info.user_turn = speaker == "user"
    turn_info.turn_id = str(turn_id)

    utterance = turn["utterance"]
    utterance = utterance.lower() if FLAGS.lowercase else utterance
    state_dict["conversation"].append((speaker, utterance))
    if FLAGS.lowercase:
        turn_info.in_conversations = turn_info.in_conversations.lower()

    turn_info_per_frame = []
    for frame_id, frame in enumerate(turn["frames"]):
        # Multi-service turns are possible, each frame corresponds to one
        # service (domain).
        domain = frame["service"].lower() if FLAGS.lowercase else frame["service"]
        turn_info.frame_id = str(frame_id)
        turn_info.domain = domain

        if turn_info.user_turn:
            desc_to_id, turn_info = _process_user_turn(
                frame, turn_info, cumu_slots, item_desc, state_dict
            )
        else:
            desc_to_id, turn_info = _process_agent_turn(
                frame, turn_info, cumu_slots, item_desc, state_dict, desc_to_id
            )

        turn_info_per_frame.append(copy.deepcopy(turn_info))

    return desc_to_id, turn_info_per_frame


def write_examples(turn_list: List[TurnInfo], out_file) -> None:
    """Format output example strings and write to file.

    Args:
      turn_list: A list of dict accmulating essential info from each turn.
      out_file: A GFile object for file output.
    """
    for turn_info in turn_list:
        # Write samples to file. Each example is divided into two parts
        # separated by \t, the first part being inputs to the model, and the
        # second part are labels for prediction.
        input_text = ""
        output_text = ""
        if FLAGS.level == "dst":
            if not turn_info.user_turn:
                # Only output at user turns.
                input_text = f"{turn_info.in_params} {turn_info.in_user_actions} {turn_info.in_system_actions} {turn_info.in_dependencies} {turn_info.in_target_actions} {turn_info.in_constraints} {turn_info.in_conversations}"
                output_text = f"{turn_info.out_states} {turn_info.out_history} {turn_info.out_next_actions}"
        if input_text != "" and output_text != "":
            # Add dialogue ID, turn ID and frame ID to the target for later eval.
            # Occasionally some examples include newline in the middle.
            example = (
                f"{input_text}\t{output_text}\t"
                + f"{turn_info.dialogue_id}\t{turn_info.turn_id}\t{turn_info.frame_id}"
            )
            if FLAGS.lowercase:
                example = example.lower()

            logging.debug(example)
            out_file.write(f"{example}\n")


def example_filter(turn_list: List[TurnInfo]):
    """Extract specified percentage of examples.

    And ensure uniform domain distribution if specified.

    Args:
      turn_list: A list of TurnInfo containing all examples.

    Returns:
      Specified percentage of examples, with uniform domain distribution if
      needed.
    """
    if FLAGS.data_percent == 0.0:
        return turn_list

    out_sample_num = int(len(turn_list) * FLAGS.data_percent)
    if not FLAGS.uniform_domain_distribution:
        if FLAGS.randomize_items:
            random.shuffle(turn_list)
        return turn_list[:out_sample_num]
    else:
        domain_examples = {}
        domain_id = {}
        domain_count = 0
        for turn in turn_list:
            if turn.turn_domain in domain_id:
                domain_examples[domain_id[turn.turn_domain]].append(turn)
            else:
                domain_examples[domain_count] = [turn]
                domain_id[turn.turn_domain] = domain_count
                domain_count += 1

        # How many examples from each domain has been added to the final list.
        consumed_examples = {d: 0 for d in range(domain_count)}
        uniform_turn_list = []
        for s in range(out_sample_num):
            # Find first domain that still has unused examples.
            domain_id = s % domain_count
            for d in range(domain_count):
                cand_domain = (domain_id + d) % domain_count
                if len(domain_examples[cand_domain]) > consumed_examples[cand_domain]:
                    domain_id = cand_domain
                    break

            uniform_turn_list.append(
                domain_examples[domain_id][consumed_examples[domain_id]]
            )
            consumed_examples[domain_id] += 1

        if FLAGS.randomize_items:
            random.shuffle(uniform_turn_list)

        return uniform_turn_list


def generate_data(ordered_slots, item_desc):
    """Generate SGD examples in text format.

    Args:
      ordered_slots: An ordered dictionary containing slot names.
      item_desc: A dictionary containing items and their descriptions.
    """
    if not os.path.isdir(os.path.dirname(FLAGS.output_file)):
        os.makedirs(os.path.dirname(FLAGS.output_file))
    with open(FLAGS.output_file, "w", encoding="utf-8") as out_file:
        all_turns_per_frame = []

        sgd_folder = pathlib.Path(FLAGS.sgd_file)
        for sgd_file in sgd_folder.rglob("dialogues_*.json"):
            logging.info(f"processing {sgd_file}")
            with open(sgd_file, "r", encoding="utf-8") as sgd_in:
                for dlg in json.load(sgd_in):
                    if len(dlg["services"]) > 1:
                        dialogue_id = dlg["dialogue_id"]
                        logging.warning(f"Skipping file {dialogue_id.split('_')[0]}")
                        break
                    # Cumulative states throughout this dialog.
                    cumu_slots = copy.deepcopy(ordered_slots)
                    turn_info = TurnInfo()
                    desc_to_id = {}
                    state_dict = {"history": [], "conversation": []}  # TODO: clean up
                    turn_info.dialogue_id = dlg["dialogue_id"]
                    for turn_idx, turn in enumerate(dlg["turns"]):
                        desc_to_id, per_frame_turn_info = process_turn(
                            turn,
                            turn_info,
                            cumu_slots,
                            item_desc,
                            turn_idx,
                            desc_to_id,
                            state_dict,
                        )
                        logging.debug(per_frame_turn_info)
                        all_turns_per_frame.extend(copy.deepcopy(per_frame_turn_info))

                write_examples(example_filter(all_turns_per_frame), out_file)


def main(_):
    slots, item_desc = load_schema()
    generate_data(slots, item_desc)


if __name__ == "__main__":
    flags.mark_flag_as_required("sgd_file")
    flags.mark_flag_as_required("schema_file")
    flags.mark_flag_as_required("output_file")
    app.run(main)
