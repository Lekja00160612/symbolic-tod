import os
import re
import json
from typing import List, Dict
import collections
from enum import Enum

from absl import logging, flags, app
import numpy as np
from experiment import metrics
import experiment.flags

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS

params_prefix = "[params]"
useracts_prefix = "[useracts]"
state_prefix = "[states]"
history_prefix = "[history]"
nextacts_prefix = "[nextacts]"
sysacts_prefix = "[sysacts]"

output_part_delimiter = input_part_delimiter = meta_delimiter = "\t"
useracts_delimiter = sysacts_delimiter = state_delimiter = history_turn_delimiter = ";"
action_delimiter = ","
state_key_value_seperator = "="

metrics.QUERY_MATCH = "query_match"
ALL_SERVICES = "#ALL_SERVICES"
SEEN_SERVICES = "#SEEN_SERVICES"
UNSEEN_SERVICES = "#UNSEEN_SERVICES"

class F1(Enum):
    TP = (1,1)
    FP = (0,1)
    TN = (0,0)
    FN = (1,0)

def compute_f1(sequence: List[F1]):
    TP = sequence.count(F1.TP)
    FP = sequence.count(F1.FP)
    FN = sequence.count(F1.FN)
    if FN + TP != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    if TP + FP != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0
    if recall + precision != 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0
    return metrics.F1Scores(f1=f1,recall=recall,precision=precision)

pseudo_input = "[params] p1=address of restaurant; p30=phone number to contact restaurant; p33=tentative date of restaurant reservation; p41=the category of food offered by the restaurant; p43=whether the restaurant has outdoor seating available 43a) false 43b) true; p48=average user rating for restaurant on a scale of 5; p68=number of seats to reserve at the restaurant; p92=city where the restaurant is located; p99=tentative time of restaurant reservation; p102=whether the restaurant has adequate vegetarian options 102a) true 102b) false; p121=name of the restaurant; p129=price range for the restaurant 129a) cheap 129b) pricey 129c) moderate 129d) ultra high-end	[useracts] u17=user select item; u39=user inform undefined information; u41=user request undefined information; u45=user inform p102; u53=user thank; u58=user goodbye; u60=user request p129; u106=user deny the offer; u107=user inform p121; u109=user inform p43; u114=user request alternative items; u138=user request p1; u139=user request p48; u144=user request p41; u153=user inform p41; u166=user inform p33; u170=user request p102; u176=user want to reserverestaurant; u178=user want to findrestaurants; u191=user inform p99; u215=user inform p92; u218=user request p43; u223=user request p30; u224=user agree to the offer; u242=user inform p68; u263=user inform p129	[sysacts] s19=inform p43; s28=ask to confirm value of p33; s32=ask to confirm value of p68; s64=ask to confirm undefined information; s65=ask to confirm value of p92; s75=ask to confirm value of p121; s93=inform number of items satified user; s123=request p99; s125=offer user p121; s128=goodbye user; s136=request undefined information; s146=ask to confirm value of p99; s157=request p121; s163=ask user if they need anything more; s177=query findrestaurants api; s201=request p92; s206=inform p102; s208=offer user reserverestaurant; s212=inform p1; s213=notify failure; s215=offer user p99; s232=inform p41; s246=inform p129; s248=inform p48; s250=inform p30; s255=query reserverestaurant api; s268=notify success; s281=inform undefined information; s284=offer user p92; s302=request p41; s305=offer user p33; s313=offer user p68	[dependencies] u153, u215 -> s177; s65, s75, s125, s146, s215, s284, u107, u191, u215 -> s255	[targetacts] s255	[constraints] user request param x, system inform the param x; target actions depend on param x and user did not inform, system request param x; after offer an item and user like the item, offer user to purchase	[conversation] [user] i want to make a restaurant reservation for vt8u3s9v people at 8y0ep779."
pseudo_ref = "[states] p68=vt8u3s9v; p99=8y0ep779	[history] u176, u191, u242	[nextacts] s157, s201"
pseudo_hyp = "[states] p68=vt8u3s9v; p99=8y0ep779	[history] u176, u191, u242	[nextacts] s157, s201"
pseudo_meta = "dialogue_id:1_00000	turn_id:1	frame_domain:restaurants_2	frame_id:0"

def compare_slot_values(ref_slot_values: Dict[str,str], 
                        hyp_slot_values: Dict[str,str], 
                        service_slots: List[str],
                        service_slots_is_categorical: Dict[str,bool],
                        use_fuzzy_match: bool=True,):
    """Get per slot accuracy.

    Args:
    ref_slot_values: dictionary contains values for each active slot in label (ground truth) example.
    hyp_slot_values: dictionary contains values for each active slot in prediction (ground truth) example.
    service_slots_value: a list of slot names in the service.
    service_slots_is_categorical: a dictionary. We use it to obtain the
        list of slots in the service and infer whether a slot is categorical.
    use_fuzzy_match: whether to use fuzzy string matching for comparing
        non-categorical slot values.

    Returns:
    list_corr: a list whose values represent per slot accuracy, ordered as the input service_slots.
    """
    # Compare slot values
    list_corr = []
    for slot_id in service_slots:
        # REF=active
        if slot_id in ref_slot_values:
            if slot_id in hyp_slot_values:
                if service_slots_is_categorical[slot_id] == True:
                    list_corr.append(float(ref_slot_values[slot_id] == hyp_slot_values[slot_id]))
                else:
                    use_fuzzy_match = True # TODO:FLAGS.get_flag_value("use_fuzzy", True)
                    list_corr.append(metrics.noncat_slot_value_match([ref_slot_values[slot_id]], hyp_slot_values[slot_id], use_fuzzy_match))
            else:
                list_corr.append(0.0)
        # REF=off
        else:
            # REF=off but HYP=on
            if slot_id in hyp_slot_values:
                list_corr.append(0.0)
            # REF=off and HYP=off
            else:
                list_corr.append(1.0)
    return list_corr

def get_average_and_joint_goal_accuracy(ref_slot_values: Dict[str,str], 
                                        hyp_slot_values: Dict[str,str], 
                                        service_slots: List[str],
                                        service_slots_is_categorical: Dict[str,bool],
                                        use_fuzzy_match: bool=True,):
    """Get average and joint goal accuracy for an example.

    Args:
    ref_slot_values: dictionary contains values for each active slot in label (ground truth) example.
    hyp_slot_values: dictionary contains values for each active slot in prediction (ground truth) example.
    service_slots_value: a list of slot names in the service.
    service_slots_is_categorical: a dictionary. We use it to obtain the
        list of slots in the service and infer whether a slot is categorical.
    use_fuzzy_match: whether to use fuzzy string matching for comparing
        non-categorical slot values.

    Returns:
        goal_acc: a dict whose values are average / joint
            all-goal / categorical-goal / non-categorical-goal accuracies.
    """
    goal_acc = {}

    list_acc = compare_slot_values(
        ref_slot_values, hyp_slot_values,
        service_slots, service_slots_is_categorical, use_fuzzy_match)

    # (4) Average goal accuracy.
    active_acc = [acc for acc, slot_name in zip(list_acc, service_slots) if slot_name in ref_slot_values]
    goal_acc[metrics.AVERAGE_GOAL_ACCURACY] = np.mean(
        active_acc) if active_acc else metrics.NAN_VAL
    # (4-a) categorical.
    active_cat_acc = [
        acc for acc, slot_name in zip(list_acc, service_slots)
        if slot_name in ref_slot_values and service_slots_is_categorical[slot_name] is True
    ]
    goal_acc[metrics.AVERAGE_CAT_ACCURACY] = (
        np.mean(active_cat_acc) if active_cat_acc else metrics.NAN_VAL)
    # (4-b) non-categorical.
    active_noncat_acc = [
        acc for acc, slot_name in zip(list_acc, service_slots)
        if slot_name in ref_slot_values and service_slots_is_categorical[slot_name] is False
    ]
    goal_acc[metrics.AVERAGE_NONCAT_ACCURACY] = (
    np.mean(active_noncat_acc) if active_noncat_acc else metrics.NAN_VAL)

    # (5) Joint goal accuracy.
    goal_acc[metrics.JOINT_GOAL_ACCURACY] = np.prod(list_acc) if list_acc else metrics.NAN_VAL
    # (5-a) categorical.
    cat_acc = [acc for acc, slot_name in zip(list_acc, service_slots) if service_slots_is_categorical[slot_name] is True]
    logging.debug(f"{cat_acc=}")
    goal_acc[metrics.JOINT_CAT_ACCURACY] = np.prod(cat_acc) if cat_acc else metrics.NAN_VAL
    # (5-b) non-categorical.
    noncat_acc = [acc for acc, slot_name in zip(list_acc, service_slots) if service_slots_is_categorical[slot_name] is False]
    logging.debug(f"{noncat_acc=}")
    goal_acc[metrics.JOINT_NONCAT_ACCURACY] = np.prod(
        noncat_acc) if noncat_acc else metrics.NAN_VAL

    return goal_acc

def parse_input(input: str, take_params_is_categorical=True, take_user_request_action_ids=True, take_system_query_action_ids=True):
    input_parts = input.split(input_part_delimiter)
    return_dict = {}
    if take_params_is_categorical:
        params_is_categorical = {}
        params = list(filter(lambda x: params_prefix in x, input_parts))
        if len(params) == 1:
            params = params[0].removeprefix(params_prefix)
        else:
            logging.fatal(f"Only one instance of each tag can appear")
        for param in params.strip().split(state_delimiter):
            param_id = param.split(state_key_value_seperator)[0].strip()
            assert re.match(r"p\d+", param_id), f"param_id got wrong format: {param_id}"
            
            param_id_int = re.sub(r"p(\d+)",r"\1",param_id)
            
            if f"{param_id_int}a)" not in param:
                # This param is non_categorical
                params_is_categorical[param_id] = False 
            else:
                # This param is categorical
                params_is_categorical[param_id] = True
        return_dict["params_is_categorical"] = params_is_categorical
    
    if take_user_request_action_ids:
        user_request_action_ids = []
        useracts = list(filter(lambda x: useracts_prefix in x, input_parts))
        if len(useracts) == 1:
            useracts = useracts[0].removeprefix(useracts_prefix)
        else:
            logging.fatal(f"Only one instance of each tag can appear")
        useracts = useracts.split(useracts_delimiter)
        for action in useracts:
            action = action.strip().split(state_key_value_seperator)
            assert len(action) == 2
            action_id, action_desc = action
            if "user request" in action_desc:
                user_request_action_ids.append(action_id)
        return_dict["user_request_action_ids"] = user_request_action_ids
    
    if take_system_query_action_ids:
        system_query_action_ids = []
        sysacts = list(filter(lambda x: sysacts_prefix in x, input_parts))
        if len(sysacts) == 1:
            sysacts = sysacts[0].removeprefix(sysacts_prefix)
        else:
            logging.fatal(f"Only one instance of each tag can appear")
        sysacts = sysacts.split(sysacts_delimiter)
        for action in sysacts:
            action = action.strip().split(state_key_value_seperator)
            assert len(action) == 2
            action_id, action_desc = action
            if re.match(r"query \w+ api", action_desc.strip()):
                system_query_action_ids.append(action_id)
        return_dict["system_query_action_ids"] = user_request_action_ids
    return return_dict

def get_per_example_metrics_score(input: str, label: str, predict: str, 
                               average_and_joint_goal_accuracy=True,                
                               next_action_query_match=True, 
                               request_slots_f1=True, 
                               ):
    label_state, label_history, label_nextacts = label.split(output_part_delimiter)
    predict_state, predict_history, predict_nextacts = predict.split(output_part_delimiter)
    logging.debug(f"{predict_state=}")
    assert state_prefix in label_state and state_prefix in predict_state, \
            f"Expecting both label_state and predict_state to have the correct prefix ({state_prefix}), got:\n{label_state}\n{predict_state}"
    assert history_prefix in label_history and history_prefix in predict_history, \
            f"Expecting both label_history and predict_history to have the correct prefix ({history_prefix}), got:\n{label_history}\n{predict_history}"
    assert nextacts_prefix in label_nextacts and nextacts_prefix in predict_nextacts, \
            f"Expecting both label_nextacts and predict_nextacts to have the correct prefix ({nextacts_prefix}), got:\n{label_nextacts}\n{predict_nextacts}"

    # Construct input dictionary
    input_info = parse_input(input)
    logging.debug(input_info)

    example_metrics = {}
    # Construct reference and hypothesis dictionary
    if average_and_joint_goal_accuracy:
        params_is_categorical = input_info["params_is_categorical"]
        hyp_params_values = {}
        for param in predict_state.removeprefix(state_prefix).split(state_delimiter):
            param_id, param_value = param.strip().split(state_key_value_seperator)
            hyp_params_values[param_id] = param_value
        logging.debug(f"{hyp_params_values=}")
        
        ref_params_values = {}
        for param in label_state.removeprefix(state_prefix).split(state_delimiter):
            param_id, param_value = param.strip().split(state_key_value_seperator)
            ref_params_values[param_id] = param_value
        logging.debug(f"{ref_params_values=}")

        goal_acc = get_average_and_joint_goal_accuracy(
            ref_params_values, 
            hyp_params_values, 
            params_is_categorical.keys(), 
            params_is_categorical, True) #FLAGS.use_fuzzy)
        
        example_metrics.update(goal_acc)
    
    if request_slots_f1:
        user_request_action_ids = input_info["user_request_action_ids"]
        
        logging.debug(f"{user_request_action_ids=}")
        label_latest_turn_actions = label_history.removeprefix(history_prefix).split(
            history_turn_delimiter)[-1].strip().split(action_delimiter)
        label_latest_turn_requested_actions = [action_id.strip() for action_id in label_latest_turn_actions
                                               if action_id.strip() in user_request_action_ids]
        logging.debug(f"{label_latest_turn_requested_actions=}")
        predict_latest_turn_actions = predict_history.removeprefix(history_prefix).split(
            history_turn_delimiter)[-1].strip().split(action_delimiter)
        predict_latest_turn_requested_actions = [action_id.strip() for action_id in predict_latest_turn_actions
                                                 if action_id.strip() in user_request_action_ids]
        logging.debug(f"{predict_latest_turn_requested_actions=}")
        f1_scores = metrics.compute_f1(list_ref=label_latest_turn_requested_actions, list_hyp=predict_latest_turn_requested_actions) 
        example_metrics[metrics.REQUESTED_SLOTS_F1] = f1_scores
    
    if next_action_query_match:
        system_query_action_ids = input_info["system_query_action_ids"]
        label_next_action_ids = label_nextacts.removeprefix(nextacts_prefix).split(action_delimiter)
        label_next_action_ids = [action.strip() for action in label_next_action_ids
                                 if action.strip() in system_query_action_ids]
        predict_next_action_ids = label_nextacts.removeprefix(nextacts_prefix).split(action_delimiter)
        predict_next_action_ids = [action.strip() for action in predict_next_action_ids
                                    if action.strip() in system_query_action_ids]
        # by default, at most 1 query is available
        if len(label_next_action_ids) == 1:
            if len(predict_next_action_ids) == 1 and predict_next_action_ids[0] == label_next_action_ids[0]:
                # TRUE POSITIVE
                example_metrics[metrics.QUERY_MATCH] = F1.TP
            else:
                # FALSE NEGATIVE
                example_metrics[metrics.QUERY_MATCH] = F1.FN
        else:
            assert len(label_next_action_ids) == 0
            if len(predict_next_action_ids) == 0:
                # TRUE NEGATIVE
                example_metrics[metrics.QUERY_MATCH] = F1.TN
            else:
                # FALSE POSITIVE
                example_metrics[metrics.QUERY_MATCH] = F1.FP
    return example_metrics

def get_metrics(inputs: List[str], labels: List[str], predicts: List[str], meta_data: List[str], in_domain_services):
    """Calculate the DSTC8 metrics.

    Args:
        labels: The ground truth dataset represented as a list mapping dialogue
        id to the corresponding dialogue.
        dataset_hyp: The predictions in the same format as `dataset_ref`.
        in_domain_services: The set of services which are present in the training
        set.

    Returns:
        A dict mapping a metric collection name to a dict containing the values
        for various metrics. Each metric collection aggregates the metrics across
        a specific set of frames in the dialogues.
    """
    # Metrics can be aggregated in various ways, eg over all dialogues, only for
    # dialogues containing unseen services or for dialogues corresponding to a
    # single service. This aggregation is done through metric_collections, which
    # is a dict mapping a collection name to a dict, which maps a metric to a list
    # of values for that metric. Each value in this list is the value taken by
    # the metric on a frame.

    in_domain_services = [service_name.lower() for service_name in in_domain_services]

    # Store metrics for every frame for debugging.
    # metric_collections[domain_name][metric_key][dialogue_id][turn_id]
    metric_collections = collections.defaultdict(
        lambda : collections.defaultdict(
            lambda : collections.defaultdict(
                lambda : collections.defaultdict(
                    lambda : 1.0
                )
            )
        )
    )

    for input, label, predict, meta in zip(inputs, labels, predicts, meta_data):
        meta_parts = meta.split(meta_delimiter)
        dialogue_id = [meta_part for meta_part in meta_parts if meta_part.startswith("dialogue_id")][0].split(":")[1]
        turn_id = [meta_part for meta_part in meta_parts if meta_part.startswith("turn_id")][0].split(":")[1]
        frame_id = [meta_part for meta_part in meta_parts if meta_part.startswith("frame_id")][0].split(":")[1]
        frame_service = [meta_part for meta_part in meta_parts if meta_part.startswith("frame_domain")][0].split(":")[1].lower()
        frame_domain = frame_service.split("_")[0]
        
        # Calculate metrics for each frame in each user turn.
        frame_metric = get_per_example_metrics_score(input,label,predict,
                                      average_and_joint_goal_accuracy=True,
                                      next_action_query_f1=True,
                                      request_slots_f1=True,
                                      active_intent_accuracy=True)

        # Get the domain name of the service.
        domain_keys = [ALL_SERVICES, frame_service, frame_domain]
        if frame_service in in_domain_services:
            domain_keys.append(SEEN_SERVICES)
        else:
            domain_keys.append(UNSEEN_SERVICES)
        for domain_key in domain_keys:
            for metric_key, metric_value in frame_metric.items():
                if metric_value != metrics.NAN_VAL:
                    if issubclass(metric_value,F1):
                        if metric_collections[domain_key][
                            metric_key][dialogue_id][turn_id] != 1.0:
                            assert metric_collections[domain_key][
                            metric_key][dialogue_id][turn_id] == metric_value
                        metric_collections[domain_key][
                            metric_key][dialogue_id][turn_id] = metric_value
                    else:
                        metric_collections[domain_key][
                                metric_key][dialogue_id][turn_id] *= metric_value
    all_metric_aggregate = {}
    for domain_key, domain_metric_vals in metric_collections.items():
        domain_metric_aggregate = {}
        for metric_key, dialogue_dict in domain_metric_vals.items():
            metric_values = []
            if metric_key == metrics.QUERY_MATCH:
                for dialogue, turn_dict in dialogue_dict.items():
                    dialogue_query = []
                    for turn, metric_value in turn_dict.items(): 
                        dialogue_query.append(metric_value)    
                    dialogue_f1_score = compute_f1(dialogue_query)
                    metric_values.append(dialogue_f1_score.f1)
            else:
                for dialogue, turn_dict in dialogue_dict.items():
                    for turn, metric_value in turn_dict.items(): 
                        metric_values.append(metric_value)
            
            # Metrics are macro-averaged across all frames.
            if metric_values:
                domain_metric_aggregate[metric_key] = float(np.mean(metric_values))
            else:
                domain_metric_aggregate[metric_key] = metrics.NAN_VAL
        all_metric_aggregate[domain_key] = domain_metric_aggregate
    return all_metric_aggregate, metric_collections

def get_service_set(schema_path):
    """Get the set of all services present in a schema."""
    service_set = set()
    with open(schema_path) as f:
        schema = json.load(f)
        for service in schema:
            service_set.add(service["service_name"])
    return service_set
def get_in_domain_services(schema_path_1, schema_path_2):
    """Get the set of common services between two schemas."""
    return get_service_set(schema_path_1) & get_service_set(schema_path_2)

def get_evaluate_data():
    inputs = []
    outputs = []
    predicts = []
    meta_data = []
    with open(os.path.join(FLAGS.dstc8_data_dir, FLAGS.eval_set, f"{FLAGS.eval_set}.txt")) as f:
        for line in f:
            components = line.strip().split("\t\t")
            inputs.append(components[0])
            outputs.append(components[1])
            meta_data.append(components[2])
    predicts = outputs
    return inputs, outputs, predicts, meta_data

def main(_):    
    in_domain_services = get_in_domain_services(
        os.path.join(FLAGS.dstc8_data_dir, FLAGS.eval_set, "schema.json"),
        os.path.join(FLAGS.dstc8_data_dir, "train", "schema.json")
    )
    inputs, labels, predicts, meta_data = get_evaluate_data()
    all_metric_aggregate, _ = get_metrics(inputs, labels, predicts, meta_data,
                                        in_domain_services)
    
if __name__ == "__main__":
    flags.mark_flag_as_required("dstc8_data_dir")
    flags.mark_flag_as_required("eval_set")
    app.run(main)
logging.debug(compute_f1([F1.FN,F1.TP,F1.TN,F1.FP,F1.FN,F1.TP,F1.TN,F1.FP,F1.FN,F1.TP,F1.TN,F1.FP,F1.FN,F1.TP,F1.TN,F1.FP]))
# logging.debug(get_per_example_metrics_score(pseudo_input,pseudo_ref,pseudo_hyp))