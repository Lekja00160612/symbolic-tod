from typing import Dict
from enum import StrEnum
from collections import defaultdict
class PromptCategory(StrEnum):
    # In
    dependencies = "dependencies"
    constraints = "constraints"
    targetacts = "targetacts"
    conversation = "conversation"

    useracts = "useracts"
    sysacts = "sysacts"
    params = "params"

    # Out
    state = "state"
    history = "history"
    nextacts = "nextacts"

    # Special tags

prompts = {
    PromptCategory.dependencies: [
        f"[{PromptCategory.dependencies}]: list of actions x must appeared in the conversation before actions y can happen",
        f"[{PromptCategory.dependencies}] explains before system can do actions y, all actions x must existed in history dialogue",
        f"[{PromptCategory.dependencies}] has format x_1, x_2,... x_n -> y, indicating all x_i must happend before y",
    ],
    PromptCategory.constraints: [
        f"[{PromptCategory.constraints}]: behavioral advices that system must follow",
        f"[{PromptCategory.constraints}] are tips for the system to achieve goals",
    ],
    PromptCategory.targetacts: [
        f"[{PromptCategory.targetacts}]: actions system trying to achieve to fulfill user needs"
    ],
    PromptCategory.conversation: [
        f"[{PromptCategory.conversation}] includes dialogue turns between user and system",
    ],
    PromptCategory.useracts: [
        f"given the context, [{PromptCategory.useracts}] are possible user actions in the context",
        f"[{PromptCategory.useracts}]: possible user actions symbolized by [{PromptCategory.params}]",
        f"each action in [{PromptCategory.useracts}] is represent by <act_id>=<act_description>"
    ],
    PromptCategory.sysacts: [
        f"given the context, [{PromptCategory.sysacts}] are possible user actions in the context",
        f"[{PromptCategory.sysacts}]: possible system actions symbolized by [{PromptCategory.params}]",
        f"each action in [{PromptCategory.sysacts}] is represent by <act_id>=<act_description>"
    ],
    PromptCategory.params: [
        f"[{PromptCategory.params}]: context information that the system must capture in the [{PromptCategory.conversation}]",
        f"[{PromptCategory.params}] has format <param_id>=<param_description>"
    ],
    PromptCategory.state: [
        f"[{PromptCategory.state}]: includes all captured <param_value> in [{PromptCategory.conversation}]",
        f"[{PromptCategory.state}] is format as <param_id>=<param_value>"
    ],
    PromptCategory.history: [
        f"[{PromptCategory.history}] contains all actions appeared in [{PromptCategory.conversation}] and symbolized by [{PromptCategory.useracts}] and [{PromptCategory.sysacts}]"
        f"[{PromptCategory.history}] is presented as multiple <act_id>"
    ],
    PromptCategory.nextacts: [
        f"[{PromptCategory.nextacts}] includes actions system should response to user last turn in [{PromptCategory.conversation}]"
    ]
}

def serialize_prompt_to_file(data_output_path: str=None, file_name: str="prompt.txt"):
    with open(f"{data_output_path}{file_name}", mode="w", encoding="utf-8") as file:
        for key, key_prompts in prompts.items():
            for prompt in key_prompts:
                file.write(f"{key}\t{prompt}\n")

def get_prompts(data_path: str=None, file_name: str="prompt.txt") -> Dict[str,str]:
    prompts = defaultdict(lambda:list())
    with open(f"{data_path}{file_name}", encoding="utf-8") as file:
        for line in file:
            category, prompt = line.strip().split("\t")
            prompts[category].append(prompt)
        return prompts