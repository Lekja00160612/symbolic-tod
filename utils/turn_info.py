""" Contain all possible turn info """
from dataclasses import dataclass

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