"""
Action space definitions and cost structure for the MalWhere RL environment.

Source: train/train.ipynb, Cell 2 (lines 74-103)
Status: Copied unchanged from notebook.
Why: Action definitions must match training exactly for correct inference.
"""

from enum import IntEnum


class Action(IntEnum):
    """
    Enumeration of available actions in the reinforcement learning environment.

    Actions 0-4 represent investigative operations with incremental information revelation.
    Actions 5-6 are terminal classification decisions.
    """
    CONTINUE = 0          # Baseline analysis: Always-available metadata and basic features
    FOCUS_MEMORY = 1      # Memory analysis: Process events and injection indicators
    FOCUS_FILESYSTEM = 2  # Filesystem analysis: File operations and dropped artifacts
    FOCUS_NETWORK = 3     # Network analysis: DNS queries, HTTP requests, TCP/UDP connections
    MEMORY_DUMP = 4       # Deep memory inspection: Anomalies, encrypted buffers, payloads
    TERMINATE_MALWARE = 5 # Terminal action: Classify sample as MALWARE
    TERMINATE_BENIGN = 6  # Terminal action: Classify sample as BENIGN


NUM_ACTIONS = len(Action)
TERMINAL_ACTIONS = {Action.TERMINATE_MALWARE, Action.TERMINATE_BENIGN}

# Cost Structure: Computational penalties for different analysis operations
# Reflects real-world resource consumption in sandbox environments
ACTION_COSTS = {
    Action.CONTINUE: 0.1,
    Action.FOCUS_MEMORY: 0.5,
    Action.FOCUS_FILESYSTEM: 0.5,
    Action.FOCUS_NETWORK: 0.5,
    Action.MEMORY_DUMP: 1.0,  # Highest cost: Deep memory analysis
    Action.TERMINATE_MALWARE: 0.0,
    Action.TERMINATE_BENIGN: 0.0
}
