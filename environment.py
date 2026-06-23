"""
Simplified environment for single-report inference.

Source: train/train.ipynb, Cell 7 (lines 853-1087)
Status: Slightly modified from notebook.
Why: Removed LazyReportLoader dependency and class weight computation (training-only).
     Simplified to accept a single report dict for inference use case.
     Core logic (reset, step, get_available_actions) preserved unchanged.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from agent import Action, TERMINAL_ACTIONS, ACTION_COSTS
from features import CAPEFeatureExtractor, StateBuilder


class CapeMalwareEnv:
    """
    Simplified environment for single-report inference.

    Enables step-by-step interaction with the trained agent on a single CAPE report,
    mimicking the sequential decision-making process of a malware analyst.
    """

    def __init__(self, report: Dict, label: str = "UNKNOWN", max_steps: int = 20):
        """
        Initialize environment with a single report.

        Args:
            report: CAPE v2 JSON report dict
            label: Ground truth label ("BENIGN", "MALWARE", or "UNKNOWN")
            max_steps: Maximum actions per episode (prevents infinite loops)
        """
        self.report = report
        self.label = label
        self.max_steps = max_steps

        # Episode state tracking
        self.revealed_actions = set()
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.action_history = []

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state (basic features only).

        Returns:
            Initial state vector (35 dimensions)
        """
        self.revealed_actions = {Action.CONTINUE}  # Basic features always available
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.action_history = []

        state = StateBuilder.build_state(
            self.report,
            self.revealed_actions,
            self.step_count,
            last_action=None
        )

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and transition to next state.

        Implements the core MDP transition function: s' = T(s, a)

        Args:
            action: Integer action index

        Returns:
            Tuple of (next_state, reward, done, info_dict)
        """
        if self.done:
            raise ValueError("Episode terminated. Call reset() to start new episode.")

        action_enum = Action(action)
        reward = 0.0
        info = {
            'label': self.label,
            'step': self.step_count,
        }

        self.action_history.append(action)

        # Apply step penalty (encourages efficiency)
        reward -= 0.1

        # Apply action-specific computational cost
        if action_enum in ACTION_COSTS:
            reward -= ACTION_COSTS[action_enum]

        # Terminal action handling: Classification decision
        if action_enum in TERMINAL_ACTIONS:
            self.done = True
            predicted_malware = (action_enum == Action.TERMINATE_MALWARE)
            actual_malware = (self.label == 'MALWARE')

            if predicted_malware == actual_malware:
                if not actual_malware:
                    reward += 15.0
                else:
                    reward += 15.0

                info['correct'] = True
                info['termination'] = 'correct'
            else:
                if not actual_malware:
                    reward -= 25.0
                else:
                    reward -= 25.0

                info['correct'] = False
                info['termination'] = 'incorrect'

            info['terminated'] = True
            info['prediction'] = 'MALWARE' if predicted_malware else 'BENIGN'

        else:
            # Non-terminal action: Information gathering
            if action_enum in self.revealed_actions:
                # Penalty for redundant actions
                reward -= 0.5
                info['repeated'] = True
            else:
                # Reveal new information
                self.revealed_actions.add(action_enum)
                reward += 0.2
                info['new_info'] = True

                # Additional reward for revealing informative features
                if action_enum == Action.MEMORY_DUMP:
                    features = CAPEFeatureExtractor.extract_memory_dump_features(self.report)
                    if features.get('n_anomalies', 0) > 0 or features.get('signatures_alert_count', 0) > 0:
                        reward += 1.0
                elif action_enum == Action.FOCUS_FILESYSTEM:
                    features = CAPEFeatureExtractor.extract_filesystem_features(self.report)
                    if features.get('n_dropped', 0) > 0:
                        reward += 0.3

            # Episode length limit
            self.step_count += 1
            if self.step_count >= self.max_steps:
                self.done = True
                reward -= 5.0
                info['max_steps'] = True
                info['terminated'] = False

        self.total_reward += reward

        next_state = StateBuilder.build_state(
            self.report,
            self.revealed_actions,
            self.step_count,
            last_action=action
        )

        info['total_reward'] = self.total_reward
        info['revealed_actions'] = list(self.revealed_actions)
        info['step_count'] = self.step_count

        return next_state, reward, self.done, info

    # def get_available_actions(self) -> List[int]:
    #     """
    #     Determine legal actions in current state (action masking).

    #     Returns:
    #         List of valid action indices
    #     """
    #     if self.done:
    #         return []

    #     available = list(Action)

    #     # Constraint: Require minimum exploration before terminal actions
    #     if self.step_count < 3:
    #         available = [a for a in available if a not in TERMINAL_ACTIONS]

    #     return available
    def get_available_actions(self) -> List[int]:
        if self.done:
            return []

        # 1. Start with actions that haven't been revealed yet
        available = [a for a in Action if a not in self.revealed_actions]

        # 2. If step count is less than 3, explicitly strip out terminal actions
        if self.step_count < 3:
            TERMINAL_ACTIONS = [Action.TERMINATE_BENIGN, Action.TERMINATE_MALWARE]
            available = [a for a in available if a not in TERMINAL_ACTIONS]

        return available
        
