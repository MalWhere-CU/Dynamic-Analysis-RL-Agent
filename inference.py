"""
Inference application for the trained MalWhere DDDQN agent.

Source: train/train.ipynb, Cell 10 (lines 1448-1487) and Cell 12 (lines 1864-1897)
Status: Refactored from notebook.
Why: Extracted select_action (greedy policy only, no epsilon) from DQNAgent class.
     Model loading simplified to only load policy_net (no optimizer/replay buffer needed).
     Combined with report loading and environment interaction for standalone inference.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from model import DDDQN
from agent import Action, NUM_ACTIONS
from features import StateBuilder
from environment import CapeMalwareEnv


def load_model(model_path: str, device: Optional[torch.device] = None) -> DDDQN:
    """
    Load trained DDDQN model from checkpoint.

    Source: train/train.ipynb, Cell 10 DQNAgent.load() (lines 1580-1594)
    Modification: Only loads policy_net_state_dict. Skips optimizer, target_net,
    losses, epsilons (training-only artifacts).

    Args:
        model_path: Path to .pth checkpoint file
        device: Target device (auto-detected if None)

    Returns:
        Loaded DDDQN model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDDQN(
        dim_states=StateBuilder.TOTAL_FEATURE_DIM,
        dim_actions=NUM_ACTIONS,
        dim_hidden=256
    ).to(device)

    # weights_only=False required: checkpoint may contain numpy arrays
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle both full checkpoint and raw state_dict formats
    if 'policy_net_state_dict' in checkpoint:
        state_dict = checkpoint['policy_net_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded from {model_path} onto {device}")
    return model


def select_action(
    state: np.ndarray,
    available_actions: List[int],
    model: DDDQN,
    device: torch.device
) -> int:
    """
    Select action using greedy policy (no exploration) with legal action masking.

    Source: train/train.ipynb, Cell 10 DQNAgent.select_action() (lines 1474-1487)
    Modification: Removed epsilon-greedy exploration (training-only).
     Always uses greedy policy for inference.

    Args:
        state: Current state vector (35 dimensions)
        available_actions: List of legal action indices
        model: Trained DDDQN model
        device: Torch device for computation

    Returns:
        Selected action index
    """
    # Construct action mask for neural network
    action_mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    for action in available_actions:
        action_mask[action] = 1.0

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        mask_tensor = action_mask.unsqueeze(0).to(device)

        q_values = model(state_tensor, mask_tensor)
        action = q_values.argmax(dim=1).item()

        # Safety check: Ensure selected action is legal
        if action not in available_actions:
            action = available_actions[0]

    return action


def analyze_report(
    report_path: str,
    model: DDDQN,
    device: torch.device,
    label: str = "UNKNOWN",
    max_steps: int = 20,
    verbose: bool = True
) -> Dict:
    """
    Run full inference pipeline on a single CAPE report.

    Source: train/train.ipynb, Cell 12 evaluate_agent_simple() (lines 1864-1897)
    Modification: Adapted for single-report analysis with detailed output instead
     of batch evaluation metrics.

    Args:
        report_path: Path to CAPE v2 JSON report
        model: Trained DDDQN model
        device: Torch device
        label: Ground truth label ("BENIGN", "MALWARE", or "UNKNOWN")
        max_steps: Maximum agent steps before forced termination
        verbose: If True, print step-by-step analysis

    Returns:
        Dictionary with prediction, steps taken, actions trace, and info
    """
    # Load report
    with open(report_path, 'r') as f:
        report = json.load(f)

    # Create environment
    env = CapeMalwareEnv(report, label=label, max_steps=max_steps)

    # Run agent episode
    state = env.reset()
    done = False

    if verbose:
        print(f"\nAnalyzing: {Path(report_path).name}")
        print(f"True label: {label}")
        print("-" * 50)

    step = 0
    while not done:
        available_actions = env.get_available_actions()
        action = select_action(state, available_actions, model, device)

        next_state, reward, done, info = env.step(action)

        if verbose:
            action_name = Action(action).name
            print(f"  Step {step}: {action_name:25s} | Reward: {reward:+.2f}")

        state = next_state
        step += 1

    # Extract result
    prediction = info.get('prediction', 'UNKNOWN')
    correct = info.get('correct', None)

    result = {
        'file': Path(report_path).name,
        'true_label': label,
        'prediction': prediction,
        'correct': correct,
        'steps': step,
        'total_reward': info.get('total_reward', 0.0),
        'actions': [Action(a).name for a in env.action_history],
    }

    if verbose:
        print("-" * 50)
        status = "CORRECT" if correct else "INCORRECT" if correct is not None else "NO GROUND TRUTH"
        print(f"  Prediction: {prediction} | Status: {status}")
        print(f"  Steps: {step} | Total Reward: {info.get('total_reward', 0.0):.2f}")
        print(f"  Actions trace: {' -> '.join(result['actions'])}")
    return result


def analyze_directory(
    data_dir: str,
    model: DDDQN,
    device: torch.device,
    max_steps: int = 20
) -> List[Dict]:
    """
    Run inference on all JSON reports in a directory structure.

    Expects directory layout:
        data_dir/
            benign/*.json
            malware/*.json

    Args:
        data_dir: Root data directory
        model: Trained DDDQN model
        device: Torch device
        max_steps: Maximum agent steps

    Returns:
        List of result dictionaries
    """
    data_path = Path(data_dir)
    results = []

    for label_dir, label in [('benign', 'BENIGN'), ('malware', 'MALWARE')]:
        label_path = data_path / label_dir
        if not label_path.exists():
            print(f"Warning: {label_path} not found, skipping.")
            continue

        for report_file in sorted(label_path.glob('*.json')):
            result = analyze_report(
                str(report_file),
                model,
                device,
                label=label,
                max_steps=max_steps,
                verbose=True
            )
            results.append(result)
            print()

    # Summary
    if results:
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / total if total > 0 else 0.0

        print("=" * 50)
        print("BATCH INFERENCE SUMMARY")
        print("=" * 50)
        print(f"  Total reports: {total}")
        print(f"  Correct:       {correct}")
        print(f"  Accuracy:      {accuracy:.2%}")
        print(f"  Avg steps:     {np.mean([r['steps'] for r in results]):.1f}")
        print("=" * 50)

    return results


def main():
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(
        description="Inference for trained MalWhere DDDQN agent"
    )
    parser.add_argument(
        "--model", type=str, default="models/dqn_cape_best.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Path to a single CAPE JSON report for inference"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Root data directory with benign/ and malware/ subdirectories"
    )
    parser.add_argument(
        "--label", type=str, default="UNKNOWN",
        choices=["BENIGN", "MALWARE", "UNKNOWN"],
        help="Ground truth label for single report (ignored in batch mode)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=20,
        help="Maximum agent steps before forced termination"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (auto-detected if not specified)"
    )

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.model, device)

    # Single report inference
    if args.report:
        result = analyze_report(
            args.report, model, device,
            label=args.label, max_steps=args.max_steps
        )
        return result

    # Batch inference on data directory
    results = analyze_directory(
        args.data_dir, model, device,
        max_steps=args.max_steps
    )
    return results


if __name__ == "__main__":
    main()
