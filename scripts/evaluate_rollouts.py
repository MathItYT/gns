#!/usr/bin/env python3
"""Evaluate trained GNS checkpoints on valid/test trajectories.

The script loads one checkpoint directory or a whole model root, runs rollout
prediction on `valid.npz` and `test.npz`, stores per-example rollout pickles,
and writes VTU series for ParaView visualization.
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
from pyevtk.hl import pointsToVTK

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gns import data_loader
from gns import reading_utils
from gns.learned_simulator import LearnedSimulator


INPUT_SEQUENCE_LENGTH = 6
KINEMATIC_PARTICLE_ID = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/", help="Directory containing metadata.json and split npz files.")
    parser.add_argument(
        "--model_path",
        default="models/",
        help="Model directory or a single checkpoint file. If a directory root is given, all run subdirectories are evaluated.",
    )
    parser.add_argument(
        "--model_file",
        default="latest",
        help='Checkpoint file to load inside each run directory. Use "latest" to select the newest model-*.pt file.',
    )
    parser.add_argument(
        "--splits",
        default="valid,test",
        help="Comma-separated dataset splits to evaluate.",
    )
    parser.add_argument(
        "--output_root",
        default="evaluations",
        help="Directory where rollout pickles and VTU outputs are written.",
    )
    parser.add_argument(
        "--architecture",
        default=None,
        choices=[None, "gns", "sparse_egnn"],
        help="Fallback architecture when the checkpoint does not store its config.",
    )
    parser.add_argument(
        "--cuda_device_number",
        type=int,
        default=None,
        help="CUDA device index to use when running on GPU.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional limit on the number of trajectories per split.",
    )
    return parser.parse_args()


def discover_model_targets(model_path: str):
    path = Path(model_path)
    if path.is_file():
        return [path]

    checkpoint_files = sorted(path.glob("model-*.pt"))
    if checkpoint_files:
        return [path]

    run_dirs = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and list(child.glob("model-*.pt")):
            run_dirs.append(child)

    if not run_dirs:
        raise FileNotFoundError(f"No checkpoints found under {model_path}")

    return run_dirs


def latest_checkpoint(run_dir: Path) -> Path:
    model_files = []
    for candidate in run_dir.glob("model-*.pt"):
        match = re.search(r"model-(\d+)\.pt$", candidate.name)
        if match:
            model_files.append((int(match.group(1)), candidate))

    if not model_files:
        raise FileNotFoundError(f"No model-*.pt files found in {run_dir}")

    return max(model_files, key=lambda item: item[0])[1]


def infer_architecture(checkpoint_path: Path, checkpoint_payload=None, fallback=None):
    if fallback is not None:
        return fallback

    if isinstance(checkpoint_payload, dict):
        config = checkpoint_payload.get("config")
        if isinstance(config, dict) and config.get("architecture"):
            return config["architecture"]

    inferred_from_name = checkpoint_path.parent.name if checkpoint_path.is_file() else checkpoint_path.name
    if "sparse_egnn" in inferred_from_name:
        return "sparse_egnn"
    if "gns" in inferred_from_name:
        return "gns"

    raise ValueError(
        "Unable to infer architecture from checkpoint. Pass --architecture or use checkpoints saved with config metadata."
    )


def load_simulator(checkpoint_path: Path, metadata: dict, device: torch.device, architecture=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and "config" in checkpoint:
        config = dict(checkpoint["config"])
        config["device"] = device
        simulator = LearnedSimulator(**config)
        simulator.load_state_dict(checkpoint["state_dict"])
        return simulator

    arch = infer_architecture(checkpoint_path, checkpoint_payload=checkpoint, fallback=architecture)

    from gns.train import _get_simulator

    simulator = _get_simulator(metadata, 0.0, 0.0, arch, device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        simulator.load_state_dict(checkpoint["state_dict"])
    else:
        simulator.load_state_dict(checkpoint)
    return simulator


def rollout_example(simulator, metadata, features, device):
    positions = features[0].to(device)
    particle_type = features[1].to(device).long()

    if len(features) == 4:
        material_property = features[2].to(device)
        n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32, device=device)
    elif len(features) == 3:
        material_property = None
        n_particles_per_example = torch.tensor([int(features[2])], dtype=torch.int32, device=device)
    else:
        raise NotImplementedError(f"Unsupported trajectory feature layout: {len(features)} entries")

    if metadata.get("sequence_length") is not None:
        nsteps = int(metadata["sequence_length"]) - INPUT_SEQUENCE_LENGTH
    else:
        nsteps = positions.shape[1] - INPUT_SEQUENCE_LENGTH

    initial_positions = positions[:, :INPUT_SEQUENCE_LENGTH]
    ground_truth_positions = positions[:, INPUT_SEQUENCE_LENGTH:]
    current_positions = initial_positions
    predictions = []

    for step in range(nsteps):
        next_position = simulator.predict_positions(
            current_positions,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_type,
            material_property=material_property,
        )

        kinematic_mask = (particle_type == KINEMATIC_PARTICLE_ID).clone().detach().to(device)
        kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, current_positions.shape[-1])
        next_position_ground_truth = ground_truth_positions[:, step]
        next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)

        predictions.append(next_position)
        current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)

    predictions = torch.stack(predictions)
    ground_truth_positions = ground_truth_positions.permute(1, 0, 2)
    loss = (predictions - ground_truth_positions) ** 2

    output_dict = {
        "initial_positions": initial_positions.permute(1, 0, 2).cpu().numpy(),
        "predicted_rollout": predictions.cpu().numpy(),
        "ground_truth_rollout": ground_truth_positions.cpu().numpy(),
        "particle_types": particle_type.cpu().numpy(),
        "material_property": material_property.cpu().numpy() if material_property is not None else None,
        "metadata": metadata,
        "loss": loss.mean().detach().cpu(),
    }

    return output_dict, loss


def write_vtu_series(example_dir: Path, rollout_data: dict):
    dims = rollout_data["ground_truth_rollout"].shape[-1]
    particle_types = rollout_data["particle_types"].astype(np.int32)

    series_specs = {
        "ground_truth": rollout_data["ground_truth_rollout"],
        "predicted": rollout_data["predicted_rollout"],
    }

    for series_name, trajectory in series_specs.items():
        series_dir = example_dir / "vtk" / series_name
        series_dir.mkdir(parents=True, exist_ok=True)
        initial_position = rollout_data["initial_positions"][0]

        for step, coords in enumerate(trajectory):
            displacement = np.ascontiguousarray(np.linalg.norm(coords - initial_position, axis=1))
            x_coord = np.ascontiguousarray(coords[:, 0])
            y_coord = np.ascontiguousarray(coords[:, 1])
            z_coord = np.ascontiguousarray(np.zeros_like(coords[:, 1]) if dims == 2 else coords[:, 2])
            prefix = series_dir / f"step_{step:04d}"
            pointsToVTK(
                str(prefix),
                x_coord,
                y_coord,
                z_coord,
                data={
                    "displacement": displacement,
                    "particle_type": np.ascontiguousarray(particle_types),
                },
            )


def evaluate_split(simulator, metadata, data_path: Path, split: str, device: torch.device, output_root: Path, max_examples=None):
    dataset_file = data_path / f"{split}.npz"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Missing dataset file: {dataset_file}")

    loader = data_loader.get_data_loader_by_trajectories(path=str(dataset_file))
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    losses = []
    example_summaries = []

    with torch.no_grad():
        for example_index, features in enumerate(loader):
            if max_examples is not None and example_index >= max_examples:
                break

            rollout_data, loss = rollout_example(simulator, metadata, features, device)
            losses.append(loss.flatten())

            example_dir = split_dir / f"example_{example_index:04d}"
            example_dir.mkdir(parents=True, exist_ok=True)

            pkl_path = example_dir / "rollout.pkl"
            with open(pkl_path, "wb") as handle:
                pickle.dump(rollout_data, handle)

            write_vtu_series(example_dir, rollout_data)

            example_loss = float(rollout_data["loss"].item())
            example_summaries.append({"example": example_index, "loss": example_loss, "path": str(example_dir)})
            print(f"[{split}] example {example_index}: loss={example_loss:.6e}")

    mean_loss = float(torch.mean(torch.cat(losses)).item()) if losses else float("nan")
    summary = {"split": split, "mean_loss": mean_loss, "examples": example_summaries}

    with open(split_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[{split}] mean loss: {mean_loss:.6e}")
    return summary


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    output_root = Path(args.output_root)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.cuda_device_number is not None:
        device = torch.device(f"cuda:{args.cuda_device_number}")

    metadata = reading_utils.read_metadata(str(data_path), "rollout")
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    targets = discover_model_targets(args.model_path)

    output_root.mkdir(parents=True, exist_ok=True)

    for target in targets:
        checkpoint_path = latest_checkpoint(target) if target.is_dir() else target
        run_name = target.stem if target.is_file() else target.name
        run_output_root = output_root / run_name
        run_output_root.mkdir(parents=True, exist_ok=True)

        simulator = load_simulator(checkpoint_path, metadata, device, architecture=args.architecture)
        simulator.to(device)
        simulator.eval()

        print(f"Evaluating {checkpoint_path} on {device}")
        run_summary = {"run_name": run_name, "checkpoint": str(checkpoint_path), "device": str(device), "splits": {}}

        for split in splits:
            split_summary = evaluate_split(
                simulator=simulator,
                metadata=metadata,
                data_path=data_path,
                split=split,
                device=device,
                output_root=run_output_root,
                max_examples=args.max_examples,
            )
            run_summary["splits"][split] = split_summary

        with open(run_output_root / "run_summary.json", "w", encoding="utf-8") as handle:
            json.dump(run_summary, handle, indent=2)


if __name__ == "__main__":
    main()