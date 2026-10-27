"""Data generation script: procedural scenes, camera views, and labelled grasps."""

import argparse
import os
import sys
import yaml

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grasp.generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic grasp dataset.")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to config YAML")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for .npz files")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    d_cfg = config.get("grasp", {}).get("dataset", {})
    output_dir = args.output_dir or d_cfg.get("save_dir", "data")
    
    num_scenes = d_cfg.get("num_scenes", 8)
    views_per_scene = d_cfg.get("views_per_scene", 2)
    grasps_per_view = d_cfg.get("grasps_per_view", 16)
    seed = config.get("seed", 42)

    os.makedirs(output_dir, exist_ok=True)
    generate_dataset(
        config=config,
        output_dir=output_dir,
        num_scenes=num_scenes,
        views_per_scene=views_per_scene,
        grasps_per_view=grasps_per_view,
        seed=seed
    )
    print(f"Data generation complete. Files saved to {output_dir}")


if __name__ == "__main__":
    main()
