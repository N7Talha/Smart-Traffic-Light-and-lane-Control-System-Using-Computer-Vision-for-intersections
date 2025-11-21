import argparse
import shutil
from pathlib import Path
import yaml
from datetime import datetime
from ultralytics import YOLO
import os
import torch


# ------------------------------------------------------------
# Utility: Ensure directory exists
# ------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------
# CORRECT CLASS REMAPPING FOR UDACTITY → 5 TARGET CLASSES
# ------------------------------------------------------------

def build_class_mapping(original_names, target_classes):
    """
    Build mapping from 11-class Udacity dataset to your custom 6-class taxonomy.
    Returns: dict {old_class_index → new_class_index or None}
    """

    # Assign indices dynamically based on target_classes list
    target_to_index = {name: idx for idx, name in enumerate(target_classes)}

    mapping = {}

    for idx, name in enumerate(original_names):
        n = name.lower()

        # --------------------------
        # CAR GROUP
        # --------------------------
        if n in ["car", "biker", "truck"]:
            mapping[idx] = target_to_index["car"]

        # --------------------------
        # PERSON GROUP
        # --------------------------
        elif n == "pedestrian":
            mapping[idx] = target_to_index["person"]

        # --------------------------
        # TRAFFIC LIGHT (generic)
        # --------------------------
        elif n == "trafficlight":
            mapping[idx] = target_to_index["traffic_light"]

        # --------------------------
        # TRAFFIC LIGHT — GREEN
        # --------------------------
        elif "green" in n:
            mapping[idx] = target_to_index["traffic_light_green"]

        # --------------------------
        # TRAFFIC LIGHT — RED
        # --------------------------
        elif "red" in n:
            mapping[idx] = target_to_index["traffic_light_red"]

        # --------------------------
        # TRAFFIC LIGHT — YELLOW
        # --------------------------
        elif "yellow" in n:
            mapping[idx] = target_to_index["traffic_light_yellow"]

        else:
            mapping[idx] = None  # Ignore class

    return mapping




# ------------------------------------------------------------
# CREATE FIXED DATASET (DO NOT MODIFY ORIGINAL)
# ------------------------------------------------------------
def create_fixed_dataset(dataset_root: Path, mapping: dict, fixed_root: Path):
    ensure_dir(fixed_root)

    stats = {"processed": 0, "total": 0, "class_counts": {}}

    for split in ["train", "val", "test"]:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"

        if not img_dir.exists():
            continue

        new_img_dir = fixed_root / split / "images"
        new_lbl_dir = fixed_root / split / "labels"

        ensure_dir(new_img_dir)
        ensure_dir(new_lbl_dir)

        # Copy/symlink images
        for img in img_dir.glob("*"):
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            dst = new_img_dir / img.name
            shutil.copyfile(img, dst)

        if not lbl_dir.exists():
            continue

        for lbl in lbl_dir.glob("*.txt"):
            stats["total"] += 1
            new_lines = []

            lines = lbl.read_text().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                old_cls = int(parts[0])
                if old_cls in mapping:
                    new_cls = mapping[old_cls]
                    parts[0] = str(new_cls)
                    new_lines.append(" ".join(parts))

                    stats["class_counts"][new_cls] = stats["class_counts"].get(new_cls, 0) + 1

            (new_lbl_dir / lbl.name).write_text("\n".join(new_lines))
            stats["processed"] += 1

    return stats


# ------------------------------------------------------------
# CREATE FIXED YAML FILE
# ------------------------------------------------------------
def write_fixed_yaml(original_yaml_path, fixed_yaml_path, fixed_root, target_classes):
    with open(original_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    data["path"] = str(fixed_root)
    data["train"] = str(fixed_root / "train" / "images")
    data["val"] = str(fixed_root / "val" / "images")
    data["test"] = str(fixed_root / "test" / "images")
    data["names"] = target_classes
    data["nc"] = len(target_classes)

    with open(fixed_yaml_path, "w") as f:
        yaml.dump(data, f)

    return fixed_yaml_path


# ------------------------------------------------------------
# MAIN TRAINING LOGIC
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    original_yaml = dataset_root / "data.yaml"

    print("Loading original data.yaml...")
    with open(original_yaml, "r") as f:
        data = yaml.safe_load(f)

    original_names = data["names"]

    print("\nOriginal dataset classes:")
    for i, c in enumerate(original_names):
        print(f"  {i}: {c}")

    # Your target class list
    target_classes = [
    "car",
    "person",
    "traffic_light",
    "traffic_light_green",
    "traffic_light_red",
    "traffic_light_yellow"
]

    # Build mapping
    mapping = build_class_mapping(original_names, target_classes)

    print("\nClass remapping (old → new):")
    for old, new in mapping.items():
       if new is None:
        print(f"  {old} -> None (ignored)")
    else:
        print(f"  {old} -> {new} ({target_classes[new]})")


    # Create fixed dataset
    fixed_root = Path(str(dataset_root) + "_fixed")
    print("\nCreating new cleaned dataset at:", fixed_root)

    stats = create_fixed_dataset(dataset_root, mapping, fixed_root)

    print("\nDataset processing stats:")
    print(stats)

    # Write new YAML file
    fixed_yaml = Path("udacity_fixed.yaml")
    write_fixed_yaml(original_yaml, fixed_yaml, fixed_root, target_classes)

    print("\nGenerated fixed YAML:", fixed_yaml)

    # Train YOLO11X
    print("\nLoading YOLO11X model...")
    model = YOLO("yolo11x.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model.train(
        data=str(fixed_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        optimizer="AdamW",
        name="udacity_yolo11x",
        cache=False
    )

    print("\nTraining complete! Check runs/train/udacity_yolo11x")


if __name__ == "__main__":
    main()
