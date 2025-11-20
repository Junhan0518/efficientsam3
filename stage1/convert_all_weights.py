import argparse
import os
from pathlib import Path
import torch
from tqdm import tqdm

def _torch_load(path, map_location="cpu", **kwargs):
    try:
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, map_location=map_location, **kwargs)

def _load_state_dict(path: str):
    obj = _torch_load(path, map_location="cpu")
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ("model", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        tensors_only = all(isinstance(v, torch.Tensor) for v in obj.values())
        if tensors_only:
            return obj
    raise ValueError(f"Unable to extract a state_dict from {path}")

def merge_weights(student_sd, teacher_sd, target_prefix="image_encoder", replace_prefix=None, skip_prefixes=None):
    if skip_prefixes is None:
        skip_prefixes = []
        
    prefix = target_prefix.strip(".")
    prefix = f"{prefix}." if prefix else ""
    
    if replace_prefix is None:
        replace_prefix = target_prefix
        
    replace_prefix = replace_prefix.strip(".")
    replace_prefix = f"{replace_prefix}." if replace_prefix else ""
    
    skip_prefixes = [p.strip(".") + "." for p in skip_prefixes if p is not None]

    merged = {}
    # Copy student weights with prefix
    for key, value in student_sd.items():
        merged_key = f"{prefix}{key}" if prefix else key
        merged[merged_key] = value

    skipped = 0
    replaced = 0
    appended = 0
    
    # Copy teacher weights that are not replaced
    for key, value in teacher_sd.items():
        if replace_prefix and key.startswith(replace_prefix):
            replaced += 1
            continue
        if any(key.startswith(p) for p in skip_prefixes):
            skipped += 1
            continue
        if key in merged:
            # Prefer the student copy (shouldn't happen if prefixes match correctly)
            skipped += 1
            continue
        merged[key] = value
        appended += 1
        
    return merged, skipped, replaced, appended

def main():
    parser = argparse.ArgumentParser(description="Batch convert Stage-1 weights")
    parser.add_argument("--sam3-ckpt", type=str, required=True, help="Path to SAM3 teacher checkpoint")
    parser.add_argument("--student-dir", type=str, required=True, help="Directory containing student checkpoints")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save merged checkpoints")
    args = parser.parse_args()

    print(f"Loading teacher checkpoint from {args.sam3_ckpt}...")
    teacher_sd = _load_state_dict(args.sam3_ckpt)
    print("Teacher checkpoint loaded.")

    os.makedirs(args.output_dir, exist_ok=True)

    # List of expected config names
    configs = [
        "es_rv_s", "es_rv_m", "es_rv_l",
        "es_tv_s", "es_tv_m", "es_tv_l",
        "es_ev_s", "es_ev_m", "es_ev_l"
    ]

    for cfg in tqdm(configs):
        student_ckpt_path = os.path.join(args.student_dir, cfg, "ckpt_epoch_0.pth")
        if not os.path.exists(student_ckpt_path):
            print(f"Warning: Checkpoint not found for {cfg} at {student_ckpt_path}")
            continue
            
        print(f"Processing {cfg}...")
        student_sd = _load_state_dict(student_ckpt_path)
        
        merged, skipped, replaced, appended = merge_weights(
            student_sd, 
            teacher_sd, 
            target_prefix="image_encoder"
        )
        
        output_path = os.path.join(args.output_dir, f"{cfg}.pt")
        torch.save({"model": merged}, output_path)
        print(f"Saved {output_path} (Replaced: {replaced}, Appended: {appended})")

if __name__ == "__main__":
    main()
