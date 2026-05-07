"""
generate_dataset.py

Launcher script for procedurally generating 3D scenes using Blender.
This script generates JSON configurations for each scene, including random
shapes, colors, and sizes, and uses multiprocessing to execute Blender instances
in parallel to render the images and masks.

Author: Kuinan
Usage:
    python generate_dataset.py
"""

import os
import json
import subprocess
import random
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import colorsys
import re

# === CONFIGURATION & PATHS ===
# Provide the path to your Blender executable or just "blender" if it's in your PATH
BLENDER_BIN = "blender" 
BLENDER_SCRIPT = os.path.abspath("blender_scene_generator.py")

# Output Directories (Relative to the script location)
BASE_OUTPUT_DIR = "./output_dataset"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "Images")
OUTPUT_CONFIG_DIR = os.path.join(BASE_OUTPUT_DIR, "Configs")
MASK_DIR = os.path.join(BASE_OUTPUT_DIR, "Masks")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_CONFIG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# === SCENE PARAMETERS ===
shapes = ["cube", "uv_sphere", "cylinder", "cone", "torus", "capsule", "ellipsoid", "pyramid"]
n_repetitions = 50
attempts_per_image = 5
max_attempts_per_shape = 2000
base_seed = 12
image_resolution = [1080, 1080]
counts_list = list(range(40, 7, -1))

# Multiprocessing config
MAX_WORKERS = 8 # Adjust based on your GPU/CPU capacity (Blender is GPU-heavy!)

# Size bounds (as fraction of scene)
min_size_fraction = 0.08
max_size_fraction = 0.18

# === Helper: strip seed suffix ===
def strip_seed_suffix(name):
    return re.sub(r"_seed\d+$", "", name)

# === Single job function (run in subprocess) ===
def run_blender_job(job_config_path):
    cmd = [
        BLENDER_BIN,
        "--background",
        "--python", BLENDER_SCRIPT,
        "--", job_config_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stderr, job_config_path

# === Main ===
if __name__ == "__main__":
    # Collect existing jobs to resume if interrupted
    existing_file_bases = []
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".png"):
                existing_file_bases.append(os.path.splitext(f)[0])
    existing_jobs = {strip_seed_suffix(name) for name in existing_file_bases}

    total_jobs = len(counts_list) * n_repetitions
    pbar = tqdm(total=total_jobs, desc="Generating jobs")
    job_counter = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for target_count in counts_list:
            for rep in range(n_repetitions):
                job_base = f"N{target_count}_r{rep}"
                if job_base in existing_jobs:
                    pbar.update(1)
                    job_counter += 1
                    continue

                job_key = f"{target_count}_r{rep}"
                job_seed = base_seed + hash(job_key) % (2**32)
                rng = random.Random(job_seed)
                object_specs = []

                shape = rng.choice(shapes)
                # size = rng.uniform(min_size_fraction, max_size_fraction)
                size = 0.10 # Fixed size example
                
                for _ in range(target_count):
                    if rng.random() < 0.5:
                        r, g, b = rng.random(), rng.random(), rng.random()
                    else:
                        h = rng.random()
                        s = rng.uniform(0.3, 1.0)
                        v = rng.uniform(0.3, 1.0)
                        r, g, b = colorsys.hsv_to_rgb(h, s, v)

                    r, g, b = max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b))
                    color_rgba = [r, g, b, 1.0]

                    object_specs.append({
                        "shape": shape,
                        "color": color_rgba,
                        "size": size
                    })

                job_name = f"N{target_count}_r{rep}_seed{job_seed % 10000}"
                img_out = os.path.join(OUTPUT_DIR, f"{job_name}.png")
                cfg_out = os.path.join(OUTPUT_CONFIG_DIR, f"{job_name}.json")
                cfg_path = os.path.join(OUTPUT_CONFIG_DIR, f"{job_name}_input.json")

                # Create unique temp dir for this job
                job_temp_dir = tempfile.mkdtemp(prefix=f"blender_tmp_{job_name}_")

                job_cfg = {
                    "object_specs": object_specs,
                    "target_count": target_count,
                    "repetition_index": rep,
                    "seed": job_seed,
                    "attempts_per_image": attempts_per_image,
                    "max_attempts_per_shape": max_attempts_per_shape,
                    "output_image": img_out,
                    "output_config": cfg_out,
                    "bottom_width": 20,
                    "top_width": 10,
                    "height": 20,
                    "z_offset": 0.02,
                    "min_distance_world": 0.10,
                    "min_size_fraction": min_size_fraction,
                    "max_size_fraction": max_size_fraction,
                    "allowed_overlap_fraction": 0.01,
                    "image_resolution": image_resolution,
                    "job_name": job_name,
                    "temp_dir": job_temp_dir,  # critical for isolation
                    "mask_dir": MASK_DIR
                }

                with open(cfg_path, "w") as f:
                    json.dump(job_cfg, f)

                # Submit job
                future = executor.submit(run_blender_job, cfg_path)
                futures.append(future)

        # Collect results
        for future in as_completed(futures):
            returncode, stderr, cfg_path = future.result()
            if returncode != 0:
                print(f"\nJob failed (returncode {returncode}) for config: {cfg_path}")
                if stderr:
                    print("stderr snippet:", stderr[:500])
            
            # Clean up temp dir
            try:
                temp_dir = json.load(open(cfg_path))["temp_dir"]
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            
            pbar.update(1)

    pbar.close()
    print("All jobs completed.")