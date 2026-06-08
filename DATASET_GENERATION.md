# Procedural 3D Scene Dataset Generator

This guide covers generating the SolidCount synthetic dataset from scratch using Blender.

## Prerequisites

- **Blender 4.0+**: Ensure the Blender binary is accessible in your system `PATH`, or update the `BLENDER_BIN` variable in `generate_dataset.py`.
- **Python 3.8+**
- **Required packages**:
    ```bash
    pip install tqdm numpy
    ```
    *(Blender's internal Python environment should already have `numpy` installed.)*

## File Structure

| File | Description |
|------|-------------|
| `generate_dataset.py` | Python launcher script that creates JSON configs and spawns parallel Blender processes |
| `blender_scene_generator.py` | Background worker script executed by Blender to build scenes and render images |

---

## Usage

1. **Configure paths** in `generate_dataset.py` under the `=== CONFIGURATION & PATHS ===` header:
   - Set `BLENDER_BIN` to your Blender executable path
   - Adjust `MAX_WORKERS` based on your hardware (rendering is GPU/VRAM intensive)

2. **Run the generator:**
    ```bash
    python generate_dataset.py
    ```

3. **Check outputs:** Rendered images, JSON configs, and segmentation masks will be in `./output_dataset`.

---

## Running Models

After generating the dataset, run each model and save results as pickle files:

**VLMs** (Claude, Gemini, GPT, Qwen):
- Run with three prompt strategies: `estimate`, `label`, `locate`
- Save as: `{model}_result_{method}_{dataset}.pkl`

**Counting Models** (PseCo, T2ICount, TFOC):
- Save as: `{model}_{dataset}.pkl`

Place all pickle files in `data/results/` to use with `plots.py`.

---

## Customizing the Scene

### Shapes and Sizes

Modify the `shapes` list in `generate_dataset.py`:
```python
shapes = ["cube", "uv_sphere", "cylinder", "cone", "torus", "capsule", "ellipsoid", "pyramid"]
```

For randomized sizing, uncomment:
```python
# size = rng.uniform(min_size_fraction, max_size_fraction)
```

To add new shapes, add a condition block to `add_shape_primitive()` in `blender_scene_generator.py` using `bpy.ops.mesh.primitive_*` API.

### Colors

Colors are generated using random RGB or HSV values in `generate_dataset.py`:
```python
if rng.random() < 0.5:
    r, g, b = rng.random(), rng.random(), rng.random()
else:
    h = rng.random()
    s = rng.uniform(0.3, 1.0)
    v = rng.uniform(0.3, 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
```

### Background and Floor

Modify materials in `setup_background_and_trapezoid()` in `blender_scene_generator.py`:

```python
checker.inputs["Scale"].default_value = 18.0
col_a.outputs[0].default_value = (0.02, 0.02, 0.02, 1.0)  # Dark gray
col_b.outputs[0].default_value = (0.06, 0.06, 0.06, 1.0)  # Lighter gray
```

To use a solid color instead of checkerboard, connect a `ShaderNodeRGB` directly to the `Principled BSDF` Base Color input.
