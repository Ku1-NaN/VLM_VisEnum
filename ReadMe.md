# [Supplementary PDF](SI_VLM_Counting.pdf)

### Official implementation of ***Assessing the Visual Enumeration Abilities of Specialized Counting Architectures and Vision-Language Models*** (ICPR 2026)

---

## Reproducing the Analysis

The `data/` folder contains pre-computed model responses that allow you to reproduce all analysis plots without running the models.

### Environment Setup

```bash
conda activate base
pip install matplotlib seaborn pandas scipy scikit-learn pydantic
```

Or create a dedicated environment:

```bash
conda create -n vlm_count python=3.10
conda activate vlm_count
pip install matplotlib seaborn pandas scipy scikit-learn pydantic
```

### Generate Plots

```bash
python plots.py              # Generate all 3 plots
python plots.py --list       # List available plots
python plots.py --plot 1     # Generate specific plot
python plots.py --plot 1 2   # Generate multiple plots
python plots.py --no-stats   # Suppress statistical analysis output
```

**Output** (saved to `plots/` folder):

| Plot | File | Description |
|------|------|-------------|
| 1 | `Combined_Effects_Analysis.pdf` | Background, shape, and color effects on counting accuracy/MAE |
| 2 | `confusion_matrices.pdf` | Prediction vs Truth heatmaps across datasets and model types |
| 3 | `accuracy_by_prompt_method.pdf` | VLM accuracy comparison across prompt strategies |

### Data Contents

| File | Size | Description |
|------|------|-------------|
| `data/annotation_FSC147_384_n_40.json` | 14MB | FSC-147 annotations (filtered to n<=40) |
| `data/FSCD_n40.json` | 0.9MB | FSCD-LVIS annotations (filtered to n<=40) |
| `data/results/*.pkl` | 95MB | Pre-computed responses from 7 models |

**Models included:** Claude, Gemini, GPT, Qwen, PseCo, T2ICount, TFOC

---

## Generating Raw Data

If you want to generate the full dataset from scratch (rather than using the pre-computed results), follow the steps below.

### Prerequisites

- **Blender 4.0+**: Ensure the Blender binary is accessible in your system `PATH`, or update the `BLENDER_BIN` variable in `generate_dataset.py`.
- **Python 3.8+**
- **Required packages**:
    ```bash
    pip install tqdm numpy
    ```

### Dataset Generation

1. **Configure paths** in `generate_dataset.py` under the `=== CONFIGURATION & PATHS ===` header.
2. **Run the generator:**
    ```bash
    python generate_dataset.py
    ```
3. **Check outputs:** Rendered images, JSON configs, and segmentation masks will be in `./output_dataset`.

### Running Models

After generating the dataset, you need to:

1. Run each VLM (Claude, Gemini, GPT, Qwen) with the three prompt strategies (estimate, label, locate)
2. Run counting models (PseCo, T2ICount, TFOC) 
3. Save results as pickle files in `data/results/`

The pickle files should follow the naming convention:
- `{model}_result_{method}_{dataset}.pkl` for VLMs
- `{model}_{dataset}.pkl` for counting models

---

## Customizing the Scene Generator

### Shapes and Sizes

Modify the `shapes` list in `generate_dataset.py`:
```python
shapes = ["cube", "uv_sphere", "cylinder", "cone", "torus", "capsule", "ellipsoid", "pyramid"]
```

For randomized sizing, uncomment:
```python
# size = rng.uniform(min_size_fraction, max_size_fraction)
```

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
