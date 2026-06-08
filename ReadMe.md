# [Supplementary PDF](SI_VLM_Counting.pdf)

### Official implementation of ***Assessing the Visual Enumeration Abilities of Specialized Counting Architectures and Vision-Language Models*** (ICPR 2026)

---

## Reproducing the Analysis

The `data/` folder contains pre-computed model responses that allow you to reproduce all analysis plots without running the models.

### Environment Setup

```bash
conda activate base
pip install -r requirements.txt
```

Or create a dedicated environment:

```bash
conda create -n vlm_count python=3.10
conda activate vlm_count
pip install -r requirements.txt
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

---

## Data

| File | Size | Description |
|------|------|-------------|
| `data/annotation_FSC147_384_n_40.json` | 14MB | FSC-147 annotations (filtered to n<=40) |
| `data/FSCD_n40.json` | 0.9MB | FSCD-LVIS annotations (filtered to n<=40) |
| `data/results/*.pkl` | 95MB | Pre-computed responses from 7 models |

**Models included:** Claude, Gemini, GPT, Qwen, PseCo, T2ICount, TFOC

---

## Generating Raw Data

To generate the synthetic dataset from scratch and run models yourself, see [DATASET_GENERATION.md](DATASET_GENERATION.md).
