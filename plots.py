#!/usr/bin/env python3
"""
Reproduce analysis plots from pre-computed model responses.

Usage:
    python plots.py              # Generate all plots
    python plots.py --plot 1     # Generate specific plot (1, 2, or 3)
    python plots.py --list       # List available plots
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from scipy import stats


# Pydantic models needed for unpickling GPT results
class BoundingBox(BaseModel):
    box_2d: List[int]
    label: str

class Locate(BaseModel):
    detections: list[BoundingBox]

class LabelResult(BaseModel):
    count: int
    labels: List[str]

class EstimateResult(BaseModel):
    estimation: int

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = DATA_DIR / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

DISPLAY_ORDER = ['Claude', 'Gemini', 'GPT', 'Qwen', 'PseCo', 'T2ICount', 'TFOC']

MODEL_NAME_MAP = {
    'gemini': 'Gemini',
    'claude': 'Claude',
    'gpt5': 'GPT',
    'gpt': 'GPT',
    'qwen': 'Qwen',
    'pseco': 'PseCo',
    't2icount': 'T2ICount',
    'trainfree': 'TFOC'
}

DATASET_NORMALIZATION = {
    'allcolornew': 'Allcolornew',
    'allcolornewbg': 'AllcolornewBG',
    'alldistinctcolor': 'Alldistinctcolor',
    'alldistinctcolorbg': 'AlldistinctcolorBG',
    'fixshapecolornew': 'FIXshapecolornew',
    'fixshapecolornewbg': 'FIXshapecolornewBG',
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_annotations():
    """Load FSC-147 and FSCD-LVIS annotations."""
    with open(DATA_DIR / "annotation_FSC147_384_n_40.json", 'r') as f:
        fsc_annotations = json.load(f)

    with open(DATA_DIR / "FSCD_n40.json", 'r') as f:
        fscd_annotations = json.load(f)

    fn_map = {ann['file_name']: len(ann['points']) for ann in fscd_annotations['annotations']}

    return fsc_annotations, fn_map


def parse_filename(pkl_name):
    """Parse model, method, and dataset from pickle filename."""
    parts = pkl_name.replace('.pkl', '').split('_')
    model_candidate = parts[0].lower()

    if model_candidate in ['pseco', 't2icount', 'trainfree']:
        model = model_candidate
        method = ''
        dataset_str = '_'.join(parts[1:])

        if 'fsc147' in dataset_str.lower():
            dataset = 'FSC147'
        elif 'fscd' in dataset_str.lower():
            dataset = 'FSCD'
        else:
            dataset = dataset_str
            if dataset_str.lower() == 'allcolornew':
                dataset = 'Allcolornew'
            elif dataset_str.lower() == 'allcolornewbg':
                dataset = 'AllcolornewBG'
    else:
        if len(parts) < 3:
            return None, None, None
        model = parts[0]
        method = parts[2]

        if 'fsc147' in pkl_name.lower():
            dataset = 'FSC147'
        elif 'fscd' in pkl_name.lower():
            dataset = 'FSCD'
        else:
            dataset = parts[-1]

    return model, method, dataset


def extract_prediction(response, model, method):
    """Extract prediction count from model response."""
    pred = 0

    if model in ['pseco', 't2icount', 'trainfree']:
        if isinstance(response, dict):
            pred = response.get('pred_count', response.get('count', 0))
        elif isinstance(response, (int, float)):
            pred = response
        elif isinstance(response, list) and len(response) > 0:
            pred = response[0]
    elif 'gemini' in model or 'claude' in model:
        if isinstance(response, dict):
            if method == 'estimate':
                pred = response.get('estimation', 0)
            elif method == 'label':
                pred = response.get('count', 0)
            else:
                det = response.get('detections', [])
                pred = len(det) if isinstance(det, list) else 0
    elif 'gpt' in model:
        if hasattr(response, 'estimation') and method == 'estimate':
            pred = response.estimation
        elif hasattr(response, 'count') and method == 'label':
            pred = response.count
        elif hasattr(response, 'detections'):
            pred = len(response.detections)
        elif isinstance(response, dict):
            pred = response.get('estimation', 0) if method == 'estimate' else response.get('count', 0)
    elif 'qwen' in model:
        pred = response if isinstance(response, (int, float)) else 0
    else:
        pred = response if isinstance(response, (int, float)) else 0

    return 0 if pred > 1000 else pred


def load_results():
    """Load all results and build DataFrame."""
    fsc_annotations, fn_map = load_annotations()
    all_results = []

    print("Loading results...")

    for pkl in os.listdir(RESULTS_DIR):
        if not pkl.endswith('.pkl'):
            continue

        model, method, dataset = parse_filename(pkl)
        if model is None:
            continue

        try:
            with open(RESULTS_DIR / pkl, 'rb') as f:
                result = pickle.load(f)
        except Exception as e:
            print(f"Error loading {pkl}: {e}")
            continue

        for im, response in result.items():
            if dataset == 'FSC147':
                truth = len(fsc_annotations[im]['annotations']) if im in fsc_annotations else 0
            elif dataset == 'FSCD':
                truth = fn_map.get(im, 0)
            else:
                try:
                    truth = int(im.split('_')[0].replace('N', ''))
                except:
                    truth = 0

            pred = extract_prediction(response, model, method)

            all_results.append({
                'MODEL': model,
                'METHOD': method,
                'DATASET': dataset,
                'ImageName': im,
                'Truth': truth,
                'Prediction': pred,
                'MAE': abs(pred - truth),
                'RMSE': (pred - truth) ** 2,
                'NAE': abs(pred - truth) / truth if truth != 0 else np.nan,
                'SRE': ((pred - truth) ** 2) / truth if truth != 0 else np.nan,
                'Accuracy': int(round(pred) == truth),
            })

    df = pd.DataFrame(all_results)
    df['DATASET'] = df['DATASET'].replace(DATASET_NORMALIZATION)

    print(f"Loaded {len(df)} records from {df['MODEL'].nunique()} models")

    return df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_model_name(name):
    """Standardize model name for display."""
    name_lower = str(name).lower()
    if 'pseco' in name_lower: return 'PseCo'
    if 't2i' in name_lower: return 'T2ICount'
    if 'trainfree' in name_lower: return 'TFOC'
    if 'claude' in name_lower: return 'Claude'
    if 'gemini' in name_lower: return 'Gemini'
    if 'gpt' in name_lower: return 'GPT'
    if 'qwen' in name_lower: return 'Qwen'
    return None


def get_dataset_group(dataset_name):
    """Map dataset name to group."""
    if dataset_name in ['FSC147', 'FSCD']:
        return dataset_name
    return 'SolidCount'


def get_dataset_category(ds_name):
    """Map dataset name to display category."""
    ds_name = ds_name.lower()
    if 'fsc147' in ds_name: return 'FSC-147'
    if 'fscd' in ds_name: return 'FSCD-LVIS'
    return 'SolidCount'


# ============================================================================
# PLOT 1: Combined Effects Analysis (Background, Shape, Color)
# ============================================================================

def plot_combined_effects(df, output_path):
    """Generate combined effects analysis plot."""
    print("Generating Plot 1: Combined Effects Analysis...")

    df_clean = df.copy()
    df_clean['Display_Model'] = df_clean['MODEL'].apply(format_model_name)
    df_clean = df_clean.dropna(subset=['Display_Model'])

    df_syn = df_clean[~df_clean['DATASET'].str.contains('FSC|FSCD', case=False, na=False)].copy()

    # Background Effect
    df_bg = df_syn.copy()
    df_bg['Condition'] = df_bg['DATASET'].apply(lambda x: 'Gray Fabric' if 'BG' in x else 'Checkerboard')

    # Color Effect
    df_color = df_syn.copy()
    df_color['Condition'] = df_color['DATASET'].apply(
        lambda x: 'Hetero Color' if 'distinct' in str(x).lower() else 'Homo Color'
    )

    # Shape Effect
    def classify_shape(row):
        ds = str(row['DATASET']).lower()
        if 'all' in ds: return 'Hetero Shape'
        elif 'fixshapesize' in ds: return 'Homo Shape'
        return None

    df_shape = df_syn.copy()
    df_shape['Condition'] = df_shape.apply(classify_shape, axis=1)
    df_shape = df_shape.dropna(subset=['Condition'])

    def aggregate(data):
        agg = data.groupby(['Display_Model', 'Condition'])[['Accuracy', 'MAE']].mean().reset_index()
        agg['Display_Model'] = pd.Categorical(agg['Display_Model'], categories=DISPLAY_ORDER, ordered=True)
        return agg.sort_values('Display_Model')

    agg_bg = aggregate(df_bg)
    agg_color = aggregate(df_color)
    agg_shape = aggregate(df_shape)

    colors = {
        'Checkerboard': '#87CEFA', 'Gray Fabric': '#A9A9A9',
        'Hetero Shape': '#e74c3c', 'Homo Shape': "#e6727bcd",
        'Hetero Color': "#591574", 'Homo Color': "#d155e4c2"
    }

    pairs = [
        ('Background', ['Checkerboard', 'Gray Fabric'], agg_bg),
        ('Shape', ['Hetero Shape', 'Homo Shape'], agg_shape),
        ('Color', ['Hetero Color', 'Homo Color'], agg_color)
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    plt.subplots_adjust(hspace=0.05)

    bar_width = 0.12
    group_gap = 0.15
    offsets = [
        (-2.0 * bar_width - group_gap, -1.0 * bar_width - group_gap),
        (-0.5 * bar_width, 0.5 * bar_width),
        (1.0 * bar_width + group_gap, 2.0 * bar_width + group_gap)
    ]

    x_indices = np.arange(len(DISPLAY_ORDER))
    metrics = [('Accuracy', axes[0], '.2f'), ('MAE', axes[1], '.1f')]

    for metric, ax, fmt in metrics:
        for i, (group_name, conditions, df_agg) in enumerate(pairs):
            for j, cond in enumerate(conditions):
                vals = []
                for m in DISPLAY_ORDER:
                    row = df_agg[(df_agg['Display_Model'] == m) & (df_agg['Condition'] == cond)]
                    vals.append(row[metric].values[0] if not row.empty else 0)

                pos = x_indices + offsets[i][j]
                rects = ax.bar(pos, vals, bar_width,
                              label=cond if metric == 'Accuracy' else "",
                              color=colors[cond], edgecolor='black', alpha=0.9)

                labels = [f"{v:{fmt}}" if v > 0 else "" for v in vals]
                ax.bar_label(rects, labels=labels, padding=2, fontsize=11, rotation=90)

        ax.set_ylabel(metric, fontsize=15, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        for x in x_indices[:-1]:
            ax.axvline(x + 0.5, color='gray', linestyle=':', alpha=0.5)

    for ax in axes:
        ax.set_xticks(x_indices)
        ax.set_xticklabels(DISPLAY_ORDER, fontsize=14, fontweight='bold')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.23)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels_leg, loc='upper right', ncol=3, fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


# ============================================================================
# PLOT 2: Confusion Matrix Heatmaps
# ============================================================================

def plot_confusion_matrices(df, output_path):
    """Generate confusion matrix heatmaps."""
    print("Generating Plot 2: Confusion Matrix Heatmaps...")

    df_viz = df.copy()
    df_viz['Dataset_Group'] = df_viz['DATASET'].apply(get_dataset_group)

    counting_models = ['pseco', 't2icount', 'trainfree']

    def get_category(row):
        if row['MODEL'] in counting_models: return 'Counting'
        if row['METHOD'] == 'estimate': return 'VLM_Estimate'
        if row['METHOD'] == 'locate': return 'VLM_Locate'
        if row['METHOD'] == 'label': return 'VLM_Label'
        return 'Other'

    df_viz['Category'] = df_viz.apply(get_category, axis=1)

    rows_config = [
        {'label': 'Counting Models', 'category': 'Counting', 'model_filter': None},
        {'label': 'Naive Count (Gemini)', 'category': 'VLM_Estimate', 'model_filter': 'gemini'},
        {'label': 'Point & Count (Gemini)', 'category': 'VLM_Locate', 'model_filter': 'gemini'},
        {'label': 'Label & Count (Claude)', 'category': 'VLM_Label', 'model_filter': 'claude'},
        {'label': 'Point & Count (GPT)', 'category': 'VLM_Locate', 'model_filter': 'gpt5'}
    ]

    cols_config = [
        {'name': 'FSC147', 'range': (8, 40)},
        {'name': 'FSCD', 'range': (21, 40)},
        {'name': 'SolidCount', 'range': (8, 40)}
    ]

    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 20))

    for r_idx, row_cfg in enumerate(rows_config):
        for c_idx, col_cfg in enumerate(cols_config):
            ax = axes[r_idx, c_idx]

            d_group = col_cfg['name']
            cat = row_cfg['category']
            t_min, t_max = col_cfg['range']

            candidates = df_viz[(df_viz['Dataset_Group'] == d_group) & (df_viz['Category'] == cat)]

            target_model = row_cfg.get('model_filter')
            if target_model:
                candidates = candidates[candidates['MODEL'] == target_model]

            if candidates.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            best_perf = candidates.groupby(['MODEL', 'METHOD'])['MAE'].mean()
            if best_perf.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue

            best_model, best_method = best_perf.idxmin()
            subset = candidates[(candidates['MODEL'] == best_model) & (candidates['METHOD'] == best_method)].copy()
            subset = subset[(subset['Truth'] >= t_min) & (subset['Truth'] <= t_max)]

            y_dim = 41
            x_dim = t_max - t_min + 1
            matrix = np.zeros((y_dim, x_dim))

            if not subset.empty:
                preds = subset['Prediction'].round().astype(int).values
                truths = subset['Truth'].values
                x_indices_cm = truths - t_min
                y_indices_cm = np.clip(preds - 1, 0, 40)

                for x, y in zip(x_indices_cm, y_indices_cm):
                    if 0 <= x < x_dim:
                        matrix[y, x] += 1

            ax.imshow(matrix, origin='lower', cmap='viridis', aspect='auto', interpolation='nearest')

            ax.set_yticks([0, 9, 19, 29, 40])
            ax.set_yticklabels(['1', '10', '20', '30', '40+'])

            x_tick_vals = list(range(t_min, t_max + 1, 4))
            if t_max not in x_tick_vals:
                x_tick_vals.append(t_max)
            ax.set_xticks([v - t_min for v in x_tick_vals])
            ax.set_xticklabels([str(v) for v in x_tick_vals])

            name_map = {'FSC147': 'FSC-147', 'FSCD': 'FSCD-LVIS', 'SolidCount': 'SolidCount'}

            if r_idx == 0:
                ax.set_title(name_map[d_group], fontsize=16, fontweight='bold', pad=10)
            if r_idx == 4:
                ax.set_xlabel("Truth", fontsize=12, fontweight='bold')
            if c_idx == 0:
                ax.set_ylabel("Prediction", fontsize=12, fontweight='bold')
                ax.text(-0.35, 0.5, row_cfg['label'], transform=ax.transAxes,
                       rotation=90, va='center', ha='center', fontsize=14, fontweight='bold')

            model_display = MODEL_NAME_MAP.get(best_model, best_model)
            ax.text(0.02, 0.95, model_display, transform=ax.transAxes,
                   color='white', fontsize=10, fontweight='bold', va='top',
                   bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

            line_x = np.arange(0, x_dim)
            line_y = line_x + t_min - 1
            valid_line = (line_y >= 0) & (line_y <= 39)
            ax.plot(line_x[valid_line], line_y[valid_line], 'r-', alpha=0.6, linewidth=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


# ============================================================================
# PLOT 3: Accuracy by Prompt Method
# ============================================================================

def calculate_eta_squared(groups):
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_total = np.sum((all_data - grand_mean)**2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    return ss_between / ss_total if ss_total != 0 else 0


def calculate_epsilon_squared(H, k, n):
    if n <= k: return 0
    return (H - k + 1) / (n - k)


def interpret_effect(d, test_type='anova'):
    if test_type == 'anova':
        if d < 0.01: return "Negligible"
        if d < 0.06: return "Small"
        if d < 0.14: return "Medium"
        return "Large"
    else:
        if d < 0.01: return "Negligible"
        if d < 0.08: return "Small"
        if d < 0.26: return "Medium"
        return "Large"


def plot_prompt_method_accuracy(df, output_path, print_stats=True):
    """Generate accuracy by prompt method comparison."""
    print("Generating Plot 3: Accuracy by Prompt Method...")

    target_models = ['claude', 'gemini', 'gpt']
    df_analysis = df[df['MODEL'].apply(lambda x: any(m in x.lower() for m in target_models))].copy()

    method_map = {
        'estimate': 'Naive Count',
        'label': 'Label and Count',
        'locate': 'Point, Label and Count'
    }
    df_analysis['METHOD'] = df_analysis['METHOD'].replace(method_map)

    def get_model_family(model_name):
        model_name = model_name.lower()
        if 'claude' in model_name: return 'Claude'
        if 'gemini' in model_name: return 'Gemini'
        if 'gpt' in model_name: return 'GPT'
        return model_name

    df_analysis['Model_Family'] = df_analysis['MODEL'].apply(get_model_family)
    df_analysis['Dataset_Category'] = df_analysis['DATASET'].apply(get_dataset_category)

    if print_stats:
        data_splits = {
            'SolidCount': df_analysis[df_analysis['Dataset_Category'] == 'SolidCount'],
            'FSC-147': df_analysis[df_analysis['Dataset_Category'] == 'FSC-147'],
            'FSCD-LVIS': df_analysis[df_analysis['Dataset_Category'] == 'FSCD-LVIS']
        }

        print(f"\n{'='*70}")
        print(f"{'STATISTICAL ANALYSIS: Method Impact by Dataset':^70}")
        print(f"{'='*70}")

        for dataset_name, data_subset in data_splits.items():
            if len(data_subset) == 0:
                continue
            print(f"\n>>> {dataset_name} (n={len(data_subset)})")

            for model_key in target_models:
                model_data = data_subset[data_subset['MODEL'].str.contains(model_key, case=False)]
                if len(model_data) == 0:
                    continue

                methods_found = model_data['METHOD'].unique()
                groups_mae, groups_acc, method_names = [], [], []

                for m in methods_found:
                    subset_method = model_data[model_data['METHOD'] == m]
                    if len(subset_method) > 1:
                        groups_mae.append(subset_method['MAE'].values)
                        groups_acc.append(subset_method['Accuracy'].values)
                        method_names.append(m)

                if len(method_names) < 2:
                    continue

                try:
                    _, p_mae = stats.f_oneway(*groups_mae)
                    eta_sq = calculate_eta_squared(groups_mae)
                    eff_mae = interpret_effect(eta_sq, 'anova')
                except:
                    p_mae, eta_sq, eff_mae = 1.0, 0, "Error"

                try:
                    h_stat, p_acc = stats.kruskal(*groups_acc)
                    n_total = sum(len(g) for g in groups_acc)
                    epsilon_sq = calculate_epsilon_squared(h_stat, len(groups_acc), n_total)
                    eff_acc = interpret_effect(epsilon_sq, 'kruskal')
                except:
                    p_acc, epsilon_sq, eff_acc = 1.0, 0, "Error"

                print(f"  [{model_key.upper()}] MAE p={p_mae:.3f} ({eff_mae}), ACC p={p_acc:.3f} ({eff_acc})")

    method_order = ['Naive Count', 'Label and Count', 'Point, Label and Count']
    dataset_order = ['FSC-147', 'FSCD-LVIS', 'SolidCount']
    model_families = ['Claude', 'Gemini', 'GPT']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for i, model in enumerate(model_families):
        subset = df_analysis[df_analysis['Model_Family'] == model]

        sns.barplot(
            data=subset,
            x='Dataset_Category',
            y='Accuracy',
            hue='METHOD',
            hue_order=method_order,
            order=dataset_order,
            ax=axes[i],
            palette='viridis',
            capsize=0.05,
            errorbar=('ci', 95)
        )

        axes[i].set_title(f'{model}', fontsize=18)
        axes[i].set_xlabel('', fontsize=14)
        axes[i].set_ylabel('Accuracy' if i == 0 else '', fontsize=15)
        axes[i].tick_params(axis='both', which='major', labelsize=14)

        if i == 2:
            axes[i].legend(title='Prompt Method', loc='best', fontsize=13)
        else:
            axes[i].get_legend().remove()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

PLOTS = {
    1: ("Combined_Effects_Analysis.pdf", plot_combined_effects,
        "Background, Shape, and Color effects on counting accuracy/MAE"),
    2: ("confusion_matrices.pdf", plot_confusion_matrices,
        "Prediction vs Truth heatmaps across datasets and model types"),
    3: ("accuracy_by_prompt_method.pdf", plot_prompt_method_accuracy,
        "VLM accuracy comparison across prompt strategies"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis plots from pre-computed model responses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plots.py              # Generate all plots
  python plots.py --plot 1     # Generate only Plot 1
  python plots.py --plot 1 2   # Generate Plots 1 and 2
  python plots.py --list       # List available plots
        """
    )
    parser.add_argument('--plot', type=int, nargs='+', choices=[1, 2, 3],
                       help='Plot number(s) to generate (default: all)')
    parser.add_argument('--list', action='store_true',
                       help='List available plots and exit')
    parser.add_argument('--no-stats', action='store_true',
                       help='Suppress statistical analysis output')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable plots:")
        print("-" * 60)
        for num, (filename, _, description) in PLOTS.items():
            print(f"  {num}. {filename}")
            print(f"     {description}\n")
        return

    PLOTS_DIR.mkdir(exist_ok=True)

    df = load_results()

    plots_to_generate = args.plot if args.plot else [1, 2, 3]

    print(f"\nGenerating {len(plots_to_generate)} plot(s)...\n")

    for plot_num in plots_to_generate:
        filename, plot_func, _ = PLOTS[plot_num]
        output_path = PLOTS_DIR / filename

        if plot_num == 3:
            plot_func(df, output_path, print_stats=not args.no_stats)
        else:
            plot_func(df, output_path)

    print(f"\nDone! Plots saved to: {PLOTS_DIR.absolute()}")


if __name__ == "__main__":
    main()
