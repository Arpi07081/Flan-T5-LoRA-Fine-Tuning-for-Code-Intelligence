import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "presentation_plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": {
        "bg_color": "white",
        "text_color": "#333333",
        "palette": ["#a1c4fd", "#ffb199"],
        "style": "whitegrid"
    },
    "dark": {
        "bg_color": "#1e1e1e",
        "text_color": "white",
        "palette": ["#4db6ac", "#ba68c8"], # Brighter vivid colors for dark backgrounds
        "style": "darkgrid"
    }
}

def load_json(path):
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def plot_individual_metric(df, task_name, metric_name, filename, theme_key):
    task_metric_df = df[(df["Task"] == task_name) & (df["Metric"] == metric_name)]
    if task_metric_df.empty:
        return
        
    theme = THEMES[theme_key]
    
    # Set the seaborn global background grid style
    sns.set_theme(style=theme["style"], rc={"axes.spines.top": False, "axes.spines.right": False})
    
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=theme['bg_color'])
    ax.set_facecolor(theme['bg_color'])
    
    sns.barplot(data=task_metric_df, x="Model", y="Score", palette=theme['palette'], ax=ax)
    
    plt.title(f"{task_name} Evaluation: {metric_name}", pad=25, fontweight='black', fontsize=18, color=theme['text_color'])
    
    original_max = task_metric_df['Score'].max()
    
    # Ceiling rules
    if original_max == 0.0:
        y_ceiling = 10  
    else:
        y_ceiling = min(100, original_max * 1.3)
        
    plt.ylim(0, y_ceiling)
    plt.ylabel("Score (%)", fontsize=14, fontweight='bold', labelpad=15, color=theme['text_color'])
    plt.xlabel("", fontsize=14)
    plt.xticks(fontsize=14, fontweight='bold', color=theme['text_color'])
    plt.yticks(fontsize=12, color=theme['text_color'])
    
    # Configure axes tick colors
    ax.tick_params(colors=theme['text_color'])
    ax.spines['bottom'].set_color(theme['text_color'])
    ax.spines['left'].set_color(theme['text_color'])
    
    # We DO NOT fake the data to 0.5 anymore. It stays exactly at 0 mathematically.
    # Labeling loop
    for p in ax.patches:
        height = p.get_height()
        
        ax.annotate(f"{height:.2f}%", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    xytext=(0, 6), 
                    textcoords='offset points',
                    fontweight='black',
                    fontsize=14,
                    color=theme['text_color'])
            
    plt.tight_layout()
    
    save_path = PLOTS_DIR / f"{filename}_{theme_key}.png"
    plt.savefig(save_path, dpi=400, facecolor=theme['bg_color'], transparent=False, bbox_inches='tight')
    plt.close()

def generate_plots():
    print("Loading metrics...\n")
    base_mbpp = load_json(RESULTS_DIR / "base_model" / "mbpp_metrics.json")
    base_spider = load_json(RESULTS_DIR / "base_model" / "spider_metrics.json")
    
    tuned_mbpp = load_json(RESULTS_DIR / "finetuned" / "mbpp_metrics.json")
    tuned_spider = load_json(RESULTS_DIR / "finetuned" / "spider_metrics.json")
    
    data = []

    if base_mbpp:
        data.append({"Model": "Base (Flan-T5)", "Task": "MBPP", "Metric": "Strict Pass@1", "Score": base_mbpp.get("pass_at_1_percentage", 0)})
        data.append({"Model": "Base (Flan-T5)", "Task": "MBPP", "Metric": "AST Compilation Rate", "Score": base_mbpp.get("compilation_rate_percentage", 0)})
        data.append({"Model": "Base (Flan-T5)", "Task": "MBPP", "Metric": "ROUGE-L Overlap", "Score": base_mbpp.get("rougeL", 0)})
        data.append({"Model": "Base (Flan-T5)", "Task": "MBPP", "Metric": "BLEU-2 Overlap", "Score": base_mbpp.get("bleu", 0)})
    if tuned_mbpp:
        data.append({"Model": "Finetuned (LoRA)", "Task": "MBPP", "Metric": "Strict Pass@1", "Score": tuned_mbpp.get("pass_at_1_percentage", 0)})
        data.append({"Model": "Finetuned (LoRA)", "Task": "MBPP", "Metric": "AST Compilation Rate", "Score": tuned_mbpp.get("compilation_rate_percentage", 0)})
        data.append({"Model": "Finetuned (LoRA)", "Task": "MBPP", "Metric": "ROUGE-L Overlap", "Score": tuned_mbpp.get("rougeL", 0)})
        data.append({"Model": "Finetuned (LoRA)", "Task": "MBPP", "Metric": "BLEU-2 Overlap", "Score": tuned_mbpp.get("bleu", 0)})

    if base_spider:
        data.append({"Model": "Base (Flan-T5)", "Task": "SPIDER", "Metric": "Strict Exact Match", "Score": base_spider.get("em_strict_percentage", 0)})
        data.append({"Model": "Base (Flan-T5)", "Task": "SPIDER", "Metric": "Relaxed Exact Match", "Score": base_spider.get("em_relaxed_percentage", 0)})
        data.append({"Model": "Base (Flan-T5)", "Task": "SPIDER", "Metric": "ROUGE-L Overlap", "Score": base_spider.get("rougeL", 0)})
        data.append({"Model": "Base (Flan-T5)", "Task": "SPIDER", "Metric": "BLEU-2 Overlap", "Score": base_spider.get("bleu", 0)})
    if tuned_spider:
        data.append({"Model": "Finetuned (LoRA)", "Task": "SPIDER", "Metric": "Strict Exact Match", "Score": tuned_spider.get("em_strict_percentage", 0)})
        data.append({"Model": "Finetuned (LoRA)", "Task": "SPIDER", "Metric": "Relaxed Exact Match", "Score": tuned_spider.get("em_relaxed_percentage", 0)})
        data.append({"Model": "Finetuned (LoRA)", "Task": "SPIDER", "Metric": "ROUGE-L Overlap", "Score": tuned_spider.get("rougeL", 0)})
        data.append({"Model": "Finetuned (LoRA)", "Task": "SPIDER", "Metric": "BLEU-2 Overlap", "Score": tuned_spider.get("bleu", 0)})

    df = pd.DataFrame(data)
    
    if df.empty:
        return
        
    print("=== Generating Dark & Light Mode Plots ===\n")
    
    for task in ["MBPP", "SPIDER"]:
        task_metrics = df[df["Task"] == task]["Metric"].unique()
        for metric in task_metrics:
            safe_name = metric.replace(" ", "_").replace("@", "").lower()
            filename = f"{task.lower()}_{safe_name}"
            
            # Generate Light Theme
            plot_individual_metric(df, task, metric, filename, "light")
            # Generate Dark Theme
            plot_individual_metric(df, task, metric, filename, "dark")
            
            print(f"✓ Rendered: {filename}")
            
    print(f"\nAll done! 16 themed charts were saved to:\n{PLOTS_DIR}")

if __name__ == "__main__":
    generate_plots()
