import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

# Define the dataset and metrics
dataset = "credit"
metrics_list = ["avg_score", "model_shift", "acc", "failToRecourse", "avgRecourseCost", "t_rate", "avg_score_on_last_train", "recourse_cost_ratio", "avgNewRecourseCost", "recourse_cost_ratio2"]

# Folder paths
# folder_paths = {
#     "Folder2": "New Experiments/diversek_output/5-15",
#     "Folder3": "New Experiments/topk_continual_static_output/5-15",
#     "Folder4": "New Experiments/topk_output/5-15",
#     "Folder5": "New Experiments/diversek_continual_output/5-15",
#     "Folder6": "New Experiments/topk_continual_output/5-15",
# }

folder_paths = {
    "Folder2": "New Experiments/diversek_MLP_output/5-15",
    "Folder3": "New Experiments/topk_continual_static_MLP_output/5-15",
    "Folder4": "New Experiments/topk_MLP_output/5-15",
    "Folder5": "New Experiments/diversek_continual_MLP_output/5-15",
    "Folder6": "New Experiments/topk_continual_MLP_output/5-15",
}

# Dictionary to store extracted data
data_dict = {
    folder: {metric: [] for metric in metrics_list}
    for folder in folder_paths
}

# Load data
for folder_name, folder_path in folder_paths.items():
    if not os.path.exists(folder_path):
        continue  # Skip if folder doesn't exist

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            parts = filename.split("_")
            if len(parts) < 6:
                continue  # Skip invalid filenames

            dataset_name = parts[4]  # Extract dataset name
            if dataset_name != dataset:  # Only process the "synthetic" dataset
                continue

            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Store values for each metric
            for metric in metrics_list:
                if "avgNewRecourseCost" in df.columns and "avgOriginalRecourseCost" in df.columns:
                    try:
                        new_costs = pd.to_numeric(df["avgNewRecourseCost"], errors="coerce")
                        original_costs = pd.to_numeric(df["avgOriginalRecourseCost"], errors="coerce")
                        valid_mask = (new_costs != 0) & (original_costs != 0)

                        # Apply mask and compute ratio only on valid rows
                        ratio = new_costs[valid_mask] / original_costs[valid_mask]
                        # Skip the same way as other metrics
                        skip_count = 1
                        cleaned_ratio = ratio[skip_count:].dropna().tolist()
                        data_dict[folder_name]["recourse_cost_ratio"].extend(cleaned_ratio)
                    except Exception as e:
                        print(f"Error computing ratio in {filename}: {e}")

                if "avgRecourseCost" in df.columns and "avgNewRecourseCost" in df.columns:
                    try:
                        new_costs = pd.to_numeric(df["avgNewRecourseCost"], errors="coerce")
                        avg_costs = pd.to_numeric(df["avgRecourseCost"], errors="coerce")
                        valid_mask = (new_costs != 0) & (avg_costs != 0)

                        # Apply mask and compute ratio only on valid rows
                        ratio = new_costs[valid_mask] / avg_costs[valid_mask]
                        # Skip the same way as other metrics
                        skip_count = 1
                        cleaned_ratio = ratio[skip_count:].dropna().tolist()
                        data_dict[folder_name]["recourse_cost_ratio2"].extend(cleaned_ratio)
                    except Exception as e:
                        print(f"Error computing ratio in {filename}: {e}")

                if metric in df.columns:
                    values = df[metric].astype(str).values  # Convert to string
                    cleaned_values = []

                    # Skip first 6 values if metric is "acc", otherwise skip 3
                    skip_count = 1 if metric == "acc" or metric == "model_shift" or metric == "failToRecourse" or metric == "avgRecourseCost" else 0
                    values = values[skip_count:]

                    for val in values:
                        if "tensor" in val:  # Check if it's a tensor format
                            try:
                                num = float(val.replace("tensor(", "").replace(")", ""))  # Extract numeric value
                                cleaned_values.append(num)
                            except ValueError:
                                print(f"Skipping invalid tensor value: {val}")  # Debugging
                        else:
                            try:
                                cleaned_values.append(float(val))  # Convert normal numbers
                            except ValueError:
                                print(f"Skipping invalid value: {val}")

                    data_dict[folder_name][metric].extend(cleaned_values)

# Ensure data is loaded correctly
print("Data dictionary preview:", data_dict)

# Colors and labels
colors = {
    "Folder2": "#ff7f0e",
    "Folder3": "#1f77b4",
    "Folder4": "#9467bd",
    "Folder5": "#d62728",
    "Folder6": "#2ca02c"
}
linestyles = {
    "Folder2": "-",
    "Folder3": "-",
    "Folder4": "-",
    "Folder5": "-",
    "Folder6": "-",
}
labels = {
    "Folder2": "Fair Top-k",
    "Folder3": "Top-k with static continual learning",
    "Folder4": "Top-k",
    "Folder5": "Fair Top-k with DCL",
    "Folder6": "Top-k with DCL",
}

# Y-axis limits for each metric
y_limits = {
    "avg_score": (5, -15),
    "model_shift": (0, 5),
    "acc": (0.5, 1)
}

# Compute adaptive y-limits for "model_shift"
all_model_shift_values = []
for folder_name in folder_paths.keys():
    all_model_shift_values.extend(data_dict[folder_name]["model_shift"])

# Set adaptive limits
if all_model_shift_values:
    min_shift, max_shift = min(all_model_shift_values), max(all_model_shift_values)
    margin = (max_shift - min_shift) * 0.1  # 10% margin
    y_limits["model_shift"] = (max(0, min_shift - margin), max_shift + margin)
else:
    y_limits["model_shift"] = (0, 0)  # Default fallback in case no data exists

# Store legend handles for separate legend plot
legend_handles = []
legend_labels = []

title_map = {
    "avg_score": "Higher Standard2",
    "model_shift": "Model Shift",
    "acc": "Short-Term Accuracy",
    "failToRecourse": "Recourse Failure Rate",
    "avgRecourseCost": "Average Recourse Cost",
    "t_rate": "Test Acceptance Rate",
    "avg_score_on_last_train": "Higher Standard",
    "recourse_cost_ratio": "Ratio of Effort",
    "avgNewRecourseCost": "Average New Recourse Cost",
    }
colors = {
    "Fair Top-k": "#ff7f0e",
    "Top-k with static continual learning": "#1f77b4",
    "Top-k": "#9467bd",
    "Fair Top-k with DCL": "#d62728",
    "Top-k with DCL": "#2ca02c"
}

# Generate separate figures for each metric
for metric in metrics_list:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # Prepare data for seaborn boxplot
    plot_data = []
    plot_labels = []
    plot_colors = []

    for folder_name in folder_paths:
        values = data_dict[folder_name][metric]
        if values:
            plot_data.extend(values)
            plot_labels.extend([labels[folder_name]] * len(values))
            plot_colors.extend([colors[labels[folder_name]]] * len(values))

    # Create DataFrame for seaborn
    box_df = pd.DataFrame({
        metric: plot_data,
        "Method": plot_labels
    })

    # Create boxplot using seaborn
    sns.boxplot(data=box_df, x="Method", y=metric, palette=colors, ax=ax)
    ax.set_xticks([])
    # Customize labels and title
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right', fontsize=14)
    ax.set_yticklabels(ax.get_yticks(), fontsize=14)
    

    if metric == "avg_score":
        ax.set_ylabel("Log Sum", fontsize=16)
    elif metric == "acc":
        ax.set_ylabel("Accuracy", fontsize=16)
    elif metric == "model_shift":
        ax.set_ylabel("Model Shift", fontsize=16)
    elif metric == "failToRecourse":
        ax.set_ylabel("Failure Rate", fontsize=16)
    elif metric == "avgRecourseCost":
        ax.set_ylabel("Recourse Cost", fontsize=16)
    elif metric == "t_rate":
        ax.set_ylabel("Test Acceptance Rate", fontsize=16)
    elif metric == "avg_score_on_last_train":
        ax.set_ylabel("Log Sum", fontsize=16)
    else:
        ax.set_ylabel(metric, fontsize=16)

    ax.set_title(title_map.get(metric, ""), fontsize=20, fontweight="normal", pad=15)
    ax.set_xlabel(None)
    ax.xaxis.label.set_visible(False)
    ax.tick_params(axis='x', labelsize=14, labelrotation=15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax.grid(True, linestyle=":", alpha=0.6)

    # Save plot
    plt.savefig(f"Result/5-15_box_{metric}_comparison.png")
    plt.show()


legend_labels = list(colors.keys())
legend_handles = [Patch(facecolor=colors[label], label=label) for label in legend_labels]

# Sort handles and labels based on the preferred order
preferred_order = ["Fair Top-k", "Fair Top-k with DCL"]
sorted_legend = sorted(
    zip(legend_labels, legend_handles),
    key=lambda x: preferred_order.index(x[0]) if x[0] in preferred_order else len(preferred_order)
)

if sorted_legend:
    legend_labels, legend_handles = zip(*sorted_legend)

    fig_legend, ax_legend = plt.subplots(figsize=(20, 1.5))
    ax_legend.axis("off")
    ax_legend.legend(legend_handles, legend_labels, loc="upper center", 
                     fontsize=20, frameon=True, ncol=3)

    plt.savefig("Result/LTlegend_plot.png")
    plt.show()
else:
    print("Warning: Legend data is empty. Skipping legend plot.")

