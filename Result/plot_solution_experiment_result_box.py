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
#     "Folder2": "New Experiments/diversek_MLP_output/5-16",
#     "Folder3": "New Experiments/topk_continual_static_MLP_output/5-16",
#     "Folder4": "New Experiments/topk_MLP_output/5-16",
#     "Folder5": "New Experiments/diversek_continual_MLP_output/5-16",
#     "Folder6": "New Experiments/topk_continual_MLP_output/5-16",
# }

folder_paths = {
    "Folder2": "New Experiments/diversek_output/5-16",
    "Folder3": "New Experiments/topk_continual_static_output/5-16",
    "Folder4": "New Experiments/topk_output/5-16",
    "Folder5": "New Experiments/diversek_continual_output/5-16",
    "Folder6": "New Experiments/topk_continual_output/5-16",
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
            if dataset_name != dataset:  # Only process the specified dataset
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

# Labels and colors for each folder
labels = {
    "Folder2": "Fair Top-k",
    "Folder3": "Top-k with static continual learning",
    "Folder4": "Top-k",
    "Folder5": "Fair Top-k with DCL",
    "Folder6": "Top-k with DCL",
}

colors = {
    "Fair Top-k": "#ff7f0e",
    "Fair Top-k with DCL": "#d62728",
    "Top-k with DCL": "#2ca02c",
    "Top-k": "#cab3de",
    "Top-k with static continual learning": "#8fbbda"
}

# Define the desired order for the box plots
plot_order = ["Fair Top-k", "Fair Top-k with DCL", "Top-k with DCL", "Top-k with static continual learning", "Top-k"]

# Map from folder names to their corresponding labels
folder_to_label = {folder_name: labels[folder_name] for folder_name in folder_paths}

# Get the subset of folders that match our desired methods
selected_folders = []
for folder_name, label in folder_to_label.items():
    if label in plot_order:
        selected_folders.append(folder_name)

# Title mapping for metrics
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

# Generate separate figures for each metric
for metric in metrics_list:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # Prepare data for seaborn boxplot with your specified order
    plot_data = []
    plot_labels = []
    
    # Only include data for the folders in our desired order
    for folder_name in selected_folders:
        label = labels[folder_name]
        if label in plot_order:  # Double-check it's in our desired order
            values = data_dict[folder_name][metric]
            if values:
                plot_data.extend(values)
                plot_labels.extend([label] * len(values))

    # Create DataFrame for seaborn
    box_df = pd.DataFrame({
        metric: plot_data,
        "Method": plot_labels
    })
    
    # Create boxplot using seaborn with explicit order parameter
    sns.boxplot(data=box_df, x="Method", y=metric, order=plot_order, palette=colors, ax=ax)
    
    # Customize labels and title
    # ax.set_xticklabels(plot_order, rotation=15, ha='right', fontsize=14)
    ax.set_yticklabels(ax.get_yticks(), fontsize=40)

    ax.set_title(title_map.get(metric, ""), fontsize=46, fontweight="normal")
    ax.set_xlabel(None)
    ax.xaxis.label.set_visible(False)
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xticklabels([])

    # Your existing code already has these lines which are good:
    ax.set_xticklabels([])  # Remove the tick labels (method names under each box)
    ax.set_xlabel("")       # Clear the x-axis label text
    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=False)  # Keep ticks but hide labels

    # Full context - place this after the boxplot is created:
    sns.boxplot(data=box_df, x="Method", y=metric, order=plot_order, palette=colors, ax=ax)
        
    ax.set_xticklabels([])  # Remove x-tick labels
    ax.set_xlabel("")       # Remove x-axis label
    if metric == "avgRecourseCost":
        ax.set_ylabel("Logistic Model", fontsize=46, fontweight="normal")
    else:
        ax.set_ylabel("")
    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=False)  # Keep x-ticks, hide labels
    ax.tick_params(axis='y', which='both', left=True, labelleft=True)      # Keep y-ticks, hide labels

    # Save plot
    plt.savefig(f"Result/5-16L_box_{metric}_comparison.png")
    plt.show()

# Create a separate legend plot
legend_labels = plot_order
legend_handles = [Patch(facecolor=colors[label], label=label) for label in legend_labels]

fig_legend, ax_legend = plt.subplots(figsize=(40, 3))
ax_legend.axis("off")
ax_legend.legend(legend_handles, legend_labels, loc="upper center", 
                fontsize=46, frameon=True, ncol=3)

plt.savefig("Result/LTlegend_ordered_plot.png")
plt.show()