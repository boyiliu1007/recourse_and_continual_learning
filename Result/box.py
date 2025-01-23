import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define datasets and metrics
datasets = ["UCIcredit", "credit", "synthetic"]
metrics_list = ["failToRecourse", "avgNewRecourseCost", "avgOriginalRecourseCost"]

# Folder paths
folder_paths = {
    "Folder2": "New Experiments/diversek_MLP_output/five2",
    "Folder3": "New Experiments/topk_continual_static_MLP_output/five2",
    "Folder4": "New Experiments/topk_MLP_output/five2",
    "Folder5": "New Experiments/diversek_continual_MLP_output/five2",
    "Folder6": "New Experiments/topk_continual_MLP_output/five2",
}

# Dictionary to store extracted data
data_records = []

# Load data
for folder_name, folder_path in folder_paths.items():
    if not os.path.exists(folder_path):
        continue  # Skip if folder doesn't exist

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            dataset_name = filename.split("_")[4]  # Extract dataset name
            if dataset_name not in datasets:
                continue  # Skip files not in our dataset list

            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Compute Cost Ratio
            if "avgNewRecourseCost" in df.columns and "avgOriginalRecourseCost" in df.columns:
                df["CostRatio"] = df["avgNewRecourseCost"] / df["avgOriginalRecourseCost"]

            # Store data for each metric
            for metric in metrics_list + ["CostRatio"]:
                if metric in df.columns:
                    for value in df[metric]:
                        data_records.append({"Folder": folder_name, "Dataset": dataset_name, "Metric": metric, "Value": value})

# Convert collected data into a DataFrame
plot_df = pd.DataFrame(data_records)

# Colors and labels
colors = {
    "Folder2": "#ff7f0e",
    "Folder3": "#2ca02c",
    "Folder4": "#d62728",
    "Folder5": "#9467bd",
    "Folder6": "#8c564b"
}
labels = {
    "Folder2": "fair-topk",
    "Folder3": "topk-continual-static-lambda",
    "Folder4": "topk",
    "Folder5": "fair-topk-continual-lambda",
    "Folder6": "topk-continual"
}
method_colors = {
    "fair-topk": "#ff7f0e",
    "topk-continual-static-lambda": "#2ca02c",
    "topk": "#d62728",
    "fair-topk-continual-lambda": "#9467bd",
    "topk-continual": "#8c564b"
}
# Set Seaborn style
sns.set(style="whitegrid")
plot_df["Method"] = plot_df["Folder"].map(labels)
# Generate box plots for each metric
for metric in metrics_list + ["CostRatio"]:
    if metric in plot_df["Metric"].unique():
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Dataset", y="Value", hue="Method", data=plot_df[plot_df["Metric"] == metric], palette=method_colors)
        plt.title(f"Box Plot of {metric} Across Datasets", fontsize=14)
        plt.xlabel("Dataset", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"Result/{metric}_boxplot.png")
        plt.show()
    else:
        print(f"Skipping {metric} - No data available.")

# Create a separate legend plot
fig_legend, ax_legend = plt.subplots(figsize=(8, 2))
ax_legend.axis("off")  # Hide axes
handles, _ = plt.gca().get_legend_handles_labels()
ax_legend.legend(handles, labels.values(), loc="center", fontsize=12, frameon=True, ncol=2)

plt.show()
