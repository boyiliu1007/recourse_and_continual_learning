import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Configuration ===
folder_paths = {
    "Logistic Model": "New Experiments/topk_output/5-16",
    "MLP": "New Experiments/topk_MLP_output/5-16"
}

datasets_list = ["synthetic", "credit", "UCIcredit"]
# metric = "avgRecourseCost"
metric = "failToRecourse"

# === Combined Data Collection ===
all_data = []

for dataset_name in datasets_list:
    for folder_label, folder_path in folder_paths.items():
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".csv") and dataset_name in file:
                try:
                    df = pd.read_csv(os.path.join(folder_path, file))
                    if metric in df.columns:
                        values = pd.to_numeric(df[metric], errors="coerce").dropna()
                        values = values[values != 0]
                        for val in values:
                            all_data.append({
                                metric: val,
                                "Method": folder_label,
                                "Dataset": dataset_name
                            })
                except Exception as e:
                    print(f"Error reading {file}: {e}")

# === Plotting ===
if not all_data:
    print("No data found for any dataset.")
else:
    plot_df = pd.DataFrame(all_data)

    custom_palette = {
        "Logistic Model": "#ff7f0e",
        "MLP": "#1f77b4"
    }

    sns.set(style="whitegrid", font_scale=4.2)
    fig, ax = plt.subplots(figsize=(14, 8))

    # Box width and spacing: Method vs metric, hue by Dataset
    sns.boxplot(
        data=plot_df,
        x="Dataset",
        y=metric,
        hue="Method",
        palette=custom_palette,
        width=0.3,
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='both', which='major', labelsize=55)
    ax.legend(title="", fontsize=40, loc="upper right")
    # ax.get_legend().remove()

    plt.tight_layout()
    os.makedirs("Result", exist_ok=True)
    plt.savefig(f"Result/combined_boxplot_{metric}.png")
    plt.show()
    plt.close()
