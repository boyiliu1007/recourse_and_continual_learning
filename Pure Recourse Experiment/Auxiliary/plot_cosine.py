import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
df1 = pd.read_csv('Pure Recourse Experiment/Results/report/cosine_list_0.1_0.1.csv')
df2 = pd.read_csv('Pure Recourse Experiment/Results/report/cosine_list_0.1_4.csv')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df1['cosine'], label='cosine_list_0.1_0.1', linewidth=2)
plt.plot(df2['cosine'], label='cosine_list_0.1_4', linewidth=2)

# Customize the plot
plt.xlabel('Round')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity over Rounds')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('Pure Recourse Experiment/Results/report/cosine_similarity_plot.png', dpi=300)
plt.close()