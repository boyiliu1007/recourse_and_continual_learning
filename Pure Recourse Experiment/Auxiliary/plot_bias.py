import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
df1 = pd.read_csv('Pure Recourse Experiment/Results/report/bias_list_0.1_0.1.csv')
df2 = pd.read_csv('Pure Recourse Experiment/Results/report/bias_list_0.1_4.csv')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df1['bias'], label='bias_list_0.1_0.1', linewidth=2)
plt.plot(df2['bias'], label='bias_list_0.1_4', linewidth=2)

# Customize the plot
plt.xlabel('Round')
plt.ylabel('Bias')
plt.title('Bias over Rounds')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('Pure Recourse Experiment/Results/report/bias_plot.png', dpi=300)
plt.close()