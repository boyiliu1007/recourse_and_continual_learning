import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

# Assuming the make_dataset function returns a train, test, sample, dataset object with x and y
from Config.config import make_dataset
train, test, sample, dataset = make_dataset(700, 500, 2500, 0.5, 'synthetic')

# Apply PCA to reduce data to 2D
pca = PCA(n_components=2)
data_pca = pca.fit_transform(train.x)

# Convert PCA result to a DataFrame for easier plotting
pca_df = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])

# Add the labels from train.y to the DataFrame (assuming train.y contains the labels)
pca_df['label'] = train.y

# Plot the data with color coding based on the labels
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['label'], cmap='viridis', alpha=0.7)
plt.title('PCA of Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter)  # Adds a color bar to the side to indicate label mapping
plt.grid(True)
plt.show()
