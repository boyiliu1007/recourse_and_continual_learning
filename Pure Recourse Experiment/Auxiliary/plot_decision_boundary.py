import matplotlib.pyplot as plt
import torch as pt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
import umap
from sklearn.linear_model import LogisticRegression

def plot_decision_boundary_pca(model, dataset, filename, n_components=2):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np
    import torch as pt

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x_data = dataset.x.numpy()
    y_data = dataset.y.numpy()

    original_dim = x_data.shape[1]
    print(f"Original feature dimensionality: {original_dim}")

    # Apply PCA
    pca = PCA(n_components=n_components)
    x_reduced = pca.fit_transform(x_data)

    # === Plot 1: Decision Boundary ===
    ax1 = axes[0]
    pos = y_data == 1
    neg = y_data == 0
    ax1.scatter(x_reduced[pos, 0], x_reduced[pos, 1], color='green', label='Class 1 (+)', alpha=0.7)
    ax1.scatter(x_reduced[neg, 0], x_reduced[neg, 1], color='red', label='Class 0 (-)', alpha=0.7)

    x_min, x_max = x_reduced[:, 0].min() - 0.5, x_reduced[:, 0].max() + 0.5
    y_min, y_max = x_reduced[:, 1].min() - 0.5, x_reduced[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    grid_points_reduced = np.zeros((xx.size, n_components))
    grid_points_reduced[:, 0] = xx.ravel()
    grid_points_reduced[:, 1] = yy.ravel()

    if n_components > 2:
        for i in range(2, n_components):
            grid_points_reduced[:, i] = np.mean(x_reduced[:, i])

    grid_points_original = pca.inverse_transform(grid_points_reduced)

    with pt.no_grad():
        tensor_input = pt.tensor(grid_points_original, dtype=pt.float32)
        predictions = model(tensor_input).reshape(xx.shape)

    ax1.contour(xx, yy, predictions.numpy(), levels=[0.5], linewidths=2, colors='blue')
    ax1.set_title(f'Decision Boundary (PCA {n_components} components)')
    ax1.set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    ax1.set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    ax1.legend()

    # === Plot 2: Scree Plot ===
    ax2 = axes[1]
    ax2.plot(
        np.arange(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        marker='o', linestyle='--', color='purple'
    )
    ax2.set_title('Scree Plot (Explained Variance Ratio)')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_xticks(np.arange(1, len(pca.explained_variance_ratio_) + 1))
    ax2.grid(True)

    # Save both plots into one image
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()




def plot_decision_boundary_tsne(model, dataset, filename, perplexity=30):
    """Visualize decision boundary after t-SNE dimensionality reduction"""
    plt.figure(figsize=(8, 6))
    x_data = dataset.x.numpy()
    y_data = dataset.y.numpy()
    
    # Apply t-SNE to reduce dimensions
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    x_reduced = tsne.fit_transform(x_data)
    
    # Plot original data in reduced space
    pos = y_data == 1
    neg = y_data == 0
    plt.scatter(x_reduced[pos, 0], x_reduced[pos, 1], color='green', label='Class 1 (+)', alpha=0.7)
    plt.scatter(x_reduced[neg, 0], x_reduced[neg, 1], color='red', label='Class 0 (-)', alpha=0.7)
    
    # Note: Can't easily visualize decision boundary with t-SNE as it's non-linear
    # and we can't map grid points back to original space
    
    plt.legend()
    plt.title(f'Data Visualization with t-SNE (perplexity={perplexity})')
    plt.savefig(filename)
    plt.close()



def plot_decision_boundary_umap(dataset, filename, n_neighbors=10, min_dist=0.1):
    """
    Visualize decision boundary in UMAP-reduced 2D space.
    NOTE: This trains a simple classifier on UMAP embeddings, 
    because UMAP is nonlinear and you can't map back to original space easily.
    """
    plt.figure(figsize=(8, 6))
    
    x_data = dataset.x.numpy()
    y_data = dataset.y.numpy()
    
    # Apply UMAP
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    x_reduced = umap_model.fit_transform(x_data)

    # Train a simple classifier on the 2D UMAP space
    classifier = LogisticRegression()
    classifier.fit(x_reduced, y_data)

    # Create grid in UMAP space
    x_min, x_max = x_reduced[:, 0].min() - 0.5, x_reduced[:, 0].max() + 0.5
    y_min, y_max = x_reduced[:, 1].min() - 0.5, x_reduced[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict over grid
    predictions = classifier.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], alpha=0.3, colors=['red', 'green'])
    
    # Plot original data in reduced space
    pos = y_data == 1
    neg = y_data == 0
    plt.scatter(x_reduced[pos, 0], x_reduced[pos, 1], color='green', label='Class 1 (+)', alpha=0.7)
    plt.scatter(x_reduced[neg, 0], x_reduced[neg, 1], color='red', label='Class 0 (-)', alpha=0.7)

    plt.legend()
    plt.title(f'Decision Boundary in UMAP Space\n(n_neighbors={n_neighbors}, min_dist={min_dist})')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.savefig(filename)
    plt.close()

def plot_decision_boundary(model, dataset, filename, round=0):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch as pt

    x_data = dataset.x.numpy()
    y_data = dataset.y.numpy()

    # Create figure
    plt.figure(figsize=(7, 6))

    # Plot data points
    pos = y_data == 1
    neg = y_data == 0
    plt.scatter(x_data[pos, 0], x_data[pos, 1], color='green', label='Class 1 (+)', alpha=0.7)
    plt.scatter(x_data[neg, 0], x_data[neg, 1], color='red', label='Class 0 (-)', alpha=0.7)

    # Create mesh grid for decision boundary
    x_min, x_max = x_data[:, 0].min() - 0.5, x_data[:, 0].max() + 0.5
    y_min, y_max = x_data[:, 1].min() - 0.5, x_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    with pt.no_grad():
        tensor_input = pt.tensor(grid_points, dtype=pt.float32)
        predictions = model(tensor_input).reshape(xx.shape)

    # Plot decision boundary
    plt.contour(xx, yy, predictions.numpy(), levels=[0.5], linewidths=2, colors='blue')
    plt.title(f'Decision Boundary round {round}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()