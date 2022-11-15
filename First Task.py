# Load necessaries packages
import qt5_applications
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QDialog
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import sys

app = QApplication()
app = qt5_applications.import_it('UI.ui')


X_original = np.load('activations/activations_0_both.npy') # Read in data from file
print(X_original.shape)
X = X_original[0, 0, ::, ::] # Slice data accordingly
print(X.shape)
print(X)
X = X.T # Transpose data for PCA such that we have 625 features
print(X.shape)

pca = PCA(n_components=2) # Create PCA Object
pca.fit(X) # Do PCA on Data X
print(pca.components_) # print PCA components
print(pca.explained_variance_) # print variances of these components

X_pca = pca.transform(X) # Transform/Projects data on it's PCA components
print(X_pca.shape)
print(X_pca)

plt.scatter(X_pca[:, 0], X_pca[:, 1]) # Plot first two PCA components
plt.show() # show plot

# Repeat process for 3 components
pca = PCA(n_components=3)
pca.fit(X)
X_pca = pca.transform(X)
print(X_pca.shape)
plt.scatter(X_pca[:, 0], X_pca[:, 2])
plt.show()

X_new = pca.inverse_transform(X_pca)
print(X_new.shape)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');
plt.show()

def plot_pca_components(X, components=2):
    """Applies PCA and plots PCA first 2 components"""
    pca = PCA(n_components=components)
    pca.fit(X)
    X_pca = pca.transform(X)
    print(X_pca)
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title("PCA components")
    plt.show()

    X_original = np.load('activations/activations_0_both.npy')
    X = X_original[0, 0, ::, ::]
    plot_pca_components(X, 2)