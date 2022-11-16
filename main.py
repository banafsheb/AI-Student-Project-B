import streamlit as st
import pandas as pd
from PIL import Image
from io import StringIO
from sklearn.decomposition import PCA
import numpy as np
import sys, os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA



header = st.container()
dataset = st.container()
feature = st.container()
model_training = st.container()


def plot_pca_components(X, components=2):
    """Applies PCA and plots PCA first 2 components"""
    pca = PCA(n_components=components)
    pca.fit(X)
    X_pca = pca.transform(X)
    #st.write(X_pca)
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title("PCA components")
    #st.pyplot(plt)

    X_original = np.load('activations/activations_0_both.npy')
    X = X_original[0, 0, ::, ::]
    plot_pca_components(X, 2)

def main():
	flagfile = False
	tabHome, tabHelp, tabAbout = st.tabs(["Home", "Help", "About"])
	

	with tabHome:
		st.subheader("Studien Projekt")
		uploaded_file = st.file_uploader("Load weights/activations file")
		
		if uploaded_file is not None:
			flagfile = True
			
		st.title("Select the process")
		check1 = st.checkbox("PCA")
		check2 = st.checkbox("SR")
		check3 = st.checkbox("Saliency")

		if st.button("Run analysis"):
			if 	flagfile == True:
				if check1 == True:
					tab1, tab2, tab3 = st.tabs(["📈 head direction", "🗃 PCA Object", "PCA components"])
					with tab1:
						st.subheader("the first head direction")
						X_data = np.load(uploaded_file)
						st.write(X_data.shape)
						X = X_data[0, 0, ::, ::] # Slice data accordingly
						st.write(X.shape)
						st.write(X)
						X = X.T # Transpose data for PCA such that we have 625 features
					with tab2:
						st.write(X.shape)
					
						st.subheader("Create PCA Object")
						pca = PCA(n_components=2) # Create PCA Object
						pca.fit(X) # Do PCA on Data X
						st.write(pca.components_) # print PCA components
						st.write(pca.explained_variance_) # print variances of these components
					with tab3:
						X_pca = pca.transform(X) # Transform/Projects data on it's PCA components
						st.write(X_pca.shape)
						st.write(X_pca)
						
						st.subheader("PCA components")
						plt.scatter(X_pca[:, 0], X_pca[:, 1]) # Plot first two PCA components
						st.pyplot(plt)

						# Repeat process for 3 components
						pca = PCA(n_components=3)
						pca.fit(X)
						X_pca = pca.transform(X)
						st.write(X_pca.shape)
						plt.scatter(X_pca[:, 0], X_pca[:, 2])
						st.pyplot(plt)

						X_new = pca.inverse_transform(X_pca)

						st.write(X_new.shape)
						plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
						plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
						plt.axis('equal');
						st.pyplot(plt)
						#plot_pca_components(X, 2) #
				if check2 == True:
					# Loading data and slice
					X_original = np.load(uploaded_file)
					X = X_original[::, 0, ::, ::] # Slice data accordingly
					st.write(X.shape)
					def plot_heatmap(data):
						#Plots heatmaps individually, carefull takes long
						for hd in range(len(data)): # number of head directions # len(data) # 6 in example
							for n in range(data.shape[2]): # number of neurons # data.shape[2] # 50 in example # data.shape[2]
								# Get matrix of corresponding head direction and neuron
								matrix = data[hd, ::, n] # example [0, 0, 0, 3, 2, 4.5, ...., 0]
								# Reshaping it into k by k matrix [[],[],[],...,[]]
								num = len(matrix)
								k = int(math.sqrt(num))
								matrix = matrix.reshape((k, k))
								#Plot heatmap with seaborn library
								sns.heatmap(matrix, linewidth=0.5) # TODO MAKE IT WITH SAME COLORMAP AND COLORCODING 
								# Add title
								plt.title(f"Headdirection {hd} Neuron {n}")
								# Saveplot
								#plt.savefig(f"plots/heatmap_{hd}_{n}.pdf", dpi=300, bbox_inches='tight')
								#plt.close()
								#plt.show()
					
					def plot_heatmap_small(data):
						"""Plots heatmaps all small in one image"""
						# Setup number of subplots
						shape = data.shape # [6, 625, 50]
						f, axes = plt.subplots(shape[2], shape[0], squeeze=True)
						for hd in range(len(data)): # number of head directions # len(data) # 6 in example
							for n in range(data.shape[2]): # number of neurons # data.shape[2] # 50 in example # data.shape[2]
								# Get matrix
								matrix = data[hd, ::, n]
								# Reshape matrix
								num = len(matrix)
								k = int(math.sqrt(num))
								matrix = matrix.reshape((k, k))
								# Normalize matrix to work with colormap -> Divide by maximum value
								maximum = np.amax(matrix)
								maximum = max(maximum, 1)
								matrix = matrix / maximum
								# Plot heatmap
								axes[n, hd].imshow(matrix, cmap='hot', interpolation='nearest')
								# Removing all ticks and numbers
								axes[n, hd].axes.get_xaxis().set_visible(False)
								axes[n, hd].axes.get_yaxis().set_visible(False)
						# Resize image to fit everything
						f.set_size_inches((5*shape[0], 5*shape[2]))
						f.savefig("heatmap_small.pdf", dpi=300, bbox_inches='tight')
						#plt.show(X)
						
					plot_heatmap_small(X)
					#plot_heatmap(X)
			



			else:
				st.write("Please upload the file. :)") 




	with tabHelp:
		st.subheader("Help")

	with tabAbout:
		st.subheader("About")
		st.text("About us ...")


if __name__ == '__main__':
	main()