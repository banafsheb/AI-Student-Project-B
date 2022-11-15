import streamlit as st
import pandas as pd
from PIL import Image
from io import StringIO

#from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import sys
import matplotlib.pyplot as plt


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
						tab1.write(X_data.shape)
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
			else:
				st.write("Please Uplaod The file") 




	with tabHelp:
		st.subheader("Help")

	with tabAbout:
		st.subheader("About")
		st.text("About us ...")


if __name__ == '__main__':
	main()