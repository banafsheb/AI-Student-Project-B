import streamlit as st
import pandas as pd
from PIL import Image

#import qt5_applications
#from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QDialog
#from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
#import plotly.graph_objects as go




header = st.container()
dataset = st.container()
feature = st.container()
model_training = st.container()

with header:
	st.title('Studien Projekt!')
	st.text('Hello, here is our study project :)')



with dataset:
	st.header('Projekt data set')


	with st.expander("Minimize"):
	    st.write("""
	        You can minimize or maximize this part.
	    """)
	    st.image('image3.jpg')
	    result = st.button("Click Here")
	    st.write(result)
	    if result:
	    	st.write(":smile:")
	    col1, col2 = st.columns(2)
	    original = Image.open('image.jpg')
	    col1.header("Original")
	    col1.image(original, use_column_width=True)
	    grayscale = original.convert('LA')
	    col2.header("Grayscale")
	    col2.image(grayscale, use_column_width=True)


	    ##############################




	    uploaded_file = st.file_uploader("Choose a file")
	    if uploaded_file is not None:
	        # To read file as bytes:
	        bytes_data = uploaded_file.getvalue()
	        st.write(bytes_data)

	        # To convert to a string based IO:
	        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
	        st.write(stringio)

	        # To read file as string:
	        string_data = stringio.read()
	        st.write(string_data)

	        # Can be used wherever a "file-like" object is accepted:
	        dataframe = pd.read_csv(uploaded_file)
	        st.write(dataframe)

	    ##############################
	    X_original = np.load('activations/activations_0_both.npy')
	    st.write(X_original.shape) # It shows (6, 1, 625, 50)
	    X = X_original[0, 0, ::, ::] # Slice data accordingly
	    st.write(X.shape)
	    st.write(X)
	    X = X.T # Transpose data for PCA such that we have 625 features
	    st.write(X.shape)
	    st.bar_chart(X)				# ta inja ok bud
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
	    #print(X_pca.shape)
	    plt.scatter(X_pca[:, 0], X_pca[:, 2])
	    #plt.pyplot()


	    arr = np.random.normal(1, 1, size=100)
	    fig, ax = plt.subplots()
	    ax.hist(arr, bins=20)

	    st.pyplot(fig)



	
	""" x_lbl, y_lbl, z_lbl = f"PCA {idx_x_pca}", f"PCA {idx_y_pca}", f"PCA {idx_z_pca}"
	 data to plot
	x_plot, y_plot, z_plot = x_pca[:,idx_x_pca-1], x_pca[:,idx_y_pca-1], x_pca[:,idx_z_pca-1]

	trace1 = go.Scatter3d(
        x=x_plot, y=y_plot, z=z_plot,
        mode='markers',
        marker=dict(
            size=5,
            color=y,
            # colorscale='Viridis'
        )
    )



    fig = go.Figure(data=[trace1])

    fig.update_layout(scene = dict(
                        xaxis_title = 1,
                        yaxis_title = 2,
                        zaxis_title = 3),
                        width=700,
                        margin=dict(r=20, b=10, l=10, t=10),
                        )
  
    st.plotly_chart(fig, use_container_width=True)"""