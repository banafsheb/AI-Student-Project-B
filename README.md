# AI-Student-Project

Task One:
The folder activations contains multiple numpy array files. 
These represent activations of units of a neural network layer over some feature (in this case 2D space)

1. Write a script to perform PCA on the data
 - load one of the numpy arrays
 it is a matrix of size (625, 6, 50). Note : The three dimensions correspond to (point in space, head direction, unit).
 - Slice array (look at numpy documentation) to get a matrix corresponding to the first head direction. 
 It should have size (625,50)
 - This is the data on which you will perform PCA. 50 - number of data points 625- number of features
 - Use PCA.fit to fit the data to 2 PCA components
 - Use PCA.transform to transform the data to the generated PCA components
 - Print the explained variance
 - Plot the transformed data as a 2D scatter plot using matplotlib
 
 Repeat for 3 PCA components
