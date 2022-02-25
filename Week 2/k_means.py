#%% 
import numpy as np 
### implementation of K-means algorithm ### 

def j_e(data, centroids, N_c):
    """ 
    quantification error function 
    Arguments       data, type == data
                    centroids, type = n_dimensional list
                    N_c = number of clusters 
    returns the normalized fitness and cluster labels """
    centroids = np.reshape(centroids, [N_c, data.shape[1]]) 

    cluster_labels, grocery_list = cluster_lab(data, centroids)
    
    fitness_unnormalized = 0 
    for cluster_index in range(len(centroids)): 
        fitness_unnormalized += (np.sum(grocery_list[cluster_labels == cluster_index, cluster_index]) / np.sum(cluster_labels == cluster_index))
    fitness_normalized = fitness_unnormalized / len(centroids)
    return fitness_normalized, cluster_labels

def Euclidean(dataloc, centroid_location):
    """ 
    Euclidean distance 
    Arguments       dataloc, type == type of position
                    centroid_location, type == type of centroid
                    
    Returns the sqrt of the dotproduct of the data/centroid matrices
    """
    return np.sqrt((dataloc-centroid_location) @ (dataloc-centroid_location).T)

def cluster_lab(data, centroids): 
    """Function to assign datapoints to clusters
        Arguments       dataloc, type == type of position
                        centroid_location, type == type of centroid
        Returns cluster label and a list with all distances calculated from each cluster's centroid to datapoint
    """
    grocery_list = np.zeros([len(data), len(centroids)])
    for i, d in enumerate(data):
        for cluster_index, centroid_location in enumerate(centroids): 
            grocery_list[i,cluster_index] = Euclidean(d, centroid_location)
    cluster_labels = np.argmin(grocery_list, axis = 1)
    return cluster_labels, grocery_list

def k_means(data, iterations, N_c): 
    """
    Implementation of K- means
    Arguments       data, type == data
                    iterations, type = int; number of iterations
                    N_c, type = int; number of clusters
    Returns best centroids and quantification errors of each iteration 
    """
    centroids = np.random.uniform(np.min(data), np.max(data), [N_c, data.shape[1]])
    j_history = []
    for i in range(iterations):
        clabels, _ = cluster_lab(data,centroids)
        for cluster in range(N_c): 
            clusterdata = data[clabels == cluster]
            centroid = np.mean(clusterdata, axis=0)
            centroids[cluster] = centroid 
        j_kmeans, _ = j_e(data, centroids, N_c)
        j_history.append(j_kmeans)


    return centroids, j_history
# %%

### EXPERIMENT 1 K-MEANS STARTS HERE - ARTIFICIAL PROBLEM 1###
### RUN THIS AND BELOW CELL 30 TIMES ###
#generate data
data = np.random.uniform(-1,1, (400, 2))
N_c = 2 # number of clusters

# %%

#plot the eventual clusters
import matplotlib.pyplot as plt 

centroids, jvalue = k_means(data, 100, N_c)
clabels, _ = cluster_lab(data, centroids)
plt.figure()
plt.title('Clustering for artificial problem 1 K-means')
plt.scatter(data[:,0], data[:,1], c = clabels)
plt.savefig('kmeans_artificial_j.png')
plt.show()

#plot the quantification error 
plt.figure()
plt.title('Quantification error for K-means Artificial problem 1')
plt.plot(jvalue)
plt.xlabel('# iterations')
plt.ylabel('Quantification error')
plt.savefig('kmeans_artificial.png')
plt.show()
# %% 

### EXPERIMENT 2 K-MEANS STARTS HERE - IRIS DATA SET ### 

#prepare IRIS data
import pandas as pd
df=pd.read_csv('iris.data', sep=',',header=None)
data_iris = df.values
label_iris = data_iris[:,4]
data_iris = data_iris[:,:-1].astype(float)

#%%

### RUN THIS CELL AND BELOW 30 TIMES  ###
#initiliaze parameters
N_c = 3 # number of clusters

centroids, jvalue_iris = k_means(data_iris, 100, N_c)

#plot eventual clustering 
clabels, _ = cluster_lab(data_iris, centroids)
plt.figure()
plt.title('Clustering for Iris data K-means')
plt.scatter(data_iris[:,0], data_iris[:,1], c = clabels)
plt.savefig('kmeans_iris_j.png')
plt.show()
print(clabels)

# %%
#plot the curve
plt.figure()
plt.title('Quantification error for K-means IRIS DATA')
plt.plot(jvalue_iris)
plt.xlabel('# iterations')
plt.ylabel('Quantification error')
plt.savefig('kmeans_irisdata.png')
plt.show()


# %%
