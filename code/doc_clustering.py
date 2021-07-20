import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def create_3d_plot(title, res, index, label1, label2):
    # creating an empty canvas
    fig = plt.figure()
    # Creating an empty 3D axes of the plot
    ax = fig.add_subplot(111, projection='3d')
    # Give a title to the plot
    ax.set_title(title)
    ax.scatter(res[0], res[1], zs=0, c='k', label=label1)
    ax.scatter(res.iloc[index, 0], res.iloc[index, 1], zs=0, c='r', label=label2,
               marker='o')
    ax.legend(loc='best', numpoints=1, fontsize=8)
    ax.set_xlim(-10, 2)
    ax.set_ylim(0.5, 2)
    ax.set_zlim(-1, 0)
    # Assign labels to the axis
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    # Showing the above 3D plot
    plt.show()

def analyse(col_name, df):
    size_of_each_cluster = df.groupby(col_name).size().reset_index()
    size_of_each_cluster.columns = [col_name, 'number_of_points']
    size_of_each_cluster['percentage'] = (size_of_each_cluster['number_of_points'] / np.sum(
        size_of_each_cluster['number_of_points'])) * 100
    print(size_of_each_cluster)


def isolationforest(feature_vector, df, res):
    # Isolation forest - to detect anomalies inside the data set
    model = IsolationForest(n_estimators=2, max_samples='auto', contamination=float(0.1), max_features=1.0,
                            random_state=42)
    model.fit(feature_vector)
    pred = model.predict(feature_vector)
    df['anomaly'] = pred

    outliers = df.loc[df['anomaly'] == -1]
    outlier_index = list(outliers.index)
    # For silhouette score
    cluster_labels = model.fit_predict(feature_vector)
    silhouette_avg = silhouette_score(feature_vector, cluster_labels)
    # Compute the silhouette scores for each sample
    # print("The average silhouette_score for isolation forest is :", silhouette_avg)

    # Making the 2D graph
    figsize = (12, 7)
    plt.figure(figsize=figsize)
    plt.title("Isolation Forest in 2D")
    plt.scatter(res[0], res[1], c='k', label="Normal points")
    plt.scatter(res.iloc[outlier_index, 0], res.iloc[outlier_index, 1], c='r', edgecolor="red",
                label="predicted outliers")
    plt.legend(loc="upper right")
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.show()

    # Making 3D graph
    create_3d_plot('Isolation Forest', res, outlier_index, 'Normal points', 'Outliers')

    # Analyzing
    analyse('anomaly', df)

def agglomerativeclustering(feature_vector, df, res):
    # Create a data frame for better working with the data
    feature_vector = np.array(feature_vector)
    model = AgglomerativeClustering(n_clusters=2, affinity='euclidean')
    model.fit(feature_vector)
    labels = model.labels_
    df['Agglomerative'] = labels
    indices = df.loc[df['Agglomerative'] == 1]
    index = list(indices.index)

    # 3D plot
    create_3d_plot('Agglomerative Clustering', res, index, 'Cluster 1', 'Cluster 2')

    # Analyzing
    analyse('Agglomerative', df)

    # For silhouette score
    cluster_labels = model.fit_predict(feature_vector)
    silhouette_avg = silhouette_score(feature_vector, cluster_labels)
    # Compute the silhouette scores for each sample
    # sample_silhouette_values = silhouette_samples(feature_vector, cluster_labels)
    # print("The average silhouette_score for agglomerative clustering is :", silhouette_avg)


def gmm(feature_vector, df, res):
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=14).fit(feature_vector)
    labels = gmm.predict(feature_vector)
    df['GMM_labels'] = labels
    indices = df.loc[df['GMM_labels'] == 1]
    index = list(indices.index)

    # 3D plot
    create_3d_plot('Gaussian Mixture Model', res, index,  'Cluster 1', 'Cluster 2')

    # Analyzing
    analyse('GMM_labels',df)

    # For silhouette score
    cluster_labels = gmm.fit_predict(feature_vector)
    silhouette_avg = silhouette_score(feature_vector, cluster_labels)
    # Compute the silhouette scores for each sample
    # sample_silhouette_values = silhouette_samples(feature_vector, cluster_labels)
    # print("The average silhouette_score for gaussian mixture model is :", silhouette_avg)


def clustering(feature_vector, df):
    pca = PCA(2)
    pca.fit_transform(feature_vector)
    res = pd.DataFrame(pca.transform(feature_vector))
    isolationforest(feature_vector, df, res)
    agglomerativeclustering(feature_vector, df,res)
    gmm(feature_vector, df, res)
    pd.DataFrame(df).to_csv('final_outcome.csv', index=False)
    # To see what each of the clustering methods entails
    isoForest = df.groupby(['anomaly'])['major_lda_topic'].value_counts(ascending=False, normalize=True)
    aggClust = df.groupby(['Agglomerative'])['major_lda_topic'].value_counts(ascending=False, normalize=True)
    gmm_group = df.groupby(['GMM_labels'])['major_lda_topic'].value_counts(ascending=False, normalize=True)