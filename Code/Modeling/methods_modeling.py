from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import DistanceMetric
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings("ignore", category=Warning) 
sns.set_style("ticks")
sns.set_context("talk")

class Methods():
    
    """Class with some functions for the project. Explanations within each method."""
    
    def __init__(self):
        self.kmeans = None
        self.hierar = None
    
    
    def clustering_kmeans(self,df, n_cluster):
        
        '''Class with some functions for the project. Explanations within each method.'''
        
        model = KMeans(
        n_clusters=n_cluster,  
        n_init=100,     
        max_iter=1000,
        random_state=0) 
        predict = model.fit_predict(df)
        self.kmeans = model.labels_
        
        return predict
    
    
    def clustering_hierar(self, df, n_cluster):
        
        '''Hierarchical cluster method'''
        
        model = AgglomerativeClustering(n_clusters=n_cluster)
        predict = model.fit_predict(df)
        self.hierar = model.labels_
        
        return predict

    
    def ideal_clusters(self, model, df):
        
        '''Optimal clustering method, using KElbowVisualizer'''
        
        _model = model
        fig, ax =plt.subplots(1, 1, figsize=(16, 8))
        visualizer = KElbowVisualizer(_model, k=(2, 12), ax=ax)
        visualizer.fit(df)
        visualizer.show()
        
    
    def report_clustering(self, df_scale, df_raw, cluster_name, n_cluster ):
        
        '''Report of each cluster and visualization of countries by group.'''
        
        for i in range(n_cluster):    
            cluster_number = df_scale[df_scale[cluster_name]==i].drop(  
                                                    ['cluster_hierar','cluster_kmeans'], axis=1)
            
            print(f"""List of Countries in cluster {i}, total {cluster_number.value_counts().sum()} countries:
            \n       {str(df_scale[df_scale[cluster_name]==i].index)[7:-40]}""")
            
            print('*'*80)
            print(f'Describe Cluster {i}')
            
            print(f"""\033[2;30;47m{
                df_raw[df_raw[cluster_name]==i].drop(['cluster_hierar','cluster_kmeans'], axis=1).describe()
                }\033[0m""")    
            
            print('*'*80)
            print(f"Boxplot cluster {i}")
            plt.figure(figsize=(16,8))
            plt.title(f'Cluster Countries Nº {i}')
            plt.xlabel('Variables')
            plt.ylabel('Range')
            sns.boxplot(cluster_number)
            plt.show();
        
    
    def best_point_scipy(self, df_scale, model_name, n_cluster):
        
        '''Using Scipy we calculate the centroids and see which has the smallest average distance'''
        
        for cluster_id in range(n_cluster):
            df_cluster = df_scale[df_scale[model_name] == cluster_id].drop(
                        ['cluster_kmeans', 'cluster_hierar'], axis=1)
            distances = pdist(df_cluster.values, metric='euclidean')    
            distance_matrix = squareform(distances)
            total_distances = np.sum(distance_matrix, axis=1)
            best_point_index = np.argmin(total_distances)
            best_point = df_cluster.iloc[best_point_index].name.upper()        
            print(f"O País com melhor ponto médio do cluster {cluster_id} é:\033[2;30;47;3m{best_point}\033[0m - Distância: {total_distances[best_point_index]}")
    
    
    def best_point_slearn(self, df_scale,model_name, n_cluster):
        
        '''Using Sklearn we calculate the centroids and see which has the smallest average distance'''
        
        for i in range(n_cluster): 
            df_distance = df_scale[df_scale[model_name]==i].drop(['cluster_kmeans','cluster_hierar'], axis=1)
            distances = pairwise_distances(df_distance.values)
            total_distances = distances.sum(axis=1)
            best_point_index = total_distances.argmin()
            best_point = df_distance.iloc[best_point_index].name.upper()
            print(f"O País com melhor ponto médio do cluster {i} é: \033[2;30;47;3m{best_point}\033[0m")

    
    def distance(self, point1, point2):
        
        """Distance between points using Numpy."""
        
        return np.sqrt(np.sum((point1 - point2)**2))

    
    def best_point_numpy(self, df_scale,model_name, n_cluster):
        
        '''Using Numpy we calculate the centroids and see which has the smallest average distance'''
        
        for k in range(n_cluster):
            distance_numpy = df_scale[df_scale[model_name]==k].drop(['cluster_kmeans','cluster_hierar'], axis=1)
            cluster_numpy = np.array(distance_numpy)
            distances = []
            
            for i, item1 in enumerate(cluster_numpy):
                avg_distance = 0
                for j, item2 in enumerate(cluster_numpy):
                    if i != j:
                        dist = self.distance(item1, item2)
                        avg_distance += dist
                avg_distance /= len(cluster_numpy) - 1 
                distances.append(avg_distance)        
            best_item_idx = np.argmin(distances) 
            best_point_np =  distance_numpy.iloc[best_item_idx].name.upper()
            print(f"O País com melhor ponto médio do cluster {k} é:\033[2;30;47;3m{best_point_np}\033[0m")

    
    def clustermap(self, df_scale, metric, method):
        
        """Generates a Clustermap chart"""
        
        distancetype = DistanceMetric.get_metric(metric)
        distances = distancetype.pairwise(df_scale) 
        distances = pd.DataFrame(distances, 
                                columns=df_scale.index,
                                index=df_scale.index)

        return sns.clustermap(distances, figsize=(16,16), cmap="Reds_r", method=method);
    
    
    def dendogram(self, df_scale, metric, method, distance_threshold):
        
        """Generates a Dendogram chart"""
        
        threshold = distance_threshold
        plt.style.use('tableau-colorblind10')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        linkage = sch.linkage(df_scale, method=method, optimal_ordering=True, metric=metric)
        dendrogram = sch.dendrogram(linkage,labels=df_scale.index)
        ax.set_title('Hierarchical Clustering')
        ax.set_xlabel('Countries')
        ax.set_ylabel('Distances')
        ax.axhline(threshold, color='magenta', ls=":");
        
        return plt.show()
    
    def silhouette(self, df_scale):
        
        """silhouette_score** to compare the two models and see which in this 
        case had better behavior"""
        
        kmeans_silhouette = silhouette_score(df_scale, self.kmeans)
        hierarchical_silhouette = silhouette_score(df_scale, self.hierar)
        print("Valor da métrica silhouette para clusterização k-means: ", kmeans_silhouette)
        print("Valor da métrica silhouette para clusterização hierárquica: ", hierarchical_silhouette)


    def medoids(self,df_scale, cluster_type):
        
        '''Using Sklearn-Extra's Kmedoids method, we check the medoids of each cluster.'''
                
        for i in range(3):
            df_medoid = df_scale[df_scale[cluster_type]==i].drop(['cluster_kmeans','cluster_hierar'], axis=1)
            df_array_medoid= np.array(df_medoid) 

            KMobj = KMedoids(n_clusters=1).fit(df_array_medoid)
            kmedoid = KMobj.cluster_centers_.flatten().tolist()
            kmedoid = list(map(float, kmedoid))

            isin_filter = df_medoid.isin(kmedoid)
            row_filter = isin_filter.sum(axis=1) == len(kmedoid)
            linha_localizada = df_medoid.loc[row_filter]

            print(f"O Medoid do Cluster {i} é: {linha_localizada.index[0].upper()}")  
