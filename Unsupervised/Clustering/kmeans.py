import numpy as np 


class kmeans: 

    def __init__(self, number_clusters, distance_metrics) -> None:

        self.number_clusters = number_clusters 
        self.distance_metrics = distance_metrics


    def eucledian_dis(self, x, y): 

        distance = np.linalg.norm(x-y)

        return distance 

    def train(self, x):

        """
        x: m dimeisional np array (N,m)
        """ 

        # clusters for ouput of each point
        clusters = np.zeros((x.shape[0],1))
        temp_clusters = np.random.randint(low=0, high = self.number_clusters-1, size=(x.shape[0],1))

        # randomly select k points in the dataframe 
        centroid_index = np.random.choice(x.shape[0], size = self.number_clusters, replace= False)
        centroids = x[centroid_index,:]

        distances = np.empty([x.shape[0], self.number_clusters])

        while clusters.flatten() != temp_clusters.flatten(): 
            
            # for each points, coupute the distanec to each cluster 
            for i_r, row in enumerate(x): 

                for i_c, c in enumerate(centroids): 

                    distance = self.eucledian_dis(c, row)
                    distances[i_r, i_c] = distance

                # for each point, assign them to clusters with minimum distances to it  
                temp_clusters[i_r,1] = np.argmin(distances[i_r,:])

            # compute the new centroid by taking the average 
            for i_c, c in enumerate(centroids): 

                # get the index of x that belongs to cluster 
                indexs = np.where(temp_clusters[:,1]==i_c)
                
                # get the mean 
                centroids[i_c,:] = np.mean(x[indexs], axis=0)
        
if __name__ == '__main__': 

    x = np.random.rand(10,3) 
    number_clusters = 2 
    distance_metrics = 'eucledian'

    model = kmeans(number_clusters, distance_metrics)
    model.train(x)

