package com.mindforge.clustering;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class KMeansTest {
    
    @Test
    void testSimpleClustering() {
        // Two clear clusters
        double[][] data = {
            {1.0, 2.0}, {1.5, 1.8}, {1.2, 2.1},  // Cluster 1
            {8.0, 8.0}, {8.5, 8.2}, {8.1, 7.9}   // Cluster 2
        };
        
        KMeans kmeans = new KMeans(2);
        int[] clusters = kmeans.cluster(data);
        
        // Points in the same group should have the same cluster
        assertEquals(clusters[0], clusters[1], "Close points should be in the same cluster");
        assertEquals(clusters[0], clusters[2], "Close points should be in the same cluster");
        assertEquals(clusters[3], clusters[4], "Close points should be in the same cluster");
        assertEquals(clusters[3], clusters[5], "Close points should be in the same cluster");
        
        // Points from different groups should be in different clusters
        assertNotEquals(clusters[0], clusters[3], "Distant points should be in different clusters");
    }
    
    @Test
    void testGetCentroids() {
        double[][] data = {{1.0, 1.0}, {2.0, 2.0}, {10.0, 10.0}, {11.0, 11.0}};
        
        KMeans kmeans = new KMeans(2);
        kmeans.cluster(data);
        
        double[][] centroids = kmeans.getCentroids();
        assertEquals(2, centroids.length, "Should have 2 centroids");
        assertEquals(2, centroids[0].length, "Each centroid should be 2-dimensional");
    }
    
    @Test
    void testPredict() {
        double[][] data = {{1.0, 1.0}, {2.0, 2.0}, {10.0, 10.0}, {11.0, 11.0}};
        
        KMeans kmeans = new KMeans(2);
        int[] clusters = kmeans.cluster(data);
        
        // Predict cluster for a new point close to the first cluster
        int cluster = kmeans.predict(new double[]{1.5, 1.5});
        assertTrue(cluster == 0 || cluster == 1, "Predicted cluster should be valid");
    }
    
    @Test
    void testNumClusters() {
        KMeans kmeans = new KMeans(3);
        assertEquals(3, kmeans.getNumClusters(), "Should have correct number of clusters");
    }
    
    @Test
    void testInvalidK() {
        assertThrows(IllegalArgumentException.class, () -> {
            new KMeans(0);
        }, "Should throw exception for k <= 0");
    }
}
