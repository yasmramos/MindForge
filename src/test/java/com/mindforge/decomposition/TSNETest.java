package com.mindforge.decomposition;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for t-SNE.
 */
public class TSNETest {
    
    @Test
    void testFitTransform() {
        // Create two clusters in high-dimensional space
        double[][] X = {
            {0, 0, 0, 0}, {0.1, 0.1, 0.1, 0.1}, {0.2, 0, 0.1, 0.2},
            {0.1, 0.2, 0, 0.1}, {0, 0.1, 0.2, 0},
            {5, 5, 5, 5}, {5.1, 5.1, 5.1, 5.1}, {5.2, 5, 5.1, 5.2},
            {5.1, 5.2, 5, 5.1}, {5, 5.1, 5.2, 5}
        };
        
        TSNE tsne = new TSNE.Builder()
            .nComponents(2)
            .perplexity(3)
            .maxIter(500)
            .randomSeed(42)
            .build();
        
        double[][] embedding = tsne.fitTransform(X);
        
        assertTrue(tsne.isFitted());
        assertEquals(10, embedding.length);
        assertEquals(2, embedding[0].length);
        
        // Verify clusters are separated in embedding
        double dist_within_cluster1 = distance(embedding[0], embedding[1]);
        double dist_within_cluster2 = distance(embedding[5], embedding[6]);
        double dist_between_clusters = distance(embedding[0], embedding[5]);
        
        assertTrue(dist_between_clusters > dist_within_cluster1,
            "Between-cluster distance should be larger than within-cluster");
    }
    
    @Test
    void testGetEmbedding() {
        double[][] X = createTestData();
        
        TSNE tsne = new TSNE.Builder()
            .nComponents(2)
            .perplexity(2)
            .maxIter(100)
            .build();
        
        double[][] embedding1 = tsne.fitTransform(X);
        double[][] embedding2 = tsne.getEmbedding();
        
        assertArrayEquals(embedding1[0], embedding2[0], 1e-10);
    }
    
    @Test
    void testKLDivergence() {
        double[][] X = createTestData();
        
        TSNE tsne = new TSNE.Builder()
            .nComponents(2)
            .perplexity(2)
            .maxIter(100)
            .build();
        
        tsne.fitTransform(X);
        
        double kl = tsne.getKLDivergence();
        assertTrue(Double.isFinite(kl));
        assertTrue(kl >= 0);
    }
    
    @Test
    void testBuilder() {
        TSNE tsne = new TSNE.Builder()
            .nComponents(3)
            .perplexity(20)
            .maxIter(500)
            .learningRate(100)
            .earlyExaggeration(8)
            .randomSeed(123)
            .build();
        
        assertEquals(3, tsne.getNComponents());
    }
    
    @Test
    void testNotFittedException() {
        TSNE tsne = new TSNE.Builder().build();
        
        assertThrows(IllegalStateException.class, () -> {
            tsne.getEmbedding();
        });
    }
    
    @Test
    void testTooFewSamples() {
        double[][] X = {{1, 2}, {3, 4}};
        
        TSNE tsne = new TSNE.Builder().build();
        
        assertThrows(IllegalArgumentException.class, () -> {
            tsne.fitTransform(X);
        });
    }
    
    private double[][] createTestData() {
        return new double[][] {
            {0, 0}, {0.5, 0.5}, {1, 0}, {0, 1},
            {10, 10}, {10.5, 10.5}, {11, 10}, {10, 11}
        };
    }
    
    private double distance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}
