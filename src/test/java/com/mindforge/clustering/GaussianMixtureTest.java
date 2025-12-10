package com.mindforge.clustering;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for GaussianMixture (GMM).
 */
public class GaussianMixtureTest {
    
    @Test
    void testFitAndPredict() {
        // Create two well-separated clusters
        double[][] X = {
            {1.0, 1.0}, {1.2, 0.8}, {0.8, 1.2}, {1.1, 1.1},
            {5.0, 5.0}, {5.2, 4.8}, {4.8, 5.2}, {5.1, 5.1}
        };
        
        GaussianMixture gmm = new GaussianMixture.Builder()
            .nComponents(2)
            .maxIter(100)
            .randomSeed(42)
            .build();
        
        gmm.fit(X);
        
        assertTrue(gmm.isFitted());
        assertEquals(2, gmm.getNumClusters());
        
        int[] labels = gmm.predict(X);
        assertEquals(8, labels.length);
        
        // First 4 points should be in same cluster
        assertEquals(labels[0], labels[1]);
        assertEquals(labels[0], labels[2]);
        assertEquals(labels[0], labels[3]);
        
        // Last 4 points should be in same cluster
        assertEquals(labels[4], labels[5]);
        assertEquals(labels[4], labels[6]);
        assertEquals(labels[4], labels[7]);
        
        // Two groups should be different
        assertNotEquals(labels[0], labels[4]);
    }
    
    @Test
    void testPredictProba() {
        double[][] X = {
            {0.0, 0.0}, {0.1, 0.1},
            {5.0, 5.0}, {5.1, 5.1}
        };
        
        GaussianMixture gmm = new GaussianMixture.Builder()
            .nComponents(2)
            .build();
        
        gmm.fit(X);
        
        double[] probs = gmm.predictProba(X[0]);
        assertEquals(2, probs.length);
        
        // Probabilities should sum to 1
        double sum = probs[0] + probs[1];
        assertEquals(1.0, sum, 0.01);
        
        // One probability should dominate
        assertTrue(Math.max(probs[0], probs[1]) > 0.8);
    }
    
    @Test
    void testCluster() {
        double[][] X = {
            {0.0, 0.0}, {0.5, 0.5},
            {10.0, 10.0}, {10.5, 10.5}
        };
        
        GaussianMixture gmm = new GaussianMixture.Builder()
            .nComponents(2)
            .build();
        
        int[] labels = gmm.cluster(X);
        
        assertEquals(4, labels.length);
        assertEquals(labels[0], labels[1]);
        assertEquals(labels[2], labels[3]);
        assertNotEquals(labels[0], labels[2]);
    }
    
    @Test
    void testGetters() {
        GaussianMixture gmm = new GaussianMixture.Builder()
            .nComponents(3)
            .build();
        
        double[][] X = {
            {0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0},
            {10.0, 0.0}, {11.0, 1.0}, {12.0, 2.0},
            {0.0, 10.0}, {1.0, 11.0}, {2.0, 12.0}
        };
        
        gmm.fit(X);
        
        assertEquals(3, gmm.getWeights().length);
        assertEquals(3, gmm.getMeans().length);
        assertEquals(3, gmm.getCovariances().length);
    }
    
    @Test
    void testBicAic() {
        double[][] X = {
            {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.0},
            {5.0, 5.0}, {5.1, 5.1}, {5.2, 5.0}
        };
        
        GaussianMixture gmm = new GaussianMixture.Builder()
            .nComponents(2)
            .build();
        
        gmm.fit(X);
        
        double bic = gmm.bic(X);
        double aic = gmm.aic(X);
        
        // BIC and AIC should be finite
        assertTrue(Double.isFinite(bic));
        assertTrue(Double.isFinite(aic));
    }
    
    @Test
    void testScore() {
        double[][] X = {
            {0.0, 0.0}, {0.5, 0.5},
            {5.0, 5.0}, {5.5, 5.5}
        };
        
        GaussianMixture gmm = new GaussianMixture.Builder()
            .nComponents(2)
            .build();
        
        gmm.fit(X);
        
        double score = gmm.score(X);
        assertTrue(Double.isFinite(score));
    }
    
    @Test
    void testNotFittedException() {
        GaussianMixture gmm = new GaussianMixture.Builder().build();
        
        assertThrows(IllegalStateException.class, () -> {
            gmm.predict(new double[]{1.0, 2.0});
        });
    }
    
    @Test
    void testBuilder() {
        GaussianMixture gmm = new GaussianMixture.Builder()
            .nComponents(5)
            .maxIter(200)
            .tol(1e-5)
            .initMethod("random")
            .randomSeed(123)
            .build();
        
        assertEquals(5, gmm.getNumClusters());
    }
}
