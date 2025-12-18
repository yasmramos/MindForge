package io.github.yasmramos.mindforge.clustering;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class OPTICSAffinityTest {

    private double[][] createClusterData() {
        return new double[][] {
            // Cluster 1
            {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.0}, {0.0, 0.2},
            // Cluster 2
            {5.0, 5.0}, {5.1, 5.1}, {5.2, 5.0}, {5.0, 5.2},
            // Cluster 3
            {10.0, 0.0}, {10.1, 0.1}, {10.2, 0.0}, {10.0, 0.2}
        };
    }

    @Test
    void testOPTICSFitPredict() {
        double[][] X = createClusterData();
        OPTICS optics = new OPTICS.Builder()
            .minSamples(2)
            .maxEps(3.0)
            .xi(0.05)
            .build();
        
        int[] labels = optics.fitPredict(X);
        
        assertEquals(X.length, labels.length);
    }

    @Test
    void testOPTICSGetters() {
        double[][] X = createClusterData();
        OPTICS optics = new OPTICS();
        optics.fit(X);
        
        assertNotNull(optics.getLabels());
        assertNotNull(optics.getReachabilityDistances());
        assertNotNull(optics.getOrdering());
        assertEquals(X.length, optics.getLabels().length);
    }

    @Test
    void testOPTICSDefaultConstructor() {
        OPTICS optics = new OPTICS();
        assertNotNull(optics);
    }

    @Test
    void testAffinityPropagationFitPredict() {
        double[][] X = createClusterData();
        AffinityPropagation ap = new AffinityPropagation.Builder()
            .damping(0.5)
            .maxIterations(100)
            .convergenceIter(10)
            .build();
        
        int[] labels = ap.fitPredict(X);
        
        assertEquals(X.length, labels.length);
    }

    @Test
    void testAffinityPropagationGetters() {
        double[][] X = createClusterData();
        AffinityPropagation ap = new AffinityPropagation();
        ap.fit(X);
        
        assertNotNull(ap.getLabels());
        assertNotNull(ap.getClusterCentersIndices());
    }

    @Test
    void testAffinityPropagationDefaultConstructor() {
        AffinityPropagation ap = new AffinityPropagation();
        assertNotNull(ap);
    }
}
