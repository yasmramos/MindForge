package io.github.yasmramos.mindforge.clustering;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class HierarchicalClusteringTest {
    
    @Test
    void testBasicClustering() {
        double[][] data = {
            {0, 0}, {0.1, 0.1}, {0.2, 0},
            {5, 5}, {5.1, 5.1}, {5, 5.2}
        };
        
        HierarchicalClustering hc = new HierarchicalClustering(2);
        int[] labels = hc.fitPredict(data);
        
        assertEquals(6, labels.length);
        
        // First 3 should be same cluster
        assertEquals(labels[0], labels[1]);
        assertEquals(labels[0], labels[2]);
        
        // Last 3 should be different cluster
        assertEquals(labels[3], labels[4]);
        assertNotEquals(labels[0], labels[3]);
    }
    
    @Test
    void testDifferentLinkages() {
        double[][] data = new double[20][2];
        Random random = new Random(42);
        for (int i = 0; i < 20; i++) {
            data[i][0] = random.nextGaussian();
            data[i][1] = random.nextGaussian();
        }
        
        HierarchicalClustering single = new HierarchicalClustering(3, HierarchicalClustering.Linkage.SINGLE);
        HierarchicalClustering complete = new HierarchicalClustering(3, HierarchicalClustering.Linkage.COMPLETE);
        
        int[] labels1 = single.fitPredict(data);
        int[] labels2 = complete.fitPredict(data);
        
        assertEquals(20, labels1.length);
        assertEquals(20, labels2.length);
    }
    
    @Test
    void testGetNClusters() {
        double[][] data = {{0, 0}, {1, 1}, {2, 2}, {10, 10}};
        HierarchicalClustering hc = new HierarchicalClustering(2);
        hc.fit(data);
        assertTrue(hc.getNClusters() >= 1);
    }
}
