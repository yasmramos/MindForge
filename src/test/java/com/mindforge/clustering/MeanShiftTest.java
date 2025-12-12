package com.mindforge.clustering;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class MeanShiftTest {
    
    @Test
    void testBasicClustering() {
        Random random = new Random(42);
        double[][] data = new double[40][2];
        
        // Two clusters
        for (int i = 0; i < 20; i++) {
            data[i][0] = random.nextGaussian() * 0.5;
            data[i][1] = random.nextGaussian() * 0.5;
        }
        for (int i = 20; i < 40; i++) {
            data[i][0] = 5 + random.nextGaussian() * 0.5;
            data[i][1] = 5 + random.nextGaussian() * 0.5;
        }
        
        MeanShift ms = new MeanShift(2.0);
        int[] labels = ms.fitPredict(data);
        
        assertEquals(40, labels.length);
        assertTrue(ms.getNClusters() >= 1);
    }
    
    @Test
    void testClusterCenters() {
        double[][] data = {
            {0, 0}, {0.1, 0}, {0, 0.1},
            {5, 5}, {5.1, 5}, {5, 5.1}
        };
        
        MeanShift ms = new MeanShift(1.0);
        ms.fit(data);
        
        double[][] centers = ms.getClusterCenters();
        assertNotNull(centers);
    }
}
