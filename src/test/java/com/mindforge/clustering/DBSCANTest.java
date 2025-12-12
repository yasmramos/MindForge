package com.mindforge.clustering;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class DBSCANTest {
    
    @Test
    void testBasicClustering() {
        double[][] data = {
            {1, 1}, {1.1, 1}, {1, 1.1}, {1.1, 1.1},
            {5, 5}, {5.1, 5}, {5, 5.1}, {5.1, 5.1},
            {10, 10}  // noise point
        };
        
        DBSCAN dbscan = new DBSCAN(0.5, 2);
        int[] labels = dbscan.fitPredict(data);
        
        assertEquals(9, labels.length);
        
        // First 4 should be same cluster
        assertEquals(labels[0], labels[1]);
        assertEquals(labels[0], labels[2]);
        assertEquals(labels[0], labels[3]);
        
        // Next 4 should be same cluster but different from first
        assertEquals(labels[4], labels[5]);
        assertNotEquals(labels[0], labels[4]);
    }
    
    @Test
    void testNoiseDetection() {
        double[][] data = {
            {0, 0}, {0.1, 0}, {0, 0.1},
            {100, 100}  // isolated point
        };
        
        DBSCAN dbscan = new DBSCAN(0.5, 2);
        int[] labels = dbscan.fitPredict(data);
        
        // Last point should be noise (-1)
        assertEquals(-1, labels[3]);
    }
    
    @Test
    void testGetters() {
        DBSCAN dbscan = new DBSCAN(1.0, 5);
        double[][] data = {{0, 0}, {0.1, 0.1}, {0.2, 0.2}};
        dbscan.fit(data);
        
        assertNotNull(dbscan.getLabels());
        assertTrue(dbscan.getNClusters() >= 0);
    }
}
