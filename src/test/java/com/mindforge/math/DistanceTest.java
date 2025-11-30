package com.mindforge.math;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DistanceTest {
    
    @Test
    void testEuclideanDistance() {
        double[] a = {0.0, 0.0};
        double[] b = {3.0, 4.0};
        
        double distance = Distance.euclidean(a, b);
        assertEquals(5.0, distance, 0.001, "Euclidean distance should be 5.0");
    }
    
    @Test
    void testManhattanDistance() {
        double[] a = {0.0, 0.0};
        double[] b = {3.0, 4.0};
        
        double distance = Distance.manhattan(a, b);
        assertEquals(7.0, distance, 0.001, "Manhattan distance should be 7.0");
    }
    
    @Test
    void testCosineDistance() {
        double[] a = {1.0, 0.0};
        double[] b = {1.0, 0.0};
        
        double distance = Distance.cosine(a, b);
        assertEquals(0.0, distance, 0.001, "Identical vectors should have 0 cosine distance");
        
        // Orthogonal vectors
        double[] c = {1.0, 0.0};
        double[] d = {0.0, 1.0};
        distance = Distance.cosine(c, d);
        assertEquals(1.0, distance, 0.001, "Orthogonal vectors should have distance 1.0");
    }
    
    @Test
    void testMinkowskiDistance() {
        double[] a = {0.0, 0.0};
        double[] b = {3.0, 4.0};
        
        // p=1 should equal Manhattan
        double distance = Distance.minkowski(a, b, 1.0);
        assertEquals(7.0, distance, 0.001, "Minkowski p=1 should equal Manhattan");
        
        // p=2 should equal Euclidean
        distance = Distance.minkowski(a, b, 2.0);
        assertEquals(5.0, distance, 0.001, "Minkowski p=2 should equal Euclidean");
    }
    
    @Test
    void testDifferentLengthArrays() {
        double[] a = {1.0, 2.0};
        double[] b = {1.0, 2.0, 3.0};
        
        assertThrows(IllegalArgumentException.class, () -> {
            Distance.euclidean(a, b);
        }, "Should throw exception for different length arrays");
    }
}
