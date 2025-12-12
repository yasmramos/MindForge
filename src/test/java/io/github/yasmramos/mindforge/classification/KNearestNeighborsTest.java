package io.github.yasmramos.mindforge.classification;

import io.github.yasmramos.mindforge.validation.Metrics;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class KNearestNeighborsTest {
    
    @Test
    void testSimpleClassification() {
        // Simple 2D dataset with two clear clusters
        double[][] X = {
            {1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0},  // Class 0
            {6.0, 5.0}, {7.0, 8.0}, {8.0, 7.0}   // Class 1
        };
        int[] y = {0, 0, 0, 1, 1, 1};
        
        KNearestNeighbors knn = new KNearestNeighbors(3);
        knn.train(X, y);
        
        // Test prediction
        int prediction = knn.predict(new double[]{2.5, 2.5});
        assertEquals(0, prediction, "Point close to class 0 should be classified as 0");
        
        prediction = knn.predict(new double[]{7.0, 7.0});
        assertEquals(1, prediction, "Point close to class 1 should be classified as 1");
    }
    
    @Test
    void testPerfectAccuracy() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {10.0}, {11.0}, {12.0}};
        int[] y = {0, 0, 0, 1, 1, 1};
        
        KNearestNeighbors knn = new KNearestNeighbors(1);
        knn.train(X, y);
        
        int[] predictions = knn.predict(X);
        double accuracy = Metrics.accuracy(y, predictions);
        assertEquals(1.0, accuracy, 0.001, "Should have perfect accuracy on training data with k=1");
    }
    
    @Test
    void testNumClasses() {
        double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}};
        int[] y = {0, 1, 2, 1};
        
        KNearestNeighbors knn = new KNearestNeighbors(2);
        knn.train(X, y);
        
        assertEquals(3, knn.getNumClasses(), "Should detect 3 classes");
    }
    
    @Test
    void testInvalidK() {
        assertThrows(IllegalArgumentException.class, () -> {
            new KNearestNeighbors(0);
        }, "Should throw exception for k <= 0");
    }
}
