package com.mindforge.math;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Distance Metrics Tests")
class DistanceTest {
    
    @Nested
    @DisplayName("Euclidean Distance Tests")
    class EuclideanTests {
        
        @Test
        @DisplayName("Should calculate 3-4-5 triangle correctly")
        void testEuclideanDistance345() {
            double[] a = {0.0, 0.0};
            double[] b = {3.0, 4.0};
            
            double distance = Distance.euclidean(a, b);
            assertEquals(5.0, distance, 0.001, "Euclidean distance should be 5.0");
        }
        
        @Test
        @DisplayName("Should return 0 for identical points")
        void testEuclideanIdenticalPoints() {
            double[] a = {3.0, 4.0, 5.0};
            double[] b = {3.0, 4.0, 5.0};
            
            double distance = Distance.euclidean(a, b);
            assertEquals(0.0, distance, 0.001, "Distance between identical points should be 0");
        }
        
        @Test
        @DisplayName("Should handle negative values")
        void testEuclideanNegativeValues() {
            double[] a = {-1.0, -1.0};
            double[] b = {1.0, 1.0};
            
            double distance = Distance.euclidean(a, b);
            assertEquals(Math.sqrt(8), distance, 0.001, "Should handle negative values correctly");
        }
        
        @Test
        @DisplayName("Should handle single dimension")
        void testEuclideanSingleDimension() {
            double[] a = {0.0};
            double[] b = {5.0};
            
            double distance = Distance.euclidean(a, b);
            assertEquals(5.0, distance, 0.001, "1D Euclidean distance should be absolute difference");
        }
        
        @Test
        @DisplayName("Should throw exception for different length arrays")
        void testEuclideanDifferentLengthArrays() {
            double[] a = {1.0, 2.0};
            double[] b = {1.0, 2.0, 3.0};
            
            assertThrows(IllegalArgumentException.class, () -> {
                Distance.euclidean(a, b);
            }, "Should throw exception for different length arrays");
        }
    }
    
    @Nested
    @DisplayName("Manhattan Distance Tests")
    class ManhattanTests {
        
        @Test
        @DisplayName("Should calculate Manhattan distance correctly")
        void testManhattanDistance() {
            double[] a = {0.0, 0.0};
            double[] b = {3.0, 4.0};
            
            double distance = Distance.manhattan(a, b);
            assertEquals(7.0, distance, 0.001, "Manhattan distance should be 7.0");
        }
        
        @Test
        @DisplayName("Should return 0 for identical points")
        void testManhattanIdenticalPoints() {
            double[] a = {3.0, 4.0};
            double[] b = {3.0, 4.0};
            
            double distance = Distance.manhattan(a, b);
            assertEquals(0.0, distance, 0.001, "Distance between identical points should be 0");
        }
        
        @Test
        @DisplayName("Should handle negative values")
        void testManhattanNegativeValues() {
            double[] a = {-2.0, -3.0};
            double[] b = {2.0, 3.0};
            
            double distance = Distance.manhattan(a, b);
            assertEquals(10.0, distance, 0.001, "Should handle negative values correctly");
        }
        
        @Test
        @DisplayName("Should throw exception for different length arrays")
        void testManhattanDifferentLengthArrays() {
            double[] a = {1.0};
            double[] b = {1.0, 2.0};
            
            assertThrows(IllegalArgumentException.class, () -> {
                Distance.manhattan(a, b);
            }, "Should throw exception for different length arrays");
        }
    }
    
    @Nested
    @DisplayName("Cosine Distance Tests")
    class CosineTests {
        
        @Test
        @DisplayName("Should return 0 for identical vectors")
        void testCosineIdenticalVectors() {
            double[] a = {1.0, 0.0};
            double[] b = {1.0, 0.0};
            
            double distance = Distance.cosine(a, b);
            assertEquals(0.0, distance, 0.001, "Identical vectors should have 0 cosine distance");
        }
        
        @Test
        @DisplayName("Should return 1 for orthogonal vectors")
        void testCosineOrthogonalVectors() {
            double[] a = {1.0, 0.0};
            double[] b = {0.0, 1.0};
            
            double distance = Distance.cosine(a, b);
            assertEquals(1.0, distance, 0.001, "Orthogonal vectors should have distance 1.0");
        }
        
        @Test
        @DisplayName("Should return 2 for opposite vectors")
        void testCosineOppositeVectors() {
            double[] a = {1.0, 0.0};
            double[] b = {-1.0, 0.0};
            
            double distance = Distance.cosine(a, b);
            assertEquals(2.0, distance, 0.001, "Opposite vectors should have distance 2.0");
        }
        
        @Test
        @DisplayName("Should return 1 for zero vector")
        void testCosineZeroVector() {
            double[] a = {0.0, 0.0};
            double[] b = {1.0, 1.0};
            
            double distance = Distance.cosine(a, b);
            assertEquals(1.0, distance, 0.001, "Zero vector should have distance 1.0");
        }
        
        @Test
        @DisplayName("Should return 1 for both zero vectors")
        void testCosineBothZeroVectors() {
            double[] a = {0.0, 0.0};
            double[] b = {0.0, 0.0};
            
            double distance = Distance.cosine(a, b);
            assertEquals(1.0, distance, 0.001, "Both zero vectors should have distance 1.0");
        }
        
        @Test
        @DisplayName("Should handle parallel vectors with different magnitudes")
        void testCosineParallelDifferentMagnitudes() {
            double[] a = {1.0, 2.0};
            double[] b = {2.0, 4.0};
            
            double distance = Distance.cosine(a, b);
            assertEquals(0.0, distance, 0.001, "Parallel vectors should have distance 0");
        }
        
        @Test
        @DisplayName("Should throw exception for different length arrays")
        void testCosineDifferentLengthArrays() {
            double[] a = {1.0, 2.0, 3.0};
            double[] b = {1.0, 2.0};
            
            assertThrows(IllegalArgumentException.class, () -> {
                Distance.cosine(a, b);
            }, "Should throw exception for different length arrays");
        }
    }
    
    @Nested
    @DisplayName("Minkowski Distance Tests")
    class MinkowskiTests {
        
        @Test
        @DisplayName("Should equal Manhattan when p=1")
        void testMinkowskiP1() {
            double[] a = {0.0, 0.0};
            double[] b = {3.0, 4.0};
            
            double distance = Distance.minkowski(a, b, 1.0);
            assertEquals(7.0, distance, 0.001, "Minkowski p=1 should equal Manhattan");
        }
        
        @Test
        @DisplayName("Should equal Euclidean when p=2")
        void testMinkowskiP2() {
            double[] a = {0.0, 0.0};
            double[] b = {3.0, 4.0};
            
            double distance = Distance.minkowski(a, b, 2.0);
            assertEquals(5.0, distance, 0.001, "Minkowski p=2 should equal Euclidean");
        }
        
        @Test
        @DisplayName("Should approach Chebyshev (max) when p is large")
        void testMinkowskiLargeP() {
            double[] a = {0.0, 0.0};
            double[] b = {3.0, 4.0};
            
            double distance = Distance.minkowski(a, b, 100.0);
            // For large p, Minkowski approaches max(|a_i - b_i|) = 4
            assertEquals(4.0, distance, 0.1, "Minkowski with large p should approach Chebyshev");
        }
        
        @Test
        @DisplayName("Should return 0 for identical points")
        void testMinkowskiIdenticalPoints() {
            double[] a = {1.0, 2.0, 3.0};
            double[] b = {1.0, 2.0, 3.0};
            
            double distance = Distance.minkowski(a, b, 3.0);
            assertEquals(0.0, distance, 0.001, "Distance between identical points should be 0");
        }
        
        @Test
        @DisplayName("Should handle fractional p values")
        void testMinkowskiFractionalP() {
            double[] a = {0.0, 0.0};
            double[] b = {1.0, 1.0};
            
            double distance = Distance.minkowski(a, b, 0.5);
            assertTrue(distance > 0, "Should handle fractional p values");
        }
        
        @Test
        @DisplayName("Should throw exception for p <= 0")
        void testMinkowskiInvalidP() {
            double[] a = {1.0, 2.0};
            double[] b = {3.0, 4.0};
            
            assertThrows(IllegalArgumentException.class, () -> {
                Distance.minkowski(a, b, 0.0);
            }, "Should throw exception for p=0");
            
            assertThrows(IllegalArgumentException.class, () -> {
                Distance.minkowski(a, b, -1.0);
            }, "Should throw exception for p<0");
        }
        
        @Test
        @DisplayName("Should throw exception for different length arrays")
        void testMinkowskiDifferentLengthArrays() {
            double[] a = {1.0, 2.0};
            double[] b = {1.0};
            
            assertThrows(IllegalArgumentException.class, () -> {
                Distance.minkowski(a, b, 2.0);
            }, "Should throw exception for different length arrays");
        }
    }
    
    @Nested
    @DisplayName("High Dimensional Tests")
    class HighDimensionalTests {
        
        @Test
        @DisplayName("Should handle high dimensional vectors")
        void testHighDimensional() {
            int dim = 100;
            double[] a = new double[dim];
            double[] b = new double[dim];
            
            for (int i = 0; i < dim; i++) {
                a[i] = i;
                b[i] = i + 1;
            }
            
            double euclidean = Distance.euclidean(a, b);
            double manhattan = Distance.manhattan(a, b);
            double cosine = Distance.cosine(a, b);
            double minkowski = Distance.minkowski(a, b, 2.0);
            
            assertTrue(euclidean > 0, "Euclidean should be positive");
            assertEquals(dim, manhattan, 0.001, "Manhattan should equal number of dimensions");
            assertTrue(cosine >= 0 && cosine <= 2, "Cosine should be in [0, 2]");
            assertEquals(euclidean, minkowski, 0.001, "Minkowski p=2 should equal Euclidean");
        }
    }
}
