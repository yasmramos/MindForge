package io.github.yasmramos.mindforge.preprocessing;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Additional coverage tests for MinMaxScaler.
 */
class MinMaxScalerCoverageTest {
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Default constructor creates [0,1] range")
        void testDefaultConstructor() {
            MinMaxScaler scaler = new MinMaxScaler();
            assertFalse(scaler.isFitted());
        }
        
        @Test
        @DisplayName("Custom range constructor")
        void testCustomRangeConstructor() {
            MinMaxScaler scaler = new MinMaxScaler(-1.0, 1.0);
            assertFalse(scaler.isFitted());
        }
        
        @Test
        @DisplayName("Invalid range - min >= max throws exception")
        void testInvalidRangeMinEqualsMax() {
            assertThrows(IllegalArgumentException.class, () -> new MinMaxScaler(1.0, 1.0));
        }
        
        @Test
        @DisplayName("Invalid range - min > max throws exception")
        void testInvalidRangeMinGreaterMax() {
            assertThrows(IllegalArgumentException.class, () -> new MinMaxScaler(2.0, 1.0));
        }
    }
    
    @Nested
    @DisplayName("Fit Tests")
    class FitTests {
        
        @Test
        @DisplayName("Fit with null data throws exception")
        void testFitNullData() {
            MinMaxScaler scaler = new MinMaxScaler();
            assertThrows(IllegalArgumentException.class, () -> scaler.fit(null));
        }
        
        @Test
        @DisplayName("Fit with empty data throws exception")
        void testFitEmptyData() {
            MinMaxScaler scaler = new MinMaxScaler();
            assertThrows(IllegalArgumentException.class, () -> scaler.fit(new double[0][]));
        }
        
        @Test
        @DisplayName("Fit with inconsistent row lengths throws exception")
        void testFitInconsistentRowLengths() {
            MinMaxScaler scaler = new MinMaxScaler();
            double[][] X = {{1.0, 2.0}, {3.0}};
            assertThrows(IllegalArgumentException.class, () -> scaler.fit(X));
        }
        
        @Test
        @DisplayName("Fit sets isFitted to true")
        void testFitSetsIsFitted() {
            MinMaxScaler scaler = new MinMaxScaler();
            double[][] X = {{1.0}, {2.0}};
            scaler.fit(X);
            assertTrue(scaler.isFitted());
        }
    }
    
    @Nested
    @DisplayName("Transform Tests")
    class TransformTests {
        
        @Test
        @DisplayName("Transform before fit throws exception")
        void testTransformBeforeFit() {
            MinMaxScaler scaler = new MinMaxScaler();
            assertThrows(IllegalStateException.class, 
                () -> scaler.transform(new double[][]{{1.0}}));
        }
        
        @Test
        @DisplayName("Transform with null data throws exception")
        void testTransformNullData() {
            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(new double[][]{{1.0}, {2.0}});
            assertThrows(IllegalArgumentException.class, () -> scaler.transform(null));
        }
        
        @Test
        @DisplayName("Transform with empty data throws exception")
        void testTransformEmptyData() {
            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(new double[][]{{1.0}, {2.0}});
            assertThrows(IllegalArgumentException.class, 
                () -> scaler.transform(new double[0][]));
        }
        
        @Test
        @DisplayName("Transform with wrong number of features throws exception")
        void testTransformWrongFeatures() {
            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
            assertThrows(IllegalArgumentException.class, 
                () -> scaler.transform(new double[][]{{1.0}}));
        }
        
        @Test
        @DisplayName("Transform with zero range uses middle value")
        void testTransformZeroRange() {
            MinMaxScaler scaler = new MinMaxScaler();
            // All values are the same
            double[][] X = {{5.0}, {5.0}, {5.0}};
            scaler.fit(X);
            
            double[][] scaled = scaler.transform(X);
            
            // When range is 0, should scale to middle of target range (0.5)
            for (double[] row : scaled) {
                assertEquals(0.5, row[0], 1e-10);
            }
        }
        
        @Test
        @DisplayName("Transform to custom range [-1, 1]")
        void testTransformCustomRange() {
            MinMaxScaler scaler = new MinMaxScaler(-1.0, 1.0);
            double[][] X = {{0}, {5}, {10}};
            double[][] scaled = scaler.fitTransform(X);
            
            assertEquals(-1.0, scaled[0][0], 1e-10);
            assertEquals(0.0, scaled[1][0], 1e-10);
            assertEquals(1.0, scaled[2][0], 1e-10);
        }
    }
    
    @Nested
    @DisplayName("Inverse Transform Tests")
    class InverseTransformTests {
        
        @Test
        @DisplayName("Inverse transform before fit throws exception")
        void testInverseTransformBeforeFit() {
            MinMaxScaler scaler = new MinMaxScaler();
            assertThrows(IllegalStateException.class, 
                () -> scaler.inverseTransform(new double[][]{{0.5}}));
        }
        
        @Test
        @DisplayName("Inverse transform with null data throws exception")
        void testInverseTransformNullData() {
            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(new double[][]{{1.0}, {2.0}});
            assertThrows(IllegalArgumentException.class, 
                () -> scaler.inverseTransform(null));
        }
        
        @Test
        @DisplayName("Inverse transform with empty data throws exception")
        void testInverseTransformEmptyData() {
            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(new double[][]{{1.0}, {2.0}});
            assertThrows(IllegalArgumentException.class, 
                () -> scaler.inverseTransform(new double[0][]));
        }
        
        @Test
        @DisplayName("Inverse transform with zero range")
        void testInverseTransformZeroRange() {
            MinMaxScaler scaler = new MinMaxScaler();
            double[][] X = {{5.0}, {5.0}};
            scaler.fit(X);
            
            double[][] scaled = scaler.transform(X);
            double[][] inverse = scaler.inverseTransform(scaled);
            
            // Should return the original constant value
            for (double[] row : inverse) {
                assertEquals(5.0, row[0], 1e-10);
            }
        }
        
        @Test
        @DisplayName("Roundtrip: transform then inverse transform")
        void testRoundtrip() {
            MinMaxScaler scaler = new MinMaxScaler();
            double[][] X = {{1.0, 10.0}, {2.0, 20.0}, {3.0, 30.0}};
            
            double[][] scaled = scaler.fitTransform(X);
            double[][] restored = scaler.inverseTransform(scaled);
            
            for (int i = 0; i < X.length; i++) {
                for (int j = 0; j < X[0].length; j++) {
                    assertEquals(X[i][j], restored[i][j], 1e-10);
                }
            }
        }
        
        @Test
        @DisplayName("Roundtrip with custom range")
        void testRoundtripCustomRange() {
            MinMaxScaler scaler = new MinMaxScaler(-10.0, 10.0);
            double[][] X = {{0}, {50}, {100}};
            
            double[][] scaled = scaler.fitTransform(X);
            double[][] restored = scaler.inverseTransform(scaled);
            
            for (int i = 0; i < X.length; i++) {
                assertEquals(X[i][0], restored[i][0], 1e-10);
            }
        }
    }
    
    @Nested
    @DisplayName("Getter Tests")
    class GetterTests {
        
        @Test
        @DisplayName("getFeatureMin before fit throws exception")
        void testGetFeatureMinBeforeFit() {
            MinMaxScaler scaler = new MinMaxScaler();
            assertThrows(IllegalStateException.class, scaler::getFeatureMin);
        }
        
        @Test
        @DisplayName("getFeatureMax before fit throws exception")
        void testGetFeatureMaxBeforeFit() {
            MinMaxScaler scaler = new MinMaxScaler();
            assertThrows(IllegalStateException.class, scaler::getFeatureMax);
        }
        
        @Test
        @DisplayName("getFeatureMin returns correct values")
        void testGetFeatureMin() {
            MinMaxScaler scaler = new MinMaxScaler();
            double[][] X = {{1.0, 100.0}, {5.0, 200.0}, {3.0, 150.0}};
            scaler.fit(X);
            
            double[] min = scaler.getFeatureMin();
            assertEquals(1.0, min[0], 1e-10);
            assertEquals(100.0, min[1], 1e-10);
        }
        
        @Test
        @DisplayName("getFeatureMax returns correct values")
        void testGetFeatureMax() {
            MinMaxScaler scaler = new MinMaxScaler();
            double[][] X = {{1.0, 100.0}, {5.0, 200.0}, {3.0, 150.0}};
            scaler.fit(X);
            
            double[] max = scaler.getFeatureMax();
            assertEquals(5.0, max[0], 1e-10);
            assertEquals(200.0, max[1], 1e-10);
        }
        
        @Test
        @DisplayName("getFeatureMin returns clone")
        void testGetFeatureMinReturnsClone() {
            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(new double[][]{{1.0}, {5.0}});
            
            double[] min1 = scaler.getFeatureMin();
            double[] min2 = scaler.getFeatureMin();
            
            assertNotSame(min1, min2);
        }
        
        @Test
        @DisplayName("getFeatureMax returns clone")
        void testGetFeatureMaxReturnsClone() {
            MinMaxScaler scaler = new MinMaxScaler();
            scaler.fit(new double[][]{{1.0}, {5.0}});
            
            double[] max1 = scaler.getFeatureMax();
            double[] max2 = scaler.getFeatureMax();
            
            assertNotSame(max1, max2);
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Single sample")
        void testSingleSample() {
            MinMaxScaler scaler = new MinMaxScaler();
            double[][] X = {{5.0, 10.0}};
            double[][] scaled = scaler.fitTransform(X);
            
            // Single sample means all values are the same, should be middle
            assertEquals(0.5, scaled[0][0], 1e-10);
            assertEquals(0.5, scaled[0][1], 1e-10);
        }
        
        @Test
        @DisplayName("Negative values")
        void testNegativeValues() {
            MinMaxScaler scaler = new MinMaxScaler();
            double[][] X = {{-10}, {0}, {10}};
            double[][] scaled = scaler.fitTransform(X);
            
            assertEquals(0.0, scaled[0][0], 1e-10);
            assertEquals(0.5, scaled[1][0], 1e-10);
            assertEquals(1.0, scaled[2][0], 1e-10);
        }
        
        @Test
        @DisplayName("Mixed positive and negative")
        void testMixedValues() {
            MinMaxScaler scaler = new MinMaxScaler();
            double[][] X = {{-100, 50}, {0, 100}, {100, 150}};
            double[][] scaled = scaler.fitTransform(X);
            
            // Check first column: -100 to 100
            assertEquals(0.0, scaled[0][0], 1e-10);
            assertEquals(0.5, scaled[1][0], 1e-10);
            assertEquals(1.0, scaled[2][0], 1e-10);
            
            // Check second column: 50 to 150
            assertEquals(0.0, scaled[0][1], 1e-10);
            assertEquals(0.5, scaled[1][1], 1e-10);
            assertEquals(1.0, scaled[2][1], 1e-10);
        }
        
        @Test
        @DisplayName("Values outside training range")
        void testValuesOutsideRange() {
            MinMaxScaler scaler = new MinMaxScaler();
            double[][] XTrain = {{0}, {10}};
            scaler.fit(XTrain);
            
            double[][] XTest = {{-5}, {5}, {15}};
            double[][] scaled = scaler.transform(XTest);
            
            assertEquals(-0.5, scaled[0][0], 1e-10); // Below 0
            assertEquals(0.5, scaled[1][0], 1e-10);  // Middle
            assertEquals(1.5, scaled[2][0], 1e-10);  // Above 1
        }
    }
}
