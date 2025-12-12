package io.github.yasmramos.mindforge.preprocessing;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;

/**
 * Comprehensive tests for PolynomialFeatures transformer.
 */
class PolynomialFeaturesTest {
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Default constructor with degree 2")
        void testDefaultDegree() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            assertEquals(2, poly.getDegree());
            assertTrue(poly.isIncludeBias());
            assertFalse(poly.isInteractionOnly());
        }
        
        @Test
        @DisplayName("Constructor with all parameters")
        void testFullConstructor() {
            PolynomialFeatures poly = new PolynomialFeatures(3, false, true);
            assertEquals(3, poly.getDegree());
            assertFalse(poly.isIncludeBias());
            assertTrue(poly.isInteractionOnly());
        }
        
        @Test
        @DisplayName("Invalid degree throws exception")
        void testInvalidDegree() {
            assertThrows(IllegalArgumentException.class, () -> new PolynomialFeatures(0));
            assertThrows(IllegalArgumentException.class, () -> new PolynomialFeatures(-1));
        }
    }
    
    @Nested
    @DisplayName("Basic Transform Tests")
    class BasicTransformTests {
        
        @Test
        @DisplayName("Degree 2 with single feature")
        void testDegree2SingleFeature() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            double[][] X = {{2}, {3}};
            double[][] result = poly.fitTransform(X);
            
            // Expected: [1, x, x^2] = [1, 2, 4] and [1, 3, 9]
            assertEquals(3, result[0].length);
            assertArrayEquals(new double[]{1, 2, 4}, result[0], 1e-10);
            assertArrayEquals(new double[]{1, 3, 9}, result[1], 1e-10);
        }
        
        @Test
        @DisplayName("Degree 2 with two features")
        void testDegree2TwoFeatures() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            double[][] X = {{2, 3}};
            double[][] result = poly.fitTransform(X);
            
            // Expected: [1, b, a, b², ab, a²] = [1, 3, 2, 9, 6, 4]
            assertEquals(6, result[0].length);
            assertArrayEquals(new double[]{1, 3, 2, 9, 6, 4}, result[0], 1e-10);
        }
        
        @Test
        @DisplayName("Degree 3 with single feature")
        void testDegree3SingleFeature() {
            PolynomialFeatures poly = new PolynomialFeatures(3);
            double[][] X = {{2}};
            double[][] result = poly.fitTransform(X);
            
            // Expected: [1, x, x^2, x^3] = [1, 2, 4, 8]
            assertEquals(4, result[0].length);
            assertArrayEquals(new double[]{1, 2, 4, 8}, result[0], 1e-10);
        }
        
        @Test
        @DisplayName("Degree 1 returns original with bias")
        void testDegree1() {
            PolynomialFeatures poly = new PolynomialFeatures(1);
            double[][] X = {{2, 3}, {4, 5}};
            double[][] result = poly.fitTransform(X);
            
            // Expected: [1, b, a] = [1, 3, 2]
            assertEquals(3, result[0].length);
            assertArrayEquals(new double[]{1, 3, 2}, result[0], 1e-10);
            assertArrayEquals(new double[]{1, 5, 4}, result[1], 1e-10);
        }
    }
    
    @Nested
    @DisplayName("Options Tests")
    class OptionsTests {
        
        @Test
        @DisplayName("Without bias column")
        void testWithoutBias() {
            PolynomialFeatures poly = new PolynomialFeatures(2, false, false);
            double[][] X = {{2, 3}};
            double[][] result = poly.fitTransform(X);
            
            // Expected: [b, a, b², ab, a²] = [3, 2, 9, 6, 4]
            assertEquals(5, result[0].length);
            assertArrayEquals(new double[]{3, 2, 9, 6, 4}, result[0], 1e-10);
        }
        
        @Test
        @DisplayName("Interaction only mode")
        void testInteractionOnly() {
            PolynomialFeatures poly = new PolynomialFeatures(2, true, true);
            double[][] X = {{2, 3}};
            double[][] result = poly.fitTransform(X);
            
            // Expected: [1, b, a, ab] = [1, 3, 2, 6] (no a², b²)
            assertEquals(4, result[0].length);
            assertArrayEquals(new double[]{1, 3, 2, 6}, result[0], 1e-10);
        }
        
        @Test
        @DisplayName("Interaction only without bias")
        void testInteractionOnlyNoBias() {
            PolynomialFeatures poly = new PolynomialFeatures(2, false, true);
            double[][] X = {{2, 3}};
            double[][] result = poly.fitTransform(X);
            
            // Expected: [b, a, ab] = [3, 2, 6]
            assertEquals(3, result[0].length);
            assertArrayEquals(new double[]{3, 2, 6}, result[0], 1e-10);
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Zero values")
        void testZeroValues() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            double[][] X = {{0, 1}};
            double[][] result = poly.fitTransform(X);
            
            // [1, b, a, b², ab, a²] = [1, 1, 0, 1, 0, 0]
            assertArrayEquals(new double[]{1, 1, 0, 1, 0, 0}, result[0], 1e-10);
        }
        
        @Test
        @DisplayName("Negative values")
        void testNegativeValues() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            double[][] X = {{-2}};
            double[][] result = poly.fitTransform(X);
            
            // [1, -2, 4]
            assertArrayEquals(new double[]{1, -2, 4}, result[0], 1e-10);
        }
        
        @Test
        @DisplayName("Large values")
        void testLargeValues() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            double[][] X = {{1000}};
            double[][] result = poly.fitTransform(X);
            
            // [1, 1000, 1000000]
            assertArrayEquals(new double[]{1, 1000, 1000000}, result[0], 1e-10);
        }
        
        @Test
        @DisplayName("Small decimal values")
        void testSmallValues() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            double[][] X = {{0.1}};
            double[][] result = poly.fitTransform(X);
            
            // [1, 0.1, 0.01]
            assertArrayEquals(new double[]{1, 0.1, 0.01}, result[0], 1e-10);
        }
        
        @Test
        @DisplayName("Null input throws exception")
        void testNullInput() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            assertThrows(IllegalArgumentException.class, () -> poly.fit(null));
        }
        
        @Test
        @DisplayName("Empty input throws exception")
        void testEmptyInput() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            assertThrows(IllegalArgumentException.class, () -> poly.fit(new double[0][]));
        }
        
        @Test
        @DisplayName("Empty features throws exception")
        void testEmptyFeatures() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            assertThrows(IllegalArgumentException.class, () -> poly.fit(new double[][]{{}}));
        }
    }
    
    @Nested
    @DisplayName("State Tests")
    class StateTests {
        
        @Test
        @DisplayName("isFitted returns correct state")
        void testIsFitted() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            assertFalse(poly.isFitted());
            
            poly.fit(new double[][]{{1, 2}});
            assertTrue(poly.isFitted());
        }
        
        @Test
        @DisplayName("Transform before fit throws exception")
        void testTransformBeforeFit() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            assertThrows(IllegalStateException.class, () -> poly.transform(new double[][]{{1}}));
        }
        
        @Test
        @DisplayName("getNInputFeatures before fit throws exception")
        void testGetNInputFeaturesBeforeFit() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            assertThrows(IllegalStateException.class, poly::getNInputFeatures);
        }
        
        @Test
        @DisplayName("getNOutputFeatures before fit throws exception")
        void testGetNOutputFeaturesBeforeFit() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            assertThrows(IllegalStateException.class, poly::getNOutputFeatures);
        }
        
        @Test
        @DisplayName("Dimension mismatch in transform throws exception")
        void testDimensionMismatch() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            poly.fit(new double[][]{{1, 2}});
            
            assertThrows(IllegalArgumentException.class, 
                () -> poly.transform(new double[][]{{1, 2, 3}}));
        }
        
        @Test
        @DisplayName("getNInputFeatures returns correct value")
        void testGetNInputFeatures() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            poly.fit(new double[][]{{1, 2, 3}});
            assertEquals(3, poly.getNInputFeatures());
        }
        
        @Test
        @DisplayName("getNOutputFeatures returns correct value")
        void testGetNOutputFeatures() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            poly.fit(new double[][]{{1, 2}});
            // For 2 features, degree 2, with bias: 1 + 2 + 3 = 6
            assertEquals(6, poly.getNOutputFeatures());
        }
    }
    
    @Nested
    @DisplayName("Serialization Tests")
    class SerializationTests {
        
        @Test
        @DisplayName("Serialization and deserialization works")
        void testSerialization() throws IOException, ClassNotFoundException {
            PolynomialFeatures poly = new PolynomialFeatures(3, false, true);
            poly.fit(new double[][]{{1, 2}, {3, 4}});
            
            // Serialize
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            oos.writeObject(poly);
            oos.close();
            
            // Deserialize
            ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bis);
            PolynomialFeatures restored = (PolynomialFeatures) ois.readObject();
            ois.close();
            
            // Verify
            assertEquals(poly.getDegree(), restored.getDegree());
            assertEquals(poly.isIncludeBias(), restored.isIncludeBias());
            assertEquals(poly.isInteractionOnly(), restored.isInteractionOnly());
            assertTrue(restored.isFitted());
            
            // Test transform produces same results
            double[][] X = {{5, 6}};
            assertArrayEquals(poly.transform(X)[0], restored.transform(X)[0], 1e-10);
        }
    }
    
    @Nested
    @DisplayName("Multiple Samples Tests")
    class MultipleSamplesTests {
        
        @Test
        @DisplayName("Multiple samples transformed correctly")
        void testMultipleSamples() {
            PolynomialFeatures poly = new PolynomialFeatures(2);
            double[][] X = {
                {1, 2},
                {3, 4},
                {5, 6}
            };
            double[][] result = poly.fitTransform(X);
            
            assertEquals(3, result.length);
            assertEquals(6, result[0].length);
            
            // [1, b, a, b², ab, a²]
            // First sample: [1, 2, 1, 4, 2, 1]
            assertArrayEquals(new double[]{1, 2, 1, 4, 2, 1}, result[0], 1e-10);
            // Second sample: [1, 4, 3, 16, 12, 9]
            assertArrayEquals(new double[]{1, 4, 3, 16, 12, 9}, result[1], 1e-10);
            // Third sample: [1, 6, 5, 36, 30, 25]
            assertArrayEquals(new double[]{1, 6, 5, 36, 30, 25}, result[2], 1e-10);
        }
    }
    
    @Nested
    @DisplayName("High Degree Tests")
    class HighDegreeTests {
        
        @Test
        @DisplayName("Degree 4 single feature")
        void testDegree4() {
            PolynomialFeatures poly = new PolynomialFeatures(4, false, false);
            double[][] X = {{2}};
            double[][] result = poly.fitTransform(X);
            
            // [2, 4, 8, 16]
            assertEquals(4, result[0].length);
            assertArrayEquals(new double[]{2, 4, 8, 16}, result[0], 1e-10);
        }
        
        @Test
        @DisplayName("Degree 3 two features")
        void testDegree3TwoFeatures() {
            PolynomialFeatures poly = new PolynomialFeatures(3, true, false);
            double[][] X = {{1, 2}};
            double[][] result = poly.fitTransform(X);
            
            // 1 (bias) + 2 (degree 1) + 3 (degree 2) + 4 (degree 3) = 10 features
            assertEquals(10, result[0].length);
            
            // Verify bias
            assertEquals(1.0, result[0][0], 1e-10);
        }
    }
}
