package io.github.yasmramos.mindforge.preprocessing;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;
import java.util.List;

/**
 * Comprehensive tests for OneHotEncoder.
 */
class OneHotEncoderTest {
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Default constructor")
        void testDefaultConstructor() {
            OneHotEncoder encoder = new OneHotEncoder();
            assertFalse(encoder.isDropFirst());
            assertFalse(encoder.isHandleUnknown());
        }
        
        @Test
        @DisplayName("Constructor with dropFirst")
        void testDropFirstConstructor() {
            OneHotEncoder encoder = new OneHotEncoder(true);
            assertTrue(encoder.isDropFirst());
            assertFalse(encoder.isHandleUnknown());
        }
        
        @Test
        @DisplayName("Full constructor")
        void testFullConstructor() {
            OneHotEncoder encoder = new OneHotEncoder(true, true, "unknown");
            assertTrue(encoder.isDropFirst());
            assertTrue(encoder.isHandleUnknown());
        }
    }
    
    @Nested
    @DisplayName("Basic Encoding Tests")
    class BasicEncodingTests {
        
        @Test
        @DisplayName("Simple string encoding")
        void testSimpleEncoding() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] X = {{"cat"}, {"dog"}, {"cat"}, {"bird"}};
            double[][] result = encoder.fitTransform(X);
            
            // Categories sorted: [bird, cat, dog]
            assertEquals(3, result[0].length);
            
            // cat = [0, 1, 0]
            assertArrayEquals(new double[]{0, 1, 0}, result[0], 1e-10);
            // dog = [0, 0, 1]
            assertArrayEquals(new double[]{0, 0, 1}, result[1], 1e-10);
            // cat = [0, 1, 0]
            assertArrayEquals(new double[]{0, 1, 0}, result[2], 1e-10);
            // bird = [1, 0, 0]
            assertArrayEquals(new double[]{1, 0, 0}, result[3], 1e-10);
        }
        
        @Test
        @DisplayName("Multiple features encoding")
        void testMultipleFeatures() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] X = {
                {"red", "small"},
                {"blue", "large"},
                {"red", "large"}
            };
            double[][] result = encoder.fitTransform(X);
            
            // Feature 1: [blue, red] = 2 columns
            // Feature 2: [large, small] = 2 columns
            // Total: 4 columns
            assertEquals(4, result[0].length);
            
            // red, small = [0,1, 0,1]
            assertArrayEquals(new double[]{0, 1, 0, 1}, result[0], 1e-10);
            // blue, large = [1,0, 1,0]
            assertArrayEquals(new double[]{1, 0, 1, 0}, result[1], 1e-10);
            // red, large = [0,1, 1,0]
            assertArrayEquals(new double[]{0, 1, 1, 0}, result[2], 1e-10);
        }
        
        @Test
        @DisplayName("Integer encoding")
        void testIntegerEncoding() {
            OneHotEncoder encoder = new OneHotEncoder();
            int[][] X = {{0}, {1}, {2}, {1}};
            double[][] result = encoder.fitTransform(X);
            
            // Categories: [0, 1, 2]
            assertEquals(3, result[0].length);
            
            assertArrayEquals(new double[]{1, 0, 0}, result[0], 1e-10);
            assertArrayEquals(new double[]{0, 1, 0}, result[1], 1e-10);
            assertArrayEquals(new double[]{0, 0, 1}, result[2], 1e-10);
            assertArrayEquals(new double[]{0, 1, 0}, result[3], 1e-10);
        }
    }
    
    @Nested
    @DisplayName("DropFirst Tests")
    class DropFirstTests {
        
        @Test
        @DisplayName("DropFirst reduces columns by 1 per feature")
        void testDropFirst() {
            OneHotEncoder encoder = new OneHotEncoder(true);
            String[][] X = {{"cat"}, {"dog"}, {"bird"}};
            double[][] result = encoder.fitTransform(X);
            
            // 3 categories but dropFirst, so 2 columns
            assertEquals(2, result[0].length);
        }
        
        @Test
        @DisplayName("DropFirst encoding values")
        void testDropFirstValues() {
            OneHotEncoder encoder = new OneHotEncoder(true);
            String[][] X = {{"a"}, {"b"}, {"c"}};
            double[][] result = encoder.fitTransform(X);
            
            // Categories sorted: [a, b, c], drop first 'a'
            // a = [0, 0] (first dropped)
            // b = [1, 0]
            // c = [0, 1]
            assertArrayEquals(new double[]{0, 0}, result[0], 1e-10);
            assertArrayEquals(new double[]{1, 0}, result[1], 1e-10);
            assertArrayEquals(new double[]{0, 1}, result[2], 1e-10);
        }
    }
    
    @Nested
    @DisplayName("Unknown Categories Tests")
    class UnknownCategoriesTests {
        
        @Test
        @DisplayName("Unknown category throws exception by default")
        void testUnknownThrows() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] train = {{"cat"}, {"dog"}};
            String[][] test = {{"bird"}};
            
            encoder.fit(train);
            assertThrows(IllegalArgumentException.class, () -> encoder.transform(test));
        }
        
        @Test
        @DisplayName("Unknown category handled when configured")
        void testHandleUnknown() {
            OneHotEncoder encoder = new OneHotEncoder(false, true, null);
            String[][] train = {{"cat"}, {"dog"}};
            String[][] test = {{"bird"}};
            
            encoder.fit(train);
            double[][] result = encoder.transform(test);
            
            // Unknown gets all zeros
            assertArrayEquals(new double[]{0, 0}, result[0], 1e-10);
        }
    }
    
    @Nested
    @DisplayName("Inverse Transform Tests")
    class InverseTransformTests {
        
        @Test
        @DisplayName("Inverse transform returns original categories")
        void testInverseTransform() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] X = {{"cat"}, {"dog"}, {"bird"}};
            double[][] encoded = encoder.fitTransform(X);
            String[][] decoded = encoder.inverseTransform(encoded);
            
            assertArrayEquals(X[0], decoded[0]);
            assertArrayEquals(X[1], decoded[1]);
            assertArrayEquals(X[2], decoded[2]);
        }
        
        @Test
        @DisplayName("Inverse transform with dropFirst")
        void testInverseTransformDropFirst() {
            OneHotEncoder encoder = new OneHotEncoder(true);
            String[][] X = {{"a"}, {"b"}, {"c"}};
            double[][] encoded = encoder.fitTransform(X);
            String[][] decoded = encoder.inverseTransform(encoded);
            
            assertArrayEquals(X[0], decoded[0]);
            assertArrayEquals(X[1], decoded[1]);
            assertArrayEquals(X[2], decoded[2]);
        }
        
        @Test
        @DisplayName("Inverse transform with multiple features")
        void testInverseTransformMultipleFeatures() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] X = {
                {"red", "small"},
                {"blue", "large"}
            };
            double[][] encoded = encoder.fitTransform(X);
            String[][] decoded = encoder.inverseTransform(encoded);
            
            assertArrayEquals(X[0], decoded[0]);
            assertArrayEquals(X[1], decoded[1]);
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Single category per feature")
        void testSingleCategory() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] X = {{"cat"}, {"cat"}, {"cat"}};
            double[][] result = encoder.fitTransform(X);
            
            assertEquals(1, result[0].length);
            assertArrayEquals(new double[]{1}, result[0], 1e-10);
        }
        
        @Test
        @DisplayName("Null value in data")
        void testNullValue() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] X = {{"cat"}, {null}, {"dog"}};
            double[][] result = encoder.fitTransform(X);
            
            // null gets all zeros
            assertEquals(2, result[0].length);
            assertArrayEquals(new double[]{0, 0}, result[1], 1e-10);
        }
        
        @Test
        @DisplayName("Empty input throws exception")
        void testEmptyInput() {
            OneHotEncoder encoder = new OneHotEncoder();
            assertThrows(IllegalArgumentException.class, () -> encoder.fit(new String[0][]));
        }
        
        @Test
        @DisplayName("Null input throws exception")
        void testNullInput() {
            OneHotEncoder encoder = new OneHotEncoder();
            assertThrows(IllegalArgumentException.class, () -> encoder.fit((String[][]) null));
        }
        
        @Test
        @DisplayName("Empty features throws exception")
        void testEmptyFeatures() {
            OneHotEncoder encoder = new OneHotEncoder();
            assertThrows(IllegalArgumentException.class, () -> encoder.fit(new String[][]{{}}));
        }
    }
    
    @Nested
    @DisplayName("State Tests")
    class StateTests {
        
        @Test
        @DisplayName("isFitted returns correct state")
        void testIsFitted() {
            OneHotEncoder encoder = new OneHotEncoder();
            assertFalse(encoder.isFitted());
            
            encoder.fit(new String[][]{{"a"}});
            assertTrue(encoder.isFitted());
        }
        
        @Test
        @DisplayName("Transform before fit throws exception")
        void testTransformBeforeFit() {
            OneHotEncoder encoder = new OneHotEncoder();
            assertThrows(IllegalStateException.class, 
                () -> encoder.transform(new String[][]{{"a"}}));
        }
        
        @Test
        @DisplayName("InverseTransform before fit throws exception")
        void testInverseTransformBeforeFit() {
            OneHotEncoder encoder = new OneHotEncoder();
            assertThrows(IllegalStateException.class, 
                () -> encoder.inverseTransform(new double[][]{{1}}));
        }
        
        @Test
        @DisplayName("getCategories before fit throws exception")
        void testGetCategoriesBeforeFit() {
            OneHotEncoder encoder = new OneHotEncoder();
            assertThrows(IllegalStateException.class, encoder::getCategories);
        }
        
        @Test
        @DisplayName("getNFeatures before fit throws exception")
        void testGetNFeaturesBeforeFit() {
            OneHotEncoder encoder = new OneHotEncoder();
            assertThrows(IllegalStateException.class, encoder::getNFeatures);
        }
        
        @Test
        @DisplayName("getNOutputFeatures before fit throws exception")
        void testGetNOutputFeaturesBeforeFit() {
            OneHotEncoder encoder = new OneHotEncoder();
            assertThrows(IllegalStateException.class, encoder::getNOutputFeatures);
        }
        
        @Test
        @DisplayName("Dimension mismatch in transform throws exception")
        void testDimensionMismatch() {
            OneHotEncoder encoder = new OneHotEncoder();
            encoder.fit(new String[][]{{"a", "b"}});
            
            assertThrows(IllegalArgumentException.class, 
                () -> encoder.transform(new String[][]{{"a"}}));
        }
        
        @Test
        @DisplayName("Dimension mismatch in inverse transform throws exception")
        void testInverseDimensionMismatch() {
            OneHotEncoder encoder = new OneHotEncoder();
            encoder.fit(new String[][]{{"a"}, {"b"}});
            
            assertThrows(IllegalArgumentException.class, 
                () -> encoder.inverseTransform(new double[][]{{1, 0, 0}}));
        }
    }
    
    @Nested
    @DisplayName("Categories Access Tests")
    class CategoriesAccessTests {
        
        @Test
        @DisplayName("getCategories returns correct categories")
        void testGetCategories() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] X = {{"dog"}, {"cat"}, {"bird"}};
            encoder.fit(X);
            
            List<String[]> categories = encoder.getCategories();
            assertEquals(1, categories.size());
            assertArrayEquals(new String[]{"bird", "cat", "dog"}, categories.get(0));
        }
        
        @Test
        @DisplayName("getNFeatures returns correct count")
        void testGetNFeatures() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] X = {{"a", "b", "c"}};
            encoder.fit(X);
            
            assertEquals(3, encoder.getNFeatures());
        }
        
        @Test
        @DisplayName("getNOutputFeatures returns correct count")
        void testGetNOutputFeatures() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] X = {{"a", "x"}, {"b", "y"}, {"c", "z"}};
            encoder.fit(X);
            
            // Feature 1: 3 categories, Feature 2: 3 categories = 6
            assertEquals(6, encoder.getNOutputFeatures());
        }
        
        @Test
        @DisplayName("getNOutputFeatures with dropFirst")
        void testGetNOutputFeaturesDropFirst() {
            OneHotEncoder encoder = new OneHotEncoder(true);
            String[][] X = {{"a", "x"}, {"b", "y"}, {"c", "z"}};
            encoder.fit(X);
            
            // Feature 1: 3-1=2, Feature 2: 3-1=2 = 4
            assertEquals(4, encoder.getNOutputFeatures());
        }
    }
    
    @Nested
    @DisplayName("Serialization Tests")
    class SerializationTests {
        
        @Test
        @DisplayName("Serialization and deserialization works")
        void testSerialization() throws IOException, ClassNotFoundException {
            OneHotEncoder encoder = new OneHotEncoder(true, true, "unknown");
            String[][] X = {{"cat"}, {"dog"}, {"bird"}};
            encoder.fit(X);
            
            // Serialize
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            oos.writeObject(encoder);
            oos.close();
            
            // Deserialize
            ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bis);
            OneHotEncoder restored = (OneHotEncoder) ois.readObject();
            ois.close();
            
            // Verify
            assertEquals(encoder.isDropFirst(), restored.isDropFirst());
            assertEquals(encoder.isHandleUnknown(), restored.isHandleUnknown());
            assertTrue(restored.isFitted());
            
            // Test transform produces same results
            String[][] test = {{"cat"}, {"bird"}};
            double[][] expected = encoder.transform(test);
            double[][] actual = restored.transform(test);
            
            assertArrayEquals(expected[0], actual[0], 1e-10);
            assertArrayEquals(expected[1], actual[1], 1e-10);
        }
    }
    
    @Nested
    @DisplayName("Special Character Tests")
    class SpecialCharacterTests {
        
        @Test
        @DisplayName("Handles special characters in categories")
        void testSpecialCharacters() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] X = {{"cat@1"}, {"dog#2"}, {"bird$3"}};
            double[][] result = encoder.fitTransform(X);
            
            assertEquals(3, result[0].length);
        }
        
        @Test
        @DisplayName("Handles whitespace in categories")
        void testWhitespace() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] X = {{"hello world"}, {"test value"}, {"hello world"}};
            double[][] result = encoder.fitTransform(X);
            
            assertEquals(2, result[0].length);
            assertArrayEquals(result[0], result[2], 1e-10);
        }
        
        @Test
        @DisplayName("Handles empty string category")
        void testEmptyString() {
            OneHotEncoder encoder = new OneHotEncoder();
            String[][] X = {{""}, {"a"}, {""}};
            double[][] result = encoder.fitTransform(X);
            
            assertEquals(2, result[0].length);
            assertArrayEquals(result[0], result[2], 1e-10);
        }
    }
}
