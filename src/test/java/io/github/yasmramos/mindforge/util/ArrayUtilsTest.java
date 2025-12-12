package io.github.yasmramos.mindforge.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;

@DisplayName("Array Utils Tests")
class ArrayUtilsTest {
    
    @Nested
    @DisplayName("Matrix Operations Tests")
    class MatrixOperationsTests {
        
        @Test
        @DisplayName("Should transpose matrix correctly")
        void testTranspose() {
            double[][] matrix = {
                {1, 2, 3},
                {4, 5, 6}
            };
            
            double[][] transposed = ArrayUtils.transpose(matrix);
            
            assertEquals(3, transposed.length);
            assertEquals(2, transposed[0].length);
            assertEquals(1, transposed[0][0], 0.001);
            assertEquals(4, transposed[0][1], 0.001);
            assertEquals(3, transposed[2][0], 0.001);
        }
        
        @Test
        @DisplayName("Should calculate dot product")
        void testDotProduct() {
            double[] a = {1, 2, 3};
            double[] b = {4, 5, 6};
            
            double result = ArrayUtils.dot(a, b);
            
            assertEquals(32, result, 0.001); // 1*4 + 2*5 + 3*6 = 32
        }
        
        @Test
        @DisplayName("Should flatten 2D array to 1D")
        void testFlatten() {
            double[][] matrix = {{1, 2}, {3, 4}};
            
            double[] flattened = ArrayUtils.flatten(matrix);
            
            assertEquals(4, flattened.length);
            assertArrayEquals(new double[]{1, 2, 3, 4}, flattened, 0.001);
        }
        
        @Test
        @DisplayName("Should reshape 1D array to 2D")
        void testReshape() {
            double[] array = {1, 2, 3, 4, 5, 6};
            
            double[][] reshaped = ArrayUtils.reshape(array, 2, 3);
            
            assertEquals(2, reshaped.length);
            assertEquals(3, reshaped[0].length);
            assertEquals(1, reshaped[0][0], 0.001);
            assertEquals(4, reshaped[1][0], 0.001);
        }
        
        @Test
        @DisplayName("Should clone 2D array")
        void testClone2D() {
            double[][] original = {{1, 2}, {3, 4}};
            
            double[][] cloned = ArrayUtils.clone2D(original);
            
            // Modify original
            original[0][0] = 99;
            // Cloned should be unchanged
            assertEquals(1, cloned[0][0], 0.001);
        }
    }
    
    @Nested
    @DisplayName("Statistics Tests")
    class StatisticsTests {
        
        @Test
        @DisplayName("Should calculate mean correctly")
        void testMean() {
            double[] array = {1, 2, 3, 4, 5};
            assertEquals(3.0, ArrayUtils.mean(array), 0.001);
        }
        
        @Test
        @DisplayName("Should calculate sum correctly")
        void testSum() {
            double[] array = {1, 2, 3, 4, 5};
            assertEquals(15.0, ArrayUtils.sum(array), 0.001);
        }
        
        @Test
        @DisplayName("Should calculate standard deviation")
        void testStd() {
            double[] array = {2, 4, 4, 4, 5, 5, 7, 9};
            double std = ArrayUtils.std(array);
            assertEquals(2.0, std, 0.01);
        }
        
        @Test
        @DisplayName("Should calculate variance")
        void testVariance() {
            double[] array = {2, 4, 4, 4, 5, 5, 7, 9};
            double variance = ArrayUtils.variance(array);
            assertEquals(4.0, variance, 0.01);
        }
        
        @Test
        @DisplayName("Should find minimum value")
        void testMin() {
            double[] array = {5, 2, 8, 1, 9};
            assertEquals(1.0, ArrayUtils.min(array), 0.001);
        }
        
        @Test
        @DisplayName("Should find maximum value")
        void testMax() {
            double[] array = {5, 2, 8, 1, 9};
            assertEquals(9.0, ArrayUtils.max(array), 0.001);
        }
        
        @Test
        @DisplayName("Should find argmax")
        void testArgmax() {
            double[] array = {5, 2, 8, 1, 9};
            assertEquals(4, ArrayUtils.argmax(array));
        }
        
        @Test
        @DisplayName("Should find argmin")
        void testArgmin() {
            double[] array = {5, 2, 8, 1, 9};
            assertEquals(3, ArrayUtils.argmin(array));
        }
    }
    
    @Nested
    @DisplayName("Normalization Tests")
    class NormalizationTests {
        
        @Test
        @DisplayName("Should normalize to zero mean and unit variance")
        void testNormalize() {
            double[] array = {2, 4, 4, 4, 5, 5, 7, 9};
            double[] normalized = ArrayUtils.normalize(array);
            
            // Mean should be close to 0
            assertEquals(0, ArrayUtils.mean(normalized), 0.001);
            // Std should be close to 1
            assertEquals(1, ArrayUtils.std(normalized), 0.01);
        }
        
        @Test
        @DisplayName("Should min-max normalize to 0-1 range")
        void testMinMaxNormalize() {
            double[] array = {0, 50, 100};
            double[] normalized = ArrayUtils.minMaxNormalize(array);
            
            assertEquals(0.0, ArrayUtils.min(normalized), 0.001);
            assertEquals(1.0, ArrayUtils.max(normalized), 0.001);
        }
    }
    
    @Nested
    @DisplayName("Array Manipulation Tests")
    class ManipulationTests {
        
        @Test
        @DisplayName("Should shuffle array with Random")
        void testShuffle() {
            double[] array1 = {1, 2, 3, 4, 5};
            double[] array2 = {1, 2, 3, 4, 5};
            
            ArrayUtils.shuffle(array1, new Random(42));
            ArrayUtils.shuffle(array2, new Random(42));
            
            assertArrayEquals(array1, array2, 0.001, "Same seed should produce same shuffle");
        }
        
        @Test
        @DisplayName("Should shuffle int array with Random")
        void testShuffleIntArray() {
            int[] array1 = {1, 2, 3, 4, 5};
            int[] array2 = {1, 2, 3, 4, 5};
            
            ArrayUtils.shuffle(array1, new Random(42));
            ArrayUtils.shuffle(array2, new Random(42));
            
            assertArrayEquals(array1, array2, "Same seed should produce same shuffle");
        }
        
        @Test
        @DisplayName("Should create array of zeros")
        void testZeros() {
            double[] zeros = ArrayUtils.zeros(5);
            
            assertEquals(5, zeros.length);
            for (double val : zeros) {
                assertEquals(0.0, val, 0.001);
            }
        }
        
        @Test
        @DisplayName("Should create array of ones")
        void testOnes() {
            double[] ones = ArrayUtils.ones(5);
            
            assertEquals(5, ones.length);
            for (double val : ones) {
                assertEquals(1.0, val, 0.001);
            }
        }
        
        @Test
        @DisplayName("Should create 2D zeros array")
        void testZeros2D() {
            double[][] zeros = ArrayUtils.zeros2D(3, 4);
            
            assertEquals(3, zeros.length);
            assertEquals(4, zeros[0].length);
            assertEquals(0.0, zeros[0][0], 0.001);
            assertEquals(0.0, zeros[2][3], 0.001);
        }
        
        @Test
        @DisplayName("Should create linspace array")
        void testLinspace() {
            double[] linspace = ArrayUtils.linspace(0, 10, 5);
            
            assertEquals(5, linspace.length);
            assertEquals(0, linspace[0], 0.001);
            assertEquals(10, linspace[4], 0.001);
            assertEquals(5, linspace[2], 0.001);
        }
        
        @Test
        @DisplayName("Should concatenate arrays")
        void testConcatenate() {
            double[] a = {1, 2};
            double[] b = {3, 4, 5};
            
            double[] result = ArrayUtils.concatenate(a, b);
            
            assertEquals(5, result.length);
            assertArrayEquals(new double[]{1, 2, 3, 4, 5}, result, 0.001);
        }
        
        @Test
        @DisplayName("Should slice array")
        void testSlice() {
            double[] array = {1, 2, 3, 4, 5};
            
            double[] sliced = ArrayUtils.slice(array, 1, 4);
            
            assertArrayEquals(new double[]{2, 3, 4}, sliced, 0.001);
        }
    }
    
    @Nested
    @DisplayName("Element-wise Operations Tests")
    class ElementWiseTests {
        
        @Test
        @DisplayName("Should add arrays element-wise")
        void testAdd() {
            double[] a = {1, 2, 3};
            double[] b = {4, 5, 6};
            double[] result = ArrayUtils.add(a, b);
            
            assertArrayEquals(new double[]{5, 7, 9}, result, 0.001);
        }
        
        @Test
        @DisplayName("Should subtract arrays element-wise")
        void testSubtract() {
            double[] a = {5, 7, 9};
            double[] b = {1, 2, 3};
            double[] result = ArrayUtils.subtract(a, b);
            
            assertArrayEquals(new double[]{4, 5, 6}, result, 0.001);
        }
        
        @Test
        @DisplayName("Should multiply arrays element-wise")
        void testMultiply() {
            double[] a = {1, 2, 3};
            double[] b = {4, 5, 6};
            double[] result = ArrayUtils.multiply(a, b);
            
            assertArrayEquals(new double[]{4, 10, 18}, result, 0.001);
        }
        
        @Test
        @DisplayName("Should scale array by scalar")
        void testScale() {
            double[] a = {1, 2, 3};
            double[] result = ArrayUtils.scale(a, 2);
            
            assertArrayEquals(new double[]{2, 4, 6}, result, 0.001);
        }
    }
    
    @Nested
    @DisplayName("Utility Tests")
    class UtilityTests {
        
        @Test
        @DisplayName("Should check if arrays are close")
        void testAllClose() {
            double[] a = {1.0, 2.0, 3.0};
            double[] b = {1.0001, 2.0001, 3.0001};
            
            assertTrue(ArrayUtils.allClose(a, b, 0.001));
            assertFalse(ArrayUtils.allClose(a, new double[]{1.1, 2.0, 3.0}, 0.001));
        }
        
        @Test
        @DisplayName("Should count unique values")
        void testCountUnique() {
            int[] array = {1, 2, 2, 3, 3, 3, 4};
            assertEquals(4, ArrayUtils.countUnique(array));
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle empty array for sum")
        void testEmptyArraySum() {
            double[] empty = {};
            assertEquals(0.0, ArrayUtils.sum(empty), 0.001);
        }
        
        @Test
        @DisplayName("Should handle single element array")
        void testSingleElementMean() {
            double[] single = {5.0};
            assertEquals(5.0, ArrayUtils.mean(single), 0.001);
        }
        
        @Test
        @DisplayName("Should handle square matrix transpose")
        void testSquareMatrixTranspose() {
            double[][] square = {
                {1, 2},
                {3, 4}
            };
            double[][] transposed = ArrayUtils.transpose(square);
            
            assertEquals(1, transposed[0][0], 0.001);
            assertEquals(3, transposed[0][1], 0.001);
            assertEquals(2, transposed[1][0], 0.001);
            assertEquals(4, transposed[1][1], 0.001);
        }
        
        @Test
        @DisplayName("Should throw exception for min of empty array")
        void testMinEmptyArray() {
            double[] empty = {};
            assertThrows(IllegalArgumentException.class, () -> ArrayUtils.min(empty));
        }
        
        @Test
        @DisplayName("Should throw exception for mismatched array lengths in dot")
        void testDotMismatchedArrays() {
            double[] a = {1, 2, 3};
            double[] b = {1, 2};
            assertThrows(IllegalArgumentException.class, () -> ArrayUtils.dot(a, b));
        }
    }
}
