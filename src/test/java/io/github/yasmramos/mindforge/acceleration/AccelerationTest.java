package io.github.yasmramos.mindforge.acceleration;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;

/**
 * Comprehensive test suite for the acceleration package.
 * Tests all components: AccelerationConfig, MemoryPool, VectorOps,
 * ParallelMatrix, AcceleratedConvolution, and AcceleratedLSTM.
 * 
 * @author MindForge Team
 * @version 1.2.0
 */
@DisplayName("Acceleration Package Tests")
public class AccelerationTest {
    
    private static final double EPSILON = 1e-9;
    private static final double LOOSE_EPSILON = 1e-6;
    
    @BeforeEach
    void setUp() {
        AccelerationConfig.getInstance().reset();
        MemoryPool.getInstance().reset();
    }
    
    // ==================== AccelerationConfig Tests ====================
    
    @Nested
    @DisplayName("AccelerationConfig Tests")
    class AccelerationConfigTests {
        
        @Test
        @DisplayName("Singleton instance works correctly")
        void testSingleton() {
            AccelerationConfig config1 = AccelerationConfig.getInstance();
            AccelerationConfig config2 = AccelerationConfig.getInstance();
            assertSame(config1, config2);
        }
        
        @Test
        @DisplayName("Thread count configuration")
        void testThreadConfiguration() {
            AccelerationConfig config = AccelerationConfig.getInstance();
            
            config.setNumThreads(4);
            assertEquals(4, config.getNumThreads());
            
            config.setNumThreads(1);
            assertEquals(1, config.getNumThreads());
        }
        
        @Test
        @DisplayName("Invalid thread count throws exception")
        void testInvalidThreadCount() {
            AccelerationConfig config = AccelerationConfig.getInstance();
            assertThrows(IllegalArgumentException.class, () -> config.setNumThreads(0));
            assertThrows(IllegalArgumentException.class, () -> config.setNumThreads(-1));
        }
        
        @Test
        @DisplayName("Block size configuration")
        void testBlockSizeConfiguration() {
            AccelerationConfig config = AccelerationConfig.getInstance();
            
            config.setBlockSize(128);
            assertEquals(128, config.getBlockSize());
            
            config.setBlockSize(32);
            assertEquals(32, config.getBlockSize());
        }
        
        @Test
        @DisplayName("Parallelization threshold")
        void testParallelizationThreshold() {
            AccelerationConfig config = AccelerationConfig.getInstance();
            
            config.setParallelThreshold(5000);
            assertTrue(config.shouldParallelize(10000));
            assertFalse(config.shouldParallelize(1000));
        }
        
        @Test
        @DisplayName("Memory pooling configuration")
        void testMemoryPoolingConfig() {
            AccelerationConfig config = AccelerationConfig.getInstance();
            
            config.enableMemoryPooling(true);
            assertTrue(config.isMemoryPoolingEnabled());
            
            config.enableMemoryPooling(false);
            assertFalse(config.isMemoryPoolingEnabled());
        }
        
        @Test
        @DisplayName("Hardware detection")
        void testHardwareDetection() {
            AccelerationConfig config = AccelerationConfig.getInstance();
            
            assertTrue(config.getAvailableProcessors() >= 1);
            assertTrue(config.getMaxMemory() > 0);
        }
        
        @Test
        @DisplayName("toString provides useful information")
        void testToString() {
            AccelerationConfig config = AccelerationConfig.getInstance();
            String str = config.toString();
            
            assertTrue(str.contains("threads"));
            assertTrue(str.contains("parallelization"));
            assertTrue(str.contains("blockSize"));
        }
    }
    
    // ==================== MemoryPool Tests ====================
    
    @Nested
    @DisplayName("MemoryPool Tests")
    class MemoryPoolTests {
        
        @Test
        @DisplayName("Singleton instance works correctly")
        void testSingleton() {
            MemoryPool pool1 = MemoryPool.getInstance();
            MemoryPool pool2 = MemoryPool.getInstance();
            assertSame(pool1, pool2);
        }
        
        @Test
        @DisplayName("Acquire and release double arrays")
        void testAcquireReleaseDouble() {
            MemoryPool pool = MemoryPool.getInstance();
            
            double[] arr1 = pool.acquire(100);
            assertNotNull(arr1);
            assertTrue(arr1.length >= 100);
            
            pool.release(arr1);
            
            double[] arr2 = pool.acquire(100);
            // May or may not be same array depending on pool state
            assertNotNull(arr2);
            assertTrue(arr2.length >= 100);
        }
        
        @Test
        @DisplayName("Acquire exact size array")
        void testAcquireExact() {
            MemoryPool pool = MemoryPool.getInstance();
            
            double[] arr = pool.acquireExact(100);
            assertEquals(100, arr.length);
        }
        
        @Test
        @DisplayName("Acquire 2D arrays")
        void testAcquire2D() {
            MemoryPool pool = MemoryPool.getInstance();
            
            double[][] arr = pool.acquire2D(10, 20);
            assertNotNull(arr);
            assertTrue(arr.length >= 10);
            assertTrue(arr[0].length >= 20);
            
            pool.release2D(arr);
        }
        
        @Test
        @DisplayName("Acquire int arrays")
        void testAcquireInt() {
            MemoryPool pool = MemoryPool.getInstance();
            
            int[] arr = pool.acquireInt(100);
            assertNotNull(arr);
            assertTrue(arr.length >= 100);
            
            pool.releaseInt(arr);
        }
        
        @Test
        @DisplayName("Pool statistics tracking")
        void testStatistics() {
            MemoryPool pool = MemoryPool.getInstance();
            pool.reset();
            
            // Make some allocations
            for (int i = 0; i < 10; i++) {
                double[] arr = pool.acquire(100);
                pool.release(arr);
            }
            
            assertTrue(pool.getTotalAllocations() >= 10);
            assertTrue(pool.getPoolMisses() >= 1);
        }
        
        @Test
        @DisplayName("Enable/disable pooling")
        void testEnableDisable() {
            MemoryPool pool = MemoryPool.getInstance();
            
            pool.setEnabled(false);
            assertFalse(pool.isEnabled());
            
            pool.setEnabled(true);
            assertTrue(pool.isEnabled());
        }
        
        @Test
        @DisplayName("Statistics string format")
        void testStatisticsString() {
            MemoryPool pool = MemoryPool.getInstance();
            String stats = pool.getStatistics();
            
            assertTrue(stats.contains("allocations"));
            assertTrue(stats.contains("hits"));
            assertTrue(stats.contains("misses"));
        }
    }
    
    // ==================== VectorOps Tests ====================
    
    @Nested
    @DisplayName("VectorOps Tests")
    class VectorOpsTests {
        
        private double[] a;
        private double[] b;
        
        @BeforeEach
        void initArrays() {
            a = new double[] {1.0, 2.0, 3.0, 4.0, 5.0};
            b = new double[] {5.0, 4.0, 3.0, 2.0, 1.0};
        }
        
        @Test
        @DisplayName("Vector addition")
        void testAdd() {
            double[] result = VectorOps.add(a, b);
            
            assertEquals(6.0, result[0], EPSILON);
            assertEquals(6.0, result[1], EPSILON);
            assertEquals(6.0, result[2], EPSILON);
            assertEquals(6.0, result[3], EPSILON);
            assertEquals(6.0, result[4], EPSILON);
        }
        
        @Test
        @DisplayName("Vector subtraction")
        void testSubtract() {
            double[] result = VectorOps.subtract(a, b);
            
            assertEquals(-4.0, result[0], EPSILON);
            assertEquals(-2.0, result[1], EPSILON);
            assertEquals(0.0, result[2], EPSILON);
            assertEquals(2.0, result[3], EPSILON);
            assertEquals(4.0, result[4], EPSILON);
        }
        
        @Test
        @DisplayName("Element-wise multiplication")
        void testMultiply() {
            double[] result = VectorOps.multiply(a, b);
            
            assertEquals(5.0, result[0], EPSILON);
            assertEquals(8.0, result[1], EPSILON);
            assertEquals(9.0, result[2], EPSILON);
            assertEquals(8.0, result[3], EPSILON);
            assertEquals(5.0, result[4], EPSILON);
        }
        
        @Test
        @DisplayName("Scalar multiplication")
        void testScale() {
            double[] result = VectorOps.scale(a, 2.0);
            
            assertEquals(2.0, result[0], EPSILON);
            assertEquals(4.0, result[1], EPSILON);
            assertEquals(6.0, result[2], EPSILON);
            assertEquals(8.0, result[3], EPSILON);
            assertEquals(10.0, result[4], EPSILON);
        }
        
        @Test
        @DisplayName("Dot product")
        void testDot() {
            double result = VectorOps.dot(a, b);
            // 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5 + 8 + 9 + 8 + 5 = 35
            assertEquals(35.0, result, EPSILON);
        }
        
        @Test
        @DisplayName("Sum of elements")
        void testSum() {
            double result = VectorOps.sum(a);
            assertEquals(15.0, result, EPSILON);
        }
        
        @Test
        @DisplayName("Max value")
        void testMax() {
            assertEquals(5.0, VectorOps.max(a), EPSILON);
        }
        
        @Test
        @DisplayName("Min value")
        void testMin() {
            assertEquals(1.0, VectorOps.min(a), EPSILON);
        }
        
        @Test
        @DisplayName("L2 norm")
        void testNorm2() {
            double result = VectorOps.norm2(a);
            // sqrt(1 + 4 + 9 + 16 + 25) = sqrt(55)
            assertEquals(Math.sqrt(55.0), result, EPSILON);
        }
        
        @Test
        @DisplayName("L1 norm")
        void testNorm1() {
            double[] arr = new double[] {-1.0, 2.0, -3.0};
            assertEquals(6.0, VectorOps.norm1(arr), EPSILON);
        }
        
        @Test
        @DisplayName("Sigmoid activation")
        void testSigmoid() {
            double[] input = new double[] {0.0, 1.0, -1.0};
            double[] result = VectorOps.sigmoid(input);
            
            assertEquals(0.5, result[0], EPSILON);
            assertEquals(1.0 / (1.0 + Math.exp(-1)), result[1], EPSILON);
            assertEquals(1.0 / (1.0 + Math.exp(1)), result[2], EPSILON);
        }
        
        @Test
        @DisplayName("Tanh activation")
        void testTanh() {
            double[] input = new double[] {0.0, 1.0, -1.0};
            double[] result = VectorOps.tanh(input);
            
            assertEquals(0.0, result[0], EPSILON);
            assertEquals(Math.tanh(1.0), result[1], EPSILON);
            assertEquals(Math.tanh(-1.0), result[2], EPSILON);
        }
        
        @Test
        @DisplayName("ReLU activation")
        void testRelu() {
            double[] input = new double[] {-2.0, 0.0, 3.0, -1.0, 5.0};
            double[] result = VectorOps.relu(input);
            
            assertEquals(0.0, result[0], EPSILON);
            assertEquals(0.0, result[1], EPSILON);
            assertEquals(3.0, result[2], EPSILON);
            assertEquals(0.0, result[3], EPSILON);
            assertEquals(5.0, result[4], EPSILON);
        }
        
        @Test
        @DisplayName("Softmax normalization")
        void testSoftmax() {
            double[] input = new double[] {1.0, 2.0, 3.0};
            double[] result = VectorOps.softmax(input);
            
            // Softmax should sum to 1
            assertEquals(1.0, VectorOps.sum(result), EPSILON);
            
            // Values should be in ascending order
            assertTrue(result[0] < result[1]);
            assertTrue(result[1] < result[2]);
        }
        
        @Test
        @DisplayName("Fused multiply-add")
        void testFma() {
            double[] c = new double[] {10.0, 20.0, 30.0, 40.0, 50.0};
            double[] result = VectorOps.fma(a, b, c);
            
            // a[i] * b[i] + c[i]
            assertEquals(15.0, result[0], EPSILON); // 1*5 + 10
            assertEquals(28.0, result[1], EPSILON); // 2*4 + 20
            assertEquals(39.0, result[2], EPSILON); // 3*3 + 30
            assertEquals(48.0, result[3], EPSILON); // 4*2 + 40
            assertEquals(55.0, result[4], EPSILON); // 5*1 + 50
        }
        
        @Test
        @DisplayName("Clip values")
        void testClip() {
            double[] input = new double[] {-5.0, 0.0, 5.0, 10.0, 15.0};
            double[] result = VectorOps.clip(input, 0.0, 10.0);
            
            assertEquals(0.0, result[0], EPSILON);
            assertEquals(0.0, result[1], EPSILON);
            assertEquals(5.0, result[2], EPSILON);
            assertEquals(10.0, result[3], EPSILON);
            assertEquals(10.0, result[4], EPSILON);
        }
        
        @Test
        @DisplayName("Large array operations")
        void testLargeArrays() {
            int size = 100000;
            double[] largeA = new double[size];
            double[] largeB = new double[size];
            
            Random rand = new Random(42);
            for (int i = 0; i < size; i++) {
                largeA[i] = rand.nextDouble();
                largeB[i] = rand.nextDouble();
            }
            
            // Should complete without error and produce valid results
            double[] sum = VectorOps.add(largeA, largeB);
            // Pool may return larger arrays (rounded to power of 2)
            assertTrue(sum.length >= size);
            
            double dot = VectorOps.dot(largeA, largeB);
            assertTrue(dot > 0);
        }
        
        @Test
        @DisplayName("Array length mismatch throws exception")
        void testLengthMismatch() {
            double[] short_arr = new double[] {1.0, 2.0};
            assertThrows(IllegalArgumentException.class, () -> VectorOps.add(a, short_arr));
        }
    }
    
    // ==================== ParallelMatrix Tests ====================
    
    @Nested
    @DisplayName("ParallelMatrix Tests")
    class ParallelMatrixTests {
        
        @Test
        @DisplayName("Matrix multiplication")
        void testMatrixMultiply() {
            double[][] a = {{1, 2}, {3, 4}};
            double[][] b = {{5, 6}, {7, 8}};
            
            double[][] c = ParallelMatrix.multiply(a, b);
            
            // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
            // [[19, 22], [43, 50]]
            assertEquals(19.0, c[0][0], EPSILON);
            assertEquals(22.0, c[0][1], EPSILON);
            assertEquals(43.0, c[1][0], EPSILON);
            assertEquals(50.0, c[1][1], EPSILON);
        }
        
        @Test
        @DisplayName("Matrix-vector multiplication")
        void testMatrixVectorMultiply() {
            double[][] a = {{1, 2, 3}, {4, 5, 6}};
            double[] x = {1, 2, 3};
            
            double[] y = ParallelMatrix.multiplyVector(a, x);
            
            // [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
            assertEquals(14.0, y[0], EPSILON);
            assertEquals(32.0, y[1], EPSILON);
        }
        
        @Test
        @DisplayName("Matrix transpose")
        void testTranspose() {
            double[][] a = {{1, 2, 3}, {4, 5, 6}};
            double[][] t = ParallelMatrix.transpose(a);
            
            assertEquals(3, t.length);
            assertEquals(2, t[0].length);
            
            assertEquals(1.0, t[0][0], EPSILON);
            assertEquals(4.0, t[0][1], EPSILON);
            assertEquals(2.0, t[1][0], EPSILON);
            assertEquals(5.0, t[1][1], EPSILON);
            assertEquals(3.0, t[2][0], EPSILON);
            assertEquals(6.0, t[2][1], EPSILON);
        }
        
        @Test
        @DisplayName("Outer product")
        void testOuterProduct() {
            double[] a = {1, 2, 3};
            double[] b = {4, 5};
            
            double[][] c = ParallelMatrix.outerProduct(a, b);
            
            assertEquals(3, c.length);
            assertEquals(2, c[0].length);
            
            assertEquals(4.0, c[0][0], EPSILON);
            assertEquals(5.0, c[0][1], EPSILON);
            assertEquals(8.0, c[1][0], EPSILON);
            assertEquals(10.0, c[1][1], EPSILON);
            assertEquals(12.0, c[2][0], EPSILON);
            assertEquals(15.0, c[2][1], EPSILON);
        }
        
        @Test
        @DisplayName("Matrix addition")
        void testMatrixAdd() {
            double[][] a = {{1, 2}, {3, 4}};
            double[][] b = {{5, 6}, {7, 8}};
            
            double[][] c = ParallelMatrix.add(a, b);
            
            assertEquals(6.0, c[0][0], EPSILON);
            assertEquals(8.0, c[0][1], EPSILON);
            assertEquals(10.0, c[1][0], EPSILON);
            assertEquals(12.0, c[1][1], EPSILON);
        }
        
        @Test
        @DisplayName("Matrix subtraction")
        void testMatrixSubtract() {
            double[][] a = {{5, 6}, {7, 8}};
            double[][] b = {{1, 2}, {3, 4}};
            
            double[][] c = ParallelMatrix.subtract(a, b);
            
            assertEquals(4.0, c[0][0], EPSILON);
            assertEquals(4.0, c[0][1], EPSILON);
            assertEquals(4.0, c[1][0], EPSILON);
            assertEquals(4.0, c[1][1], EPSILON);
        }
        
        @Test
        @DisplayName("Matrix scaling")
        void testMatrixScale() {
            double[][] a = {{1, 2}, {3, 4}};
            double[][] b = ParallelMatrix.scale(a, 2.0);
            
            assertEquals(2.0, b[0][0], EPSILON);
            assertEquals(4.0, b[0][1], EPSILON);
            assertEquals(6.0, b[1][0], EPSILON);
            assertEquals(8.0, b[1][1], EPSILON);
        }
        
        @Test
        @DisplayName("Hadamard product")
        void testHadamard() {
            double[][] a = {{1, 2}, {3, 4}};
            double[][] b = {{5, 6}, {7, 8}};
            
            double[][] c = ParallelMatrix.hadamard(a, b);
            
            assertEquals(5.0, c[0][0], EPSILON);
            assertEquals(12.0, c[0][1], EPSILON);
            assertEquals(21.0, c[1][0], EPSILON);
            assertEquals(32.0, c[1][1], EPSILON);
        }
        
        @Test
        @DisplayName("Matrix sum")
        void testMatrixSum() {
            double[][] a = {{1, 2}, {3, 4}};
            assertEquals(10.0, ParallelMatrix.sum(a), EPSILON);
        }
        
        @Test
        @DisplayName("Frobenius norm")
        void testFrobeniusNorm() {
            double[][] a = {{1, 2}, {3, 4}};
            // sqrt(1 + 4 + 9 + 16) = sqrt(30)
            assertEquals(Math.sqrt(30.0), ParallelMatrix.frobeniusNorm(a), EPSILON);
        }
        
        @Test
        @DisplayName("Identity matrix")
        void testIdentity() {
            double[][] eye = ParallelMatrix.identity(3);
            
            assertEquals(1.0, eye[0][0], EPSILON);
            assertEquals(0.0, eye[0][1], EPSILON);
            assertEquals(0.0, eye[0][2], EPSILON);
            assertEquals(0.0, eye[1][0], EPSILON);
            assertEquals(1.0, eye[1][1], EPSILON);
            assertEquals(0.0, eye[1][2], EPSILON);
            assertEquals(0.0, eye[2][0], EPSILON);
            assertEquals(0.0, eye[2][1], EPSILON);
            assertEquals(1.0, eye[2][2], EPSILON);
        }
        
        @Test
        @DisplayName("Large matrix multiplication")
        void testLargeMatrixMultiply() {
            int size = 200;
            double[][] a = new double[size][size];
            double[][] b = new double[size][size];
            
            Random rand = new Random(42);
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    a[i][j] = rand.nextDouble();
                    b[i][j] = rand.nextDouble();
                }
            }
            
            double[][] c = ParallelMatrix.multiply(a, b);
            
            assertEquals(size, c.length);
            assertEquals(size, c[0].length);
        }
        
        @Test
        @DisplayName("Dimension mismatch throws exception")
        void testDimensionMismatch() {
            double[][] a = {{1, 2, 3}, {4, 5, 6}};
            double[][] b = {{1, 2}, {3, 4}};
            
            assertThrows(IllegalArgumentException.class, () -> ParallelMatrix.multiply(a, b));
        }
    }
    
    // ==================== AcceleratedConvolution Tests ====================
    
    @Nested
    @DisplayName("AcceleratedConvolution Tests")
    class AcceleratedConvolutionTests {
        
        @Test
        @DisplayName("Basic 2D convolution")
        void testConv2d() {
            // Simple 1-channel, 4x4 input
            double[][][] input = {
                {
                    {1, 2, 3, 4},
                    {5, 6, 7, 8},
                    {9, 10, 11, 12},
                    {13, 14, 15, 16}
                }
            };
            
            // 1 filter, 1 channel, 2x2 kernel (all ones)
            double[][][][] filters = {
                {
                    {
                        {1, 1},
                        {1, 1}
                    }
                }
            };
            
            double[] biases = {0};
            
            double[][][] output = AcceleratedConvolution.conv2d(input, filters, biases, 1, 0);
            
            assertEquals(1, output.length); // 1 filter
            assertEquals(3, output[0].length); // 3x3 output
            assertEquals(3, output[0][0].length);
            
            // Top-left: 1+2+5+6 = 14
            assertEquals(14.0, output[0][0][0], EPSILON);
        }
        
        @Test
        @DisplayName("Convolution with padding")
        void testConv2dWithPadding() {
            double[][][] input = {
                {
                    {1, 2},
                    {3, 4}
                }
            };
            
            double[][][][] filters = {
                {
                    {
                        {1, 0},
                        {0, 1}
                    }
                }
            };
            
            double[] biases = {0};
            
            double[][][] output = AcceleratedConvolution.conv2d(input, filters, biases, 1, 1);
            
            // With padding 1, output should be 3x3
            assertEquals(3, output[0].length);
            assertEquals(3, output[0][0].length);
        }
        
        @Test
        @DisplayName("Convolution with stride")
        void testConv2dWithStride() {
            double[][][] input = {
                {
                    {1, 2, 3, 4},
                    {5, 6, 7, 8},
                    {9, 10, 11, 12},
                    {13, 14, 15, 16}
                }
            };
            
            double[][][][] filters = {
                {
                    {
                        {1, 1},
                        {1, 1}
                    }
                }
            };
            
            double[] biases = {0};
            
            double[][][] output = AcceleratedConvolution.conv2d(input, filters, biases, 2, 0);
            
            // With stride 2, output should be 2x2
            assertEquals(2, output[0].length);
            assertEquals(2, output[0][0].length);
        }
        
        @Test
        @DisplayName("Multiple filters")
        void testMultipleFilters() {
            double[][][] input = {
                {
                    {1, 2},
                    {3, 4}
                }
            };
            
            double[][][][] filters = {
                { { {1, 0}, {0, 0} } }, // Filter 1
                { { {0, 1}, {0, 0} } }, // Filter 2
                { { {0, 0}, {1, 0} } }  // Filter 3
            };
            
            double[] biases = {0, 0, 0};
            
            double[][][] output = AcceleratedConvolution.conv2d(input, filters, biases, 1, 0);
            
            assertEquals(3, output.length); // 3 filters
        }
        
        @Test
        @DisplayName("Im2col transformation")
        void testIm2col() {
            double[][][] input = {
                {
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9}
                }
            };
            
            double[][] cols = AcceleratedConvolution.im2col(input, 2, 2, 1, 0);
            
            // 1 channel * 2 * 2 = 4 rows
            // 2 * 2 = 4 output positions
            assertEquals(4, cols.length);
            assertEquals(4, cols[0].length);
        }
        
        @Test
        @DisplayName("Max pooling")
        void testMaxPool2d() {
            double[][][] input = {
                {
                    {1, 2, 3, 4},
                    {5, 6, 7, 8},
                    {9, 10, 11, 12},
                    {13, 14, 15, 16}
                }
            };
            
            double[][][] output = AcceleratedConvolution.maxPool2d(input, 2, 2);
            
            assertEquals(1, output.length);
            assertEquals(2, output[0].length);
            assertEquals(2, output[0][0].length);
            
            // Max of [1,2,5,6] = 6
            assertEquals(6.0, output[0][0][0], EPSILON);
            // Max of [3,4,7,8] = 8
            assertEquals(8.0, output[0][0][1], EPSILON);
            // Max of [9,10,13,14] = 14
            assertEquals(14.0, output[0][1][0], EPSILON);
            // Max of [11,12,15,16] = 16
            assertEquals(16.0, output[0][1][1], EPSILON);
        }
        
        @Test
        @DisplayName("Average pooling")
        void testAvgPool2d() {
            double[][][] input = {
                {
                    {1, 2, 3, 4},
                    {5, 6, 7, 8},
                    {9, 10, 11, 12},
                    {13, 14, 15, 16}
                }
            };
            
            double[][][] output = AcceleratedConvolution.avgPool2d(input, 2, 2);
            
            // Avg of [1,2,5,6] = 3.5
            assertEquals(3.5, output[0][0][0], EPSILON);
        }
        
        @Test
        @DisplayName("Padding and unpadding")
        void testPadding() {
            double[][][] input = {
                {
                    {1, 2},
                    {3, 4}
                }
            };
            
            double[][][] padded = AcceleratedConvolution.pad(input, 1);
            
            assertEquals(4, padded[0].length);
            assertEquals(4, padded[0][0].length);
            
            // Check corners are zero
            assertEquals(0.0, padded[0][0][0], EPSILON);
            assertEquals(0.0, padded[0][3][3], EPSILON);
            
            // Check center is preserved
            assertEquals(1.0, padded[0][1][1], EPSILON);
            assertEquals(4.0, padded[0][2][2], EPSILON);
            
            double[][][] unpadded = AcceleratedConvolution.unpad(padded, 1);
            assertEquals(2, unpadded[0].length);
            assertEquals(2, unpadded[0][0].length);
            assertEquals(1.0, unpadded[0][0][0], EPSILON);
        }
        
        @Test
        @DisplayName("Flatten and reshape")
        void testFlattenReshape() {
            double[][][] tensor = {
                {
                    {1, 2, 3},
                    {4, 5, 6}
                },
                {
                    {7, 8, 9},
                    {10, 11, 12}
                }
            };
            
            double[] flat = AcceleratedConvolution.flatten(tensor);
            assertEquals(12, flat.length);
            assertEquals(1.0, flat[0], EPSILON);
            assertEquals(12.0, flat[11], EPSILON);
            
            double[][][] reshaped = AcceleratedConvolution.reshape(flat, 2, 2, 3);
            assertEquals(1.0, reshaped[0][0][0], EPSILON);
            assertEquals(12.0, reshaped[1][1][2], EPSILON);
        }
    }
    
    // ==================== AcceleratedLSTM Tests ====================
    
    @Nested
    @DisplayName("AcceleratedLSTM Tests")
    class AcceleratedLSTMTests {
        
        private int inputSize = 4;
        private int hiddenSize = 3;
        private double[][] weightsIH;
        private double[][] weightsHH;
        private double[] biases;
        
        @BeforeEach
        void initWeights() {
            Object[] weights = AcceleratedLSTM.initializeWeights(inputSize, hiddenSize, 42);
            weightsIH = (double[][]) weights[0];
            weightsHH = (double[][]) weights[1];
            biases = (double[]) weights[2];
        }
        
        @Test
        @DisplayName("Weight initialization")
        void testWeightInitialization() {
            assertEquals(4 * hiddenSize, weightsIH.length);
            assertEquals(inputSize, weightsIH[0].length);
            assertEquals(4 * hiddenSize, weightsHH.length);
            assertEquals(hiddenSize, weightsHH[0].length);
            assertEquals(4 * hiddenSize, biases.length);
            
            // Forget gate biases should be initialized to 1.0
            for (int i = 0; i < hiddenSize; i++) {
                assertEquals(1.0, biases[hiddenSize + i], EPSILON);
            }
        }
        
        @Test
        @DisplayName("LSTM cell forward pass")
        void testLstmCellForward() {
            double[] input = {0.5, -0.3, 0.2, 0.1};
            double[] prevHidden = {0.0, 0.0, 0.0};
            double[] prevCell = {0.0, 0.0, 0.0};
            
            double[][] result = AcceleratedLSTM.lstmCellForward(
                input, prevHidden, prevCell, weightsIH, weightsHH, biases);
            
            double[] newHidden = result[0];
            double[] newCell = result[1];
            double[] gates = result[2];
            
            assertEquals(hiddenSize, newHidden.length);
            assertEquals(hiddenSize, newCell.length);
            assertEquals(4 * hiddenSize, gates.length);
            
            // Check outputs are in valid ranges
            for (int i = 0; i < hiddenSize; i++) {
                assertTrue(newHidden[i] >= -1.0 && newHidden[i] <= 1.0);
            }
        }
        
        @Test
        @DisplayName("LSTM cell backward pass")
        void testLstmCellBackward() {
            double[] input = {0.5, -0.3, 0.2, 0.1};
            double[] prevHidden = {0.0, 0.0, 0.0};
            double[] prevCell = {0.0, 0.0, 0.0};
            
            // Forward pass
            double[][] forward = AcceleratedLSTM.lstmCellForward(
                input, prevHidden, prevCell, weightsIH, weightsHH, biases);
            
            double[] newHidden = forward[0];
            double[] newCell = forward[1];
            double[] gates = forward[2];
            
            // Backward pass
            double[] gradHidden = {0.1, 0.2, -0.1};
            double[] gradCell = {0.0, 0.0, 0.0};
            
            Object[] backward = AcceleratedLSTM.lstmCellBackward(
                gradHidden, gradCell, input, prevHidden, prevCell, newCell, gates,
                weightsIH, weightsHH);
            
            double[] gradInput = (double[]) backward[0];
            double[] gradPrevHidden = (double[]) backward[1];
            double[] gradPrevCell = (double[]) backward[2];
            double[][] gradWeightsIH = (double[][]) backward[3];
            double[][] gradWeightsHH = (double[][]) backward[4];
            double[] gradBiases = (double[]) backward[5];
            
            assertEquals(inputSize, gradInput.length);
            assertEquals(hiddenSize, gradPrevHidden.length);
            assertEquals(hiddenSize, gradPrevCell.length);
            assertEquals(4 * hiddenSize, gradWeightsIH.length);
            assertEquals(inputSize, gradWeightsIH[0].length);
        }
        
        @Test
        @DisplayName("Sequence processing")
        void testProcessSequence() {
            double[][] sequence = {
                {0.5, -0.3, 0.2, 0.1},
                {0.1, 0.2, -0.4, 0.3},
                {-0.2, 0.1, 0.5, -0.1}
            };
            
            double[] initialHidden = new double[hiddenSize];
            double[] initialCell = new double[hiddenSize];
            
            double[][] outputs = AcceleratedLSTM.processSequence(
                sequence, initialHidden, initialCell, weightsIH, weightsHH, biases, true);
            
            assertEquals(3, outputs.length); // 3 time steps
            assertEquals(hiddenSize, outputs[0].length);
        }
        
        @Test
        @DisplayName("Sequence processing - return last only")
        void testProcessSequenceLastOnly() {
            double[][] sequence = {
                {0.5, -0.3, 0.2, 0.1},
                {0.1, 0.2, -0.4, 0.3},
                {-0.2, 0.1, 0.5, -0.1}
            };
            
            double[] initialHidden = new double[hiddenSize];
            double[] initialCell = new double[hiddenSize];
            
            double[][] outputs = AcceleratedLSTM.processSequence(
                sequence, initialHidden, initialCell, weightsIH, weightsHH, biases, false);
            
            assertEquals(1, outputs.length); // Only last output
            assertEquals(hiddenSize, outputs[0].length);
        }
        
        @Test
        @DisplayName("Batch processing")
        void testProcessBatch() {
            int batchSize = 4;
            int seqLength = 3;
            
            double[][][] batch = new double[batchSize][seqLength][inputSize];
            double[][] initialHidden = new double[batchSize][hiddenSize];
            double[][] initialCell = new double[batchSize][hiddenSize];
            
            Random rand = new Random(42);
            for (int b = 0; b < batchSize; b++) {
                for (int t = 0; t < seqLength; t++) {
                    for (int i = 0; i < inputSize; i++) {
                        batch[b][t][i] = rand.nextGaussian() * 0.5;
                    }
                }
            }
            
            double[][][] outputs = AcceleratedLSTM.processBatch(
                batch, initialHidden, initialCell, weightsIH, weightsHH, biases, true);
            
            assertEquals(batchSize, outputs.length);
            assertEquals(seqLength, outputs[0].length);
            assertEquals(hiddenSize, outputs[0][0].length);
        }
        
        @Test
        @DisplayName("Bidirectional processing - concat mode")
        void testBidirectionalConcat() {
            double[][] sequence = {
                {0.5, -0.3, 0.2, 0.1},
                {0.1, 0.2, -0.4, 0.3},
                {-0.2, 0.1, 0.5, -0.1}
            };
            
            double[] initHiddenFwd = new double[hiddenSize];
            double[] initCellFwd = new double[hiddenSize];
            double[] initHiddenBwd = new double[hiddenSize];
            double[] initCellBwd = new double[hiddenSize];
            
            // Use same weights for simplicity
            double[][] outputs = AcceleratedLSTM.processBidirectional(
                sequence,
                initHiddenFwd, initCellFwd,
                initHiddenBwd, initCellBwd,
                weightsIH, weightsHH, biases,
                weightsIH, weightsHH, biases,
                "concat"
            );
            
            assertEquals(3, outputs.length);
            assertEquals(2 * hiddenSize, outputs[0].length); // Concatenated
        }
        
        @Test
        @DisplayName("Bidirectional processing - sum mode")
        void testBidirectionalSum() {
            double[][] sequence = {
                {0.5, -0.3, 0.2, 0.1},
                {0.1, 0.2, -0.4, 0.3}
            };
            
            double[] initHidden = new double[hiddenSize];
            double[] initCell = new double[hiddenSize];
            
            double[][] outputs = AcceleratedLSTM.processBidirectional(
                sequence,
                initHidden, initCell,
                initHidden, initCell,
                weightsIH, weightsHH, biases,
                weightsIH, weightsHH, biases,
                "sum"
            );
            
            assertEquals(2, outputs.length);
            assertEquals(hiddenSize, outputs[0].length); // Same size after sum
        }
        
        @Test
        @DisplayName("Gradient clipping - vector")
        void testGradientClippingVector() {
            double[] gradients = {3.0, 4.0}; // norm = 5.0
            
            AcceleratedLSTM.clipGradients(gradients, 2.5);
            
            double norm = Math.sqrt(gradients[0] * gradients[0] + gradients[1] * gradients[1]);
            assertEquals(2.5, norm, LOOSE_EPSILON);
        }
        
        @Test
        @DisplayName("Gradient clipping - matrix")
        void testGradientClippingMatrix() {
            double[][] gradients = {{3.0, 0.0}, {0.0, 4.0}}; // Frobenius norm = 5.0
            
            AcceleratedLSTM.clipGradients(gradients, 2.5);
            
            double norm = ParallelMatrix.frobeniusNorm(gradients);
            assertEquals(2.5, norm, LOOSE_EPSILON);
        }
    }
    
    // ==================== Integration Tests ====================
    
    @Nested
    @DisplayName("Integration Tests")
    class IntegrationTests {
        
        @Test
        @DisplayName("VectorOps uses memory pool")
        void testVectorOpsUsesPool() {
            MemoryPool pool = MemoryPool.getInstance();
            pool.reset();
            
            double[] a = new double[1000];
            double[] b = new double[1000];
            
            VectorOps.add(a, b);
            VectorOps.multiply(a, b);
            VectorOps.sigmoid(a);
            
            assertTrue(pool.getTotalAllocations() >= 3);
        }
        
        @Test
        @DisplayName("ParallelMatrix uses memory pool")
        void testParallelMatrixUsesPool() {
            MemoryPool pool = MemoryPool.getInstance();
            pool.reset();
            
            double[][] a = new double[50][50];
            double[][] b = new double[50][50];
            
            ParallelMatrix.multiply(a, b);
            ParallelMatrix.transpose(a);
            
            assertTrue(pool.getTotalAllocations() >= 2);
        }
        
        @Test
        @DisplayName("Config affects parallelization")
        void testConfigAffectsParallelization() {
            AccelerationConfig config = AccelerationConfig.getInstance();
            
            // Disable parallelization
            config.setUseParallelization(false);
            assertFalse(config.isParallelizationEnabled());
            
            // Operations should still work
            double[] a = new double[100000];
            double[] b = new double[100000];
            double[] result = VectorOps.add(a, b);
            // Pool may return larger arrays (rounded to power of 2)
            assertTrue(result.length >= 100000);
            
            // Re-enable
            config.setUseParallelization(true);
        }
        
        @Test
        @DisplayName("End-to-end CNN acceleration")
        void testEndToEndCNN() {
            // Create a simple input
            double[][][] input = new double[3][28][28];
            Random rand = new Random(42);
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < 28; h++) {
                    for (int w = 0; w < 28; w++) {
                        input[c][h][w] = rand.nextDouble();
                    }
                }
            }
            
            // Create filters
            double[][][][] filters = new double[8][3][3][3];
            for (int f = 0; f < 8; f++) {
                for (int c = 0; c < 3; c++) {
                    for (int h = 0; h < 3; h++) {
                        for (int w = 0; w < 3; w++) {
                            filters[f][c][h][w] = rand.nextGaussian() * 0.1;
                        }
                    }
                }
            }
            
            double[] biases = new double[8];
            
            // Forward pass
            double[][][] conv = AcceleratedConvolution.conv2d(input, filters, biases, 1, 1);
            assertEquals(8, conv.length);
            assertEquals(28, conv[0].length);
            assertEquals(28, conv[0][0].length);
            
            // Max pooling
            double[][][] pooled = AcceleratedConvolution.maxPool2d(conv, 2, 2);
            assertEquals(8, pooled.length);
            assertEquals(14, pooled[0].length);
            assertEquals(14, pooled[0][0].length);
            
            // Flatten
            double[] flat = AcceleratedConvolution.flatten(pooled);
            assertEquals(8 * 14 * 14, flat.length);
        }
        
        @Test
        @DisplayName("End-to-end LSTM acceleration")
        void testEndToEndLSTM() {
            int inputSize = 10;
            int hiddenSize = 20;
            int seqLength = 15;
            int batchSize = 4;
            
            // Initialize weights
            Object[] weights = AcceleratedLSTM.initializeWeights(inputSize, hiddenSize, 42);
            double[][] weightsIH = (double[][]) weights[0];
            double[][] weightsHH = (double[][]) weights[1];
            double[] biases = (double[]) weights[2];
            
            // Create batch of sequences
            double[][][] batch = new double[batchSize][seqLength][inputSize];
            double[][] initialHidden = new double[batchSize][hiddenSize];
            double[][] initialCell = new double[batchSize][hiddenSize];
            
            Random rand = new Random(42);
            for (int b = 0; b < batchSize; b++) {
                for (int t = 0; t < seqLength; t++) {
                    for (int i = 0; i < inputSize; i++) {
                        batch[b][t][i] = rand.nextGaussian() * 0.5;
                    }
                }
            }
            
            // Process batch
            double[][][] outputs = AcceleratedLSTM.processBatch(
                batch, initialHidden, initialCell, weightsIH, weightsHH, biases, true);
            
            assertEquals(batchSize, outputs.length);
            assertEquals(seqLength, outputs[0].length);
            assertEquals(hiddenSize, outputs[0][0].length);
            
            // Check outputs are valid
            for (int b = 0; b < batchSize; b++) {
                for (int t = 0; t < seqLength; t++) {
                    for (int h = 0; h < hiddenSize; h++) {
                        assertTrue(Double.isFinite(outputs[b][t][h]));
                    }
                }
            }
        }
    }
}
