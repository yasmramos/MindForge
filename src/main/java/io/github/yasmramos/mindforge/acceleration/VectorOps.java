package io.github.yasmramos.mindforge.acceleration;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.stream.IntStream;

/**
 * High-performance vectorized operations for numerical computing.
 * 
 * Provides optimized implementations of common vector and array operations
 * using loop unrolling, cache-friendly access patterns, and parallel execution.
 * 
 * <p>Operations automatically select between sequential and parallel execution
 * based on array size and configuration thresholds.</p>
 * 
 * @author MindForge Team
 * @version 1.2.0
 */
public class VectorOps {
    
    private static final AccelerationConfig config = AccelerationConfig.getInstance();
    private static final MemoryPool pool = MemoryPool.getInstance();
    
    // Loop unrolling factor
    private static final int UNROLL_FACTOR = 8;
    
    // Private constructor - utility class
    private VectorOps() {}
    
    // ==================== Element-wise Operations ====================
    
    /**
     * Element-wise addition: result[i] = a[i] + b[i]
     * 
     * @param a first array
     * @param b second array
     * @return result array
     */
    public static double[] add(double[] a, double[] b) {
        checkSameLength(a, b);
        double[] result = pool.acquire(a.length);
        addInPlace(a, b, result);
        return result;
    }
    
    /**
     * Element-wise addition in place: result[i] = a[i] + b[i]
     * 
     * @param a first array
     * @param b second array
     * @param result output array
     */
    public static void addInPlace(double[] a, double[] b, double[] result) {
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            parallelBinaryOp(a, b, result, (x, y) -> x + y);
        } else {
            // Unrolled loop for better performance
            int i = 0;
            int limit = n - UNROLL_FACTOR + 1;
            
            for (; i < limit; i += UNROLL_FACTOR) {
                result[i] = a[i] + b[i];
                result[i + 1] = a[i + 1] + b[i + 1];
                result[i + 2] = a[i + 2] + b[i + 2];
                result[i + 3] = a[i + 3] + b[i + 3];
                result[i + 4] = a[i + 4] + b[i + 4];
                result[i + 5] = a[i + 5] + b[i + 5];
                result[i + 6] = a[i + 6] + b[i + 6];
                result[i + 7] = a[i + 7] + b[i + 7];
            }
            
            // Handle remaining elements
            for (; i < n; i++) {
                result[i] = a[i] + b[i];
            }
        }
    }
    
    /**
     * Element-wise subtraction: result[i] = a[i] - b[i]
     * 
     * @param a first array
     * @param b second array
     * @return result array
     */
    public static double[] subtract(double[] a, double[] b) {
        checkSameLength(a, b);
        double[] result = pool.acquire(a.length);
        subtractInPlace(a, b, result);
        return result;
    }
    
    /**
     * Element-wise subtraction in place.
     * 
     * @param a first array
     * @param b second array
     * @param result output array
     */
    public static void subtractInPlace(double[] a, double[] b, double[] result) {
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            parallelBinaryOp(a, b, result, (x, y) -> x - y);
        } else {
            int i = 0;
            int limit = n - UNROLL_FACTOR + 1;
            
            for (; i < limit; i += UNROLL_FACTOR) {
                result[i] = a[i] - b[i];
                result[i + 1] = a[i + 1] - b[i + 1];
                result[i + 2] = a[i + 2] - b[i + 2];
                result[i + 3] = a[i + 3] - b[i + 3];
                result[i + 4] = a[i + 4] - b[i + 4];
                result[i + 5] = a[i + 5] - b[i + 5];
                result[i + 6] = a[i + 6] - b[i + 6];
                result[i + 7] = a[i + 7] - b[i + 7];
            }
            
            for (; i < n; i++) {
                result[i] = a[i] - b[i];
            }
        }
    }
    
    /**
     * Element-wise multiplication (Hadamard product): result[i] = a[i] * b[i]
     * 
     * @param a first array
     * @param b second array
     * @return result array
     */
    public static double[] multiply(double[] a, double[] b) {
        checkSameLength(a, b);
        double[] result = pool.acquire(a.length);
        multiplyInPlace(a, b, result);
        return result;
    }
    
    /**
     * Element-wise multiplication in place.
     * 
     * @param a first array
     * @param b second array
     * @param result output array
     */
    public static void multiplyInPlace(double[] a, double[] b, double[] result) {
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            parallelBinaryOp(a, b, result, (x, y) -> x * y);
        } else {
            int i = 0;
            int limit = n - UNROLL_FACTOR + 1;
            
            for (; i < limit; i += UNROLL_FACTOR) {
                result[i] = a[i] * b[i];
                result[i + 1] = a[i + 1] * b[i + 1];
                result[i + 2] = a[i + 2] * b[i + 2];
                result[i + 3] = a[i + 3] * b[i + 3];
                result[i + 4] = a[i + 4] * b[i + 4];
                result[i + 5] = a[i + 5] * b[i + 5];
                result[i + 6] = a[i + 6] * b[i + 6];
                result[i + 7] = a[i + 7] * b[i + 7];
            }
            
            for (; i < n; i++) {
                result[i] = a[i] * b[i];
            }
        }
    }
    
    /**
     * Scalar multiplication: result[i] = a[i] * scalar
     * 
     * @param a input array
     * @param scalar scalar value
     * @return result array
     */
    public static double[] scale(double[] a, double scalar) {
        double[] result = pool.acquire(a.length);
        scaleInPlace(a, scalar, result);
        return result;
    }
    
    /**
     * Scalar multiplication in place.
     * 
     * @param a input array
     * @param scalar scalar value
     * @param result output array
     */
    public static void scaleInPlace(double[] a, double scalar, double[] result) {
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() -> 
                IntStream.range(0, n).parallel().forEach(i -> result[i] = a[i] * scalar)
            ).join();
        } else {
            int i = 0;
            int limit = n - UNROLL_FACTOR + 1;
            
            for (; i < limit; i += UNROLL_FACTOR) {
                result[i] = a[i] * scalar;
                result[i + 1] = a[i + 1] * scalar;
                result[i + 2] = a[i + 2] * scalar;
                result[i + 3] = a[i + 3] * scalar;
                result[i + 4] = a[i + 4] * scalar;
                result[i + 5] = a[i + 5] * scalar;
                result[i + 6] = a[i + 6] * scalar;
                result[i + 7] = a[i + 7] * scalar;
            }
            
            for (; i < n; i++) {
                result[i] = a[i] * scalar;
            }
        }
    }
    
    /**
     * Fused multiply-add: result[i] = a[i] * b[i] + c[i]
     * More efficient than separate multiply and add operations.
     * 
     * @param a first array
     * @param b second array
     * @param c third array
     * @return result array
     */
    public static double[] fma(double[] a, double[] b, double[] c) {
        checkSameLength(a, b);
        checkSameLength(a, c);
        double[] result = pool.acquire(a.length);
        fmaInPlace(a, b, c, result);
        return result;
    }
    
    /**
     * Fused multiply-add in place.
     * 
     * @param a first array
     * @param b second array
     * @param c third array
     * @param result output array
     */
    public static void fmaInPlace(double[] a, double[] b, double[] c, double[] result) {
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() -> 
                IntStream.range(0, n).parallel().forEach(i -> 
                    result[i] = Math.fma(a[i], b[i], c[i])
                )
            ).join();
        } else {
            int i = 0;
            int limit = n - 4 + 1;
            
            // Using Math.fma for optimal FMA instruction usage
            for (; i < limit; i += 4) {
                result[i] = Math.fma(a[i], b[i], c[i]);
                result[i + 1] = Math.fma(a[i + 1], b[i + 1], c[i + 1]);
                result[i + 2] = Math.fma(a[i + 2], b[i + 2], c[i + 2]);
                result[i + 3] = Math.fma(a[i + 3], b[i + 3], c[i + 3]);
            }
            
            for (; i < n; i++) {
                result[i] = Math.fma(a[i], b[i], c[i]);
            }
        }
    }
    
    // ==================== Reduction Operations ====================
    
    /**
     * Compute the dot product of two vectors.
     * Uses Kahan summation for numerical stability.
     * 
     * @param a first vector
     * @param b second vector
     * @return dot product
     */
    public static double dot(double[] a, double[] b) {
        checkSameLength(a, b);
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            return parallelDot(a, b);
        }
        
        // Kahan summation for better precision
        double sum = 0.0;
        double c = 0.0; // Compensation for lost low-order bits
        
        int i = 0;
        int limit = n - UNROLL_FACTOR + 1;
        
        for (; i < limit; i += UNROLL_FACTOR) {
            // Unrolled with local accumulation to reduce dependency
            double sum0 = a[i] * b[i] + a[i + 1] * b[i + 1];
            double sum1 = a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
            double sum2 = a[i + 4] * b[i + 4] + a[i + 5] * b[i + 5];
            double sum3 = a[i + 6] * b[i + 6] + a[i + 7] * b[i + 7];
            
            double y = (sum0 + sum1 + sum2 + sum3) - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        
        for (; i < n; i++) {
            double y = a[i] * b[i] - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        
        return sum;
    }
    
    /**
     * Parallel dot product implementation.
     */
    private static double parallelDot(double[] a, double[] b) {
        ForkJoinPool executor = config.getExecutor();
        int n = a.length;
        int numThreads = config.getNumThreads();
        int chunkSize = (n + numThreads - 1) / numThreads;
        
        return executor.submit(() -> 
            IntStream.range(0, numThreads)
                .parallel()
                .mapToDouble(t -> {
                    int start = t * chunkSize;
                    int end = Math.min(start + chunkSize, n);
                    double localSum = 0.0;
                    for (int i = start; i < end; i++) {
                        localSum += a[i] * b[i];
                    }
                    return localSum;
                })
                .sum()
        ).join();
    }
    
    /**
     * Compute the sum of all elements.
     * 
     * @param a input array
     * @return sum of elements
     */
    public static double sum(double[] a) {
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            ForkJoinPool executor = config.getExecutor();
            return executor.submit(() -> 
                IntStream.range(0, n).parallel().mapToDouble(i -> a[i]).sum()
            ).join();
        }
        
        // Kahan summation
        double sum = 0.0;
        double c = 0.0;
        
        for (int i = 0; i < n; i++) {
            double y = a[i] - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        
        return sum;
    }
    
    /**
     * Find the maximum value in an array.
     * 
     * @param a input array
     * @return maximum value
     */
    public static double max(double[] a) {
        if (a.length == 0) {
            return Double.NEGATIVE_INFINITY;
        }
        
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            ForkJoinPool executor = config.getExecutor();
            return executor.submit(() -> 
                IntStream.range(0, n).parallel().mapToDouble(i -> a[i]).max().orElse(Double.NEGATIVE_INFINITY)
            ).join();
        }
        
        double maxVal = a[0];
        for (int i = 1; i < n; i++) {
            if (a[i] > maxVal) {
                maxVal = a[i];
            }
        }
        return maxVal;
    }
    
    /**
     * Find the minimum value in an array.
     * 
     * @param a input array
     * @return minimum value
     */
    public static double min(double[] a) {
        if (a.length == 0) {
            return Double.POSITIVE_INFINITY;
        }
        
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            ForkJoinPool executor = config.getExecutor();
            return executor.submit(() -> 
                IntStream.range(0, n).parallel().mapToDouble(i -> a[i]).min().orElse(Double.POSITIVE_INFINITY)
            ).join();
        }
        
        double minVal = a[0];
        for (int i = 1; i < n; i++) {
            if (a[i] < minVal) {
                minVal = a[i];
            }
        }
        return minVal;
    }
    
    /**
     * Compute the L2 norm (Euclidean length) of a vector.
     * 
     * @param a input vector
     * @return L2 norm
     */
    public static double norm2(double[] a) {
        return Math.sqrt(dot(a, a));
    }
    
    /**
     * Compute the L1 norm (Manhattan length) of a vector.
     * 
     * @param a input vector
     * @return L1 norm
     */
    public static double norm1(double[] a) {
        int n = a.length;
        double sum = 0.0;
        
        for (int i = 0; i < n; i++) {
            sum += Math.abs(a[i]);
        }
        
        return sum;
    }
    
    // ==================== Transform Operations ====================
    
    /**
     * Apply sigmoid activation: result[i] = 1 / (1 + exp(-a[i]))
     * 
     * @param a input array
     * @return result array
     */
    public static double[] sigmoid(double[] a) {
        double[] result = pool.acquire(a.length);
        sigmoidInPlace(a, result);
        return result;
    }
    
    /**
     * Apply sigmoid activation in place.
     * 
     * @param a input array
     * @param result output array
     */
    public static void sigmoidInPlace(double[] a, double[] result) {
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() -> 
                IntStream.range(0, n).parallel().forEach(i -> 
                    result[i] = 1.0 / (1.0 + Math.exp(-a[i]))
                )
            ).join();
        } else {
            for (int i = 0; i < n; i++) {
                result[i] = 1.0 / (1.0 + Math.exp(-a[i]));
            }
        }
    }
    
    /**
     * Apply tanh activation.
     * 
     * @param a input array
     * @return result array
     */
    public static double[] tanh(double[] a) {
        double[] result = pool.acquire(a.length);
        tanhInPlace(a, result);
        return result;
    }
    
    /**
     * Apply tanh activation in place.
     * 
     * @param a input array
     * @param result output array
     */
    public static void tanhInPlace(double[] a, double[] result) {
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() -> 
                IntStream.range(0, n).parallel().forEach(i -> result[i] = Math.tanh(a[i]))
            ).join();
        } else {
            for (int i = 0; i < n; i++) {
                result[i] = Math.tanh(a[i]);
            }
        }
    }
    
    /**
     * Apply ReLU activation: result[i] = max(0, a[i])
     * 
     * @param a input array
     * @return result array
     */
    public static double[] relu(double[] a) {
        double[] result = pool.acquire(a.length);
        reluInPlace(a, result);
        return result;
    }
    
    /**
     * Apply ReLU activation in place.
     * 
     * @param a input array
     * @param result output array
     */
    public static void reluInPlace(double[] a, double[] result) {
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() -> 
                IntStream.range(0, n).parallel().forEach(i -> 
                    result[i] = a[i] > 0 ? a[i] : 0
                )
            ).join();
        } else {
            int i = 0;
            int limit = n - UNROLL_FACTOR + 1;
            
            for (; i < limit; i += UNROLL_FACTOR) {
                result[i] = a[i] > 0 ? a[i] : 0;
                result[i + 1] = a[i + 1] > 0 ? a[i + 1] : 0;
                result[i + 2] = a[i + 2] > 0 ? a[i + 2] : 0;
                result[i + 3] = a[i + 3] > 0 ? a[i + 3] : 0;
                result[i + 4] = a[i + 4] > 0 ? a[i + 4] : 0;
                result[i + 5] = a[i + 5] > 0 ? a[i + 5] : 0;
                result[i + 6] = a[i + 6] > 0 ? a[i + 6] : 0;
                result[i + 7] = a[i + 7] > 0 ? a[i + 7] : 0;
            }
            
            for (; i < n; i++) {
                result[i] = a[i] > 0 ? a[i] : 0;
            }
        }
    }
    
    /**
     * Apply softmax normalization.
     * 
     * @param a input array
     * @return normalized array
     */
    public static double[] softmax(double[] a) {
        double[] result = pool.acquire(a.length);
        softmaxInPlace(a, result);
        return result;
    }
    
    /**
     * Apply softmax normalization in place.
     * 
     * @param a input array
     * @param result output array
     */
    public static void softmaxInPlace(double[] a, double[] result) {
        int n = a.length;
        
        // Find max for numerical stability
        double maxVal = max(a);
        
        // Compute exp(a[i] - max)
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            result[i] = Math.exp(a[i] - maxVal);
            sum += result[i];
        }
        
        // Normalize
        double invSum = 1.0 / sum;
        scaleInPlace(result, invSum, result);
    }
    
    // ==================== Utility Methods ====================
    
    /**
     * Copy array values.
     * 
     * @param src source array
     * @param dst destination array
     */
    public static void copy(double[] src, double[] dst) {
        System.arraycopy(src, 0, dst, 0, Math.min(src.length, dst.length));
    }
    
    /**
     * Fill array with a value.
     * 
     * @param a array to fill
     * @param value value to fill with
     */
    public static void fill(double[] a, double value) {
        int n = a.length;
        
        if (config.shouldParallelize(n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() -> 
                IntStream.range(0, n).parallel().forEach(i -> a[i] = value)
            ).join();
        } else {
            java.util.Arrays.fill(a, value);
        }
    }
    
    /**
     * Clip values to a range.
     * 
     * @param a input array
     * @param minVal minimum value
     * @param maxVal maximum value
     * @return clipped array
     */
    public static double[] clip(double[] a, double minVal, double maxVal) {
        double[] result = pool.acquire(a.length);
        clipInPlace(a, minVal, maxVal, result);
        return result;
    }
    
    /**
     * Clip values to a range in place.
     * 
     * @param a input array
     * @param minVal minimum value
     * @param maxVal maximum value
     * @param result output array
     */
    public static void clipInPlace(double[] a, double minVal, double maxVal, double[] result) {
        int n = a.length;
        
        for (int i = 0; i < n; i++) {
            result[i] = Math.max(minVal, Math.min(maxVal, a[i]));
        }
    }
    
    // ==================== Helper Methods ====================
    
    @FunctionalInterface
    private interface BinaryOp {
        double apply(double a, double b);
    }
    
    private static void parallelBinaryOp(double[] a, double[] b, double[] result, BinaryOp op) {
        ForkJoinPool executor = config.getExecutor();
        int n = a.length;
        
        executor.submit(() -> 
            IntStream.range(0, n).parallel().forEach(i -> result[i] = op.apply(a[i], b[i]))
        ).join();
    }
    
    private static void checkSameLength(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                "Array length mismatch: " + a.length + " vs " + b.length);
        }
    }
}
