package com.mindforge.acceleration;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.stream.IntStream;

/**
 * High-performance parallel matrix operations.
 * 
 * Provides cache-optimized, multi-threaded implementations of common matrix
 * operations including multiplication, transposition, and element-wise operations.
 * 
 * <p>Uses blocked algorithms for better cache utilization and automatic
 * parallelization based on matrix dimensions.</p>
 * 
 * @author MindForge Team
 * @version 1.2.0
 */
public class ParallelMatrix {
    
    private static final AccelerationConfig config = AccelerationConfig.getInstance();
    private static final MemoryPool pool = MemoryPool.getInstance();
    
    // Private constructor - utility class
    private ParallelMatrix() {}
    
    // ==================== Matrix Multiplication ====================
    
    /**
     * Matrix multiplication: C = A * B
     * 
     * Uses cache-blocked algorithm for better performance on large matrices.
     * 
     * @param a matrix A [m x k]
     * @param b matrix B [k x n]
     * @return result matrix C [m x n]
     */
    public static double[][] multiply(double[][] a, double[][] b) {
        int m = a.length;
        int k = a[0].length;
        int n = b[0].length;
        
        if (b.length != k) {
            throw new IllegalArgumentException(
                "Matrix dimension mismatch: A is " + m + "x" + k + 
                ", B is " + b.length + "x" + n);
        }
        
        double[][] c = pool.acquire2D(m, n);
        
        if (config.shouldParallelizeMatMul(m, n, k)) {
            multiplyParallelBlocked(a, b, c, m, k, n);
        } else {
            multiplySequentialBlocked(a, b, c, m, k, n);
        }
        
        return c;
    }
    
    /**
     * Matrix multiplication with result provided: C = A * B
     * 
     * @param a matrix A [m x k]
     * @param b matrix B [k x n]
     * @param c result matrix C [m x n] (must be pre-allocated)
     */
    public static void multiplyInPlace(double[][] a, double[][] b, double[][] c) {
        int m = a.length;
        int k = a[0].length;
        int n = b[0].length;
        
        // Clear result
        for (int i = 0; i < m; i++) {
            java.util.Arrays.fill(c[i], 0.0);
        }
        
        if (config.shouldParallelizeMatMul(m, n, k)) {
            multiplyParallelBlocked(a, b, c, m, k, n);
        } else {
            multiplySequentialBlocked(a, b, c, m, k, n);
        }
    }
    
    /**
     * Sequential blocked matrix multiplication.
     * Uses cache-friendly block access pattern.
     */
    private static void multiplySequentialBlocked(double[][] a, double[][] b, 
                                                   double[][] c, int m, int k, int n) {
        int blockSize = config.getBlockSize();
        
        // Blocked matrix multiplication (cache-optimized)
        for (int ii = 0; ii < m; ii += blockSize) {
            int iMax = Math.min(ii + blockSize, m);
            
            for (int kk = 0; kk < k; kk += blockSize) {
                int kMax = Math.min(kk + blockSize, k);
                
                for (int jj = 0; jj < n; jj += blockSize) {
                    int jMax = Math.min(jj + blockSize, n);
                    
                    // Multiply block
                    for (int i = ii; i < iMax; i++) {
                        double[] aRow = a[i];
                        double[] cRow = c[i];
                        
                        for (int j = jj; j < jMax; j++) {
                            double sum = cRow[j];
                            
                            for (int kIdx = kk; kIdx < kMax; kIdx++) {
                                sum += aRow[kIdx] * b[kIdx][j];
                            }
                            
                            cRow[j] = sum;
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Parallel blocked matrix multiplication.
     */
    private static void multiplyParallelBlocked(double[][] a, double[][] b,
                                                double[][] c, int m, int k, int n) {
        ForkJoinPool executor = config.getExecutor();
        int blockSize = config.getBlockSize();
        
        // Parallelize over rows of C
        executor.invoke(new MatrixMultiplyTask(a, b, c, 0, m, k, n, blockSize));
    }
    
    /**
     * RecursiveAction for parallel matrix multiplication.
     */
    private static class MatrixMultiplyTask extends RecursiveAction {
        private static final int SEQUENTIAL_THRESHOLD = 64;
        
        private final double[][] a;
        private final double[][] b;
        private final double[][] c;
        private final int startRow;
        private final int endRow;
        private final int k;
        private final int n;
        private final int blockSize;
        
        MatrixMultiplyTask(double[][] a, double[][] b, double[][] c,
                          int startRow, int endRow, int k, int n, int blockSize) {
            this.a = a;
            this.b = b;
            this.c = c;
            this.startRow = startRow;
            this.endRow = endRow;
            this.k = k;
            this.n = n;
            this.blockSize = blockSize;
        }
        
        @Override
        protected void compute() {
            int numRows = endRow - startRow;
            
            if (numRows <= SEQUENTIAL_THRESHOLD) {
                // Sequential blocked multiply for this range
                for (int i = startRow; i < endRow; i++) {
                    double[] aRow = a[i];
                    double[] cRow = c[i];
                    
                    for (int jj = 0; jj < n; jj += blockSize) {
                        int jMax = Math.min(jj + blockSize, n);
                        
                        for (int kk = 0; kk < k; kk += blockSize) {
                            int kMax = Math.min(kk + blockSize, k);
                            
                            for (int j = jj; j < jMax; j++) {
                                double sum = cRow[j];
                                for (int kIdx = kk; kIdx < kMax; kIdx++) {
                                    sum += aRow[kIdx] * b[kIdx][j];
                                }
                                cRow[j] = sum;
                            }
                        }
                    }
                }
            } else {
                // Split and parallelize
                int mid = startRow + numRows / 2;
                invokeAll(
                    new MatrixMultiplyTask(a, b, c, startRow, mid, k, n, blockSize),
                    new MatrixMultiplyTask(a, b, c, mid, endRow, k, n, blockSize)
                );
            }
        }
    }
    
    // ==================== Matrix-Vector Operations ====================
    
    /**
     * Matrix-vector multiplication: y = A * x
     * 
     * @param a matrix A [m x n]
     * @param x vector x [n]
     * @return result vector y [m]
     */
    public static double[] multiplyVector(double[][] a, double[] x) {
        int m = a.length;
        int n = a[0].length;
        
        if (x.length != n) {
            throw new IllegalArgumentException(
                "Dimension mismatch: matrix has " + n + " columns, vector has " + x.length + " elements");
        }
        
        double[] y = pool.acquire(m);
        multiplyVectorInPlace(a, x, y);
        return y;
    }
    
    /**
     * Matrix-vector multiplication in place: y = A * x
     * 
     * @param a matrix A [m x n]
     * @param x vector x [n]
     * @param y result vector y [m] (must be pre-allocated)
     */
    public static void multiplyVectorInPlace(double[][] a, double[] x, double[] y) {
        int m = a.length;
        int n = a[0].length;
        
        if (config.shouldParallelize(m * n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() ->
                IntStream.range(0, m).parallel().forEach(i -> {
                    double sum = 0.0;
                    double[] aRow = a[i];
                    for (int j = 0; j < n; j++) {
                        sum += aRow[j] * x[j];
                    }
                    y[i] = sum;
                })
            ).join();
        } else {
            // Sequential with unrolling
            for (int i = 0; i < m; i++) {
                double[] aRow = a[i];
                double sum = 0.0;
                
                int j = 0;
                int limit = n - 4 + 1;
                
                for (; j < limit; j += 4) {
                    sum += aRow[j] * x[j] + aRow[j + 1] * x[j + 1] +
                           aRow[j + 2] * x[j + 2] + aRow[j + 3] * x[j + 3];
                }
                
                for (; j < n; j++) {
                    sum += aRow[j] * x[j];
                }
                
                y[i] = sum;
            }
        }
    }
    
    /**
     * Vector-matrix multiplication: y = x^T * A
     * 
     * @param x vector x [m]
     * @param a matrix A [m x n]
     * @return result vector y [n]
     */
    public static double[] vectorMultiply(double[] x, double[][] a) {
        int m = a.length;
        int n = a[0].length;
        
        if (x.length != m) {
            throw new IllegalArgumentException(
                "Dimension mismatch: vector has " + x.length + 
                " elements, matrix has " + m + " rows");
        }
        
        double[] y = pool.acquire(n);
        java.util.Arrays.fill(y, 0.0);
        
        if (config.shouldParallelize(m * n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() ->
                IntStream.range(0, n).parallel().forEach(j -> {
                    double sum = 0.0;
                    for (int i = 0; i < m; i++) {
                        sum += x[i] * a[i][j];
                    }
                    y[j] = sum;
                })
            ).join();
        } else {
            for (int i = 0; i < m; i++) {
                double xi = x[i];
                double[] aRow = a[i];
                for (int j = 0; j < n; j++) {
                    y[j] += xi * aRow[j];
                }
            }
        }
        
        return y;
    }
    
    // ==================== Outer Product ====================
    
    /**
     * Outer product: C = a * b^T
     * 
     * @param a vector a [m]
     * @param b vector b [n]
     * @return result matrix C [m x n]
     */
    public static double[][] outerProduct(double[] a, double[] b) {
        int m = a.length;
        int n = b.length;
        
        double[][] c = pool.acquire2D(m, n);
        outerProductInPlace(a, b, c);
        return c;
    }
    
    /**
     * Outer product in place: C = a * b^T
     * 
     * @param a vector a [m]
     * @param b vector b [n]
     * @param c result matrix C [m x n]
     */
    public static void outerProductInPlace(double[] a, double[] b, double[][] c) {
        int m = a.length;
        int n = b.length;
        
        if (config.shouldParallelize(m * n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() ->
                IntStream.range(0, m).parallel().forEach(i -> {
                    double ai = a[i];
                    double[] cRow = c[i];
                    for (int j = 0; j < n; j++) {
                        cRow[j] = ai * b[j];
                    }
                })
            ).join();
        } else {
            for (int i = 0; i < m; i++) {
                double ai = a[i];
                double[] cRow = c[i];
                for (int j = 0; j < n; j++) {
                    cRow[j] = ai * b[j];
                }
            }
        }
    }
    
    // ==================== Matrix Transpose ====================
    
    /**
     * Matrix transpose: B = A^T
     * 
     * @param a input matrix [m x n]
     * @return transposed matrix [n x m]
     */
    public static double[][] transpose(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        
        double[][] b = pool.acquire2D(n, m);
        transposeInPlace(a, b);
        return b;
    }
    
    /**
     * Matrix transpose in place: B = A^T
     * Uses cache-blocked algorithm for better performance.
     * 
     * @param a input matrix [m x n]
     * @param b output matrix [n x m]
     */
    public static void transposeInPlace(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        int blockSize = config.getBlockSize();
        
        if (config.shouldParallelize(m * n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() ->
                IntStream.range(0, (m + blockSize - 1) / blockSize).parallel().forEach(ii -> {
                    int iStart = ii * blockSize;
                    int iEnd = Math.min(iStart + blockSize, m);
                    
                    for (int jj = 0; jj < n; jj += blockSize) {
                        int jEnd = Math.min(jj + blockSize, n);
                        
                        for (int i = iStart; i < iEnd; i++) {
                            for (int j = jj; j < jEnd; j++) {
                                b[j][i] = a[i][j];
                            }
                        }
                    }
                })
            ).join();
        } else {
            // Blocked transpose for cache efficiency
            for (int ii = 0; ii < m; ii += blockSize) {
                int iEnd = Math.min(ii + blockSize, m);
                
                for (int jj = 0; jj < n; jj += blockSize) {
                    int jEnd = Math.min(jj + blockSize, n);
                    
                    for (int i = ii; i < iEnd; i++) {
                        for (int j = jj; j < jEnd; j++) {
                            b[j][i] = a[i][j];
                        }
                    }
                }
            }
        }
    }
    
    // ==================== Element-wise Operations ====================
    
    /**
     * Element-wise addition: C = A + B
     * 
     * @param a matrix A
     * @param b matrix B
     * @return result matrix C
     */
    public static double[][] add(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        
        double[][] c = pool.acquire2D(m, n);
        addInPlace(a, b, c);
        return c;
    }
    
    /**
     * Element-wise addition in place.
     * 
     * @param a matrix A
     * @param b matrix B
     * @param c result matrix C
     */
    public static void addInPlace(double[][] a, double[][] b, double[][] c) {
        int m = a.length;
        int n = a[0].length;
        
        if (config.shouldParallelize(m * n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() ->
                IntStream.range(0, m).parallel().forEach(i -> {
                    double[] aRow = a[i];
                    double[] bRow = b[i];
                    double[] cRow = c[i];
                    for (int j = 0; j < n; j++) {
                        cRow[j] = aRow[j] + bRow[j];
                    }
                })
            ).join();
        } else {
            for (int i = 0; i < m; i++) {
                double[] aRow = a[i];
                double[] bRow = b[i];
                double[] cRow = c[i];
                for (int j = 0; j < n; j++) {
                    cRow[j] = aRow[j] + bRow[j];
                }
            }
        }
    }
    
    /**
     * Element-wise subtraction: C = A - B
     * 
     * @param a matrix A
     * @param b matrix B
     * @return result matrix C
     */
    public static double[][] subtract(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        
        double[][] c = pool.acquire2D(m, n);
        
        if (config.shouldParallelize(m * n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() ->
                IntStream.range(0, m).parallel().forEach(i -> {
                    for (int j = 0; j < n; j++) {
                        c[i][j] = a[i][j] - b[i][j];
                    }
                })
            ).join();
        } else {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    c[i][j] = a[i][j] - b[i][j];
                }
            }
        }
        
        return c;
    }
    
    /**
     * Scalar multiplication: B = scalar * A
     * 
     * @param a matrix A
     * @param scalar scalar value
     * @return result matrix B
     */
    public static double[][] scale(double[][] a, double scalar) {
        int m = a.length;
        int n = a[0].length;
        
        double[][] b = pool.acquire2D(m, n);
        
        if (config.shouldParallelize(m * n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() ->
                IntStream.range(0, m).parallel().forEach(i -> {
                    for (int j = 0; j < n; j++) {
                        b[i][j] = a[i][j] * scalar;
                    }
                })
            ).join();
        } else {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    b[i][j] = a[i][j] * scalar;
                }
            }
        }
        
        return b;
    }
    
    /**
     * Hadamard (element-wise) product: C = A .* B
     * 
     * @param a matrix A
     * @param b matrix B
     * @return result matrix C
     */
    public static double[][] hadamard(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        
        double[][] c = pool.acquire2D(m, n);
        
        if (config.shouldParallelize(m * n)) {
            ForkJoinPool executor = config.getExecutor();
            executor.submit(() ->
                IntStream.range(0, m).parallel().forEach(i -> {
                    for (int j = 0; j < n; j++) {
                        c[i][j] = a[i][j] * b[i][j];
                    }
                })
            ).join();
        } else {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    c[i][j] = a[i][j] * b[i][j];
                }
            }
        }
        
        return c;
    }
    
    // ==================== Utility Methods ====================
    
    /**
     * Sum all elements in a matrix.
     * 
     * @param a input matrix
     * @return sum of all elements
     */
    public static double sum(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        
        if (config.shouldParallelize(m * n)) {
            ForkJoinPool executor = config.getExecutor();
            return executor.submit(() ->
                IntStream.range(0, m).parallel()
                    .mapToDouble(i -> {
                        double rowSum = 0.0;
                        for (int j = 0; j < n; j++) {
                            rowSum += a[i][j];
                        }
                        return rowSum;
                    })
                    .sum()
            ).join();
        } else {
            double sum = 0.0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    sum += a[i][j];
                }
            }
            return sum;
        }
    }
    
    /**
     * Frobenius norm: sqrt(sum of squared elements)
     * 
     * @param a input matrix
     * @return Frobenius norm
     */
    public static double frobeniusNorm(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        
        double sumSq = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sumSq += a[i][j] * a[i][j];
            }
        }
        
        return Math.sqrt(sumSq);
    }
    
    /**
     * Create a copy of a matrix.
     * 
     * @param a input matrix
     * @return copy of the matrix
     */
    public static double[][] copy(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        
        double[][] b = pool.acquire2D(m, n);
        for (int i = 0; i < m; i++) {
            System.arraycopy(a[i], 0, b[i], 0, n);
        }
        
        return b;
    }
    
    /**
     * Fill a matrix with a value.
     * 
     * @param a matrix to fill
     * @param value value to fill with
     */
    public static void fill(double[][] a, double value) {
        for (double[] row : a) {
            java.util.Arrays.fill(row, value);
        }
    }
    
    /**
     * Create an identity matrix.
     * 
     * @param n size of the matrix
     * @return n x n identity matrix
     */
    public static double[][] identity(int n) {
        double[][] a = pool.acquire2D(n, n);
        for (int i = 0; i < n; i++) {
            java.util.Arrays.fill(a[i], 0.0);
            a[i][i] = 1.0;
        }
        return a;
    }
    
    /**
     * Create a zero matrix.
     * 
     * @param m number of rows
     * @param n number of columns
     * @return m x n zero matrix
     */
    public static double[][] zeros(int m, int n) {
        double[][] a = pool.acquire2D(m, n);
        for (int i = 0; i < m; i++) {
            java.util.Arrays.fill(a[i], 0.0);
        }
        return a;
    }
}
