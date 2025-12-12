package io.github.yasmramos.mindforge.acceleration;

import java.util.concurrent.ForkJoinPool;

/**
 * Global configuration for computational acceleration in MindForge.
 * 
 * Provides centralized control over parallelization settings, memory optimization,
 * and hardware-specific tuning parameters.
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * AccelerationConfig config = AccelerationConfig.getInstance();
 * config.setNumThreads(8);
 * config.enableMemoryPooling(true);
 * config.setBlockSize(64); // For cache optimization
 * }</pre>
 * 
 * @author MindForge Team
 * @version 1.2.0
 */
public class AccelerationConfig {
    
    private static volatile AccelerationConfig instance;
    private static final Object lock = new Object();
    
    // Thread pool configuration
    private int numThreads;
    private ForkJoinPool executor;
    private boolean useParallelization;
    
    // Memory optimization
    private boolean memoryPoolingEnabled;
    private int maxPooledArrays;
    private long maxPooledMemoryBytes;
    
    // Block operations for cache optimization
    private int blockSize;
    private int vectorSize;
    
    // Thresholds for parallel execution
    private int parallelThreshold;
    private int matrixMultiplyThreshold;
    private int convolutionThreshold;
    
    // Hardware detection
    private final int availableProcessors;
    private final long maxMemory;
    private boolean simdAvailable;
    
    /**
     * Private constructor for singleton pattern.
     */
    private AccelerationConfig() {
        this.availableProcessors = Runtime.getRuntime().availableProcessors();
        this.maxMemory = Runtime.getRuntime().maxMemory();
        
        // Default settings based on hardware
        this.numThreads = Math.max(1, availableProcessors - 1);
        this.useParallelization = availableProcessors > 1;
        
        // Memory pooling defaults
        this.memoryPoolingEnabled = true;
        this.maxPooledArrays = 1000;
        this.maxPooledMemoryBytes = maxMemory / 4; // 25% of max heap
        
        // Block size tuning (cache line is typically 64 bytes = 8 doubles)
        this.blockSize = 64;
        this.vectorSize = 8;
        
        // Thresholds for parallel execution
        this.parallelThreshold = 10000;       // Elements before parallelizing
        this.matrixMultiplyThreshold = 64;    // Matrix dimension threshold
        this.convolutionThreshold = 32;       // Output size threshold
        
        // Check for SIMD availability (approximation)
        this.simdAvailable = checkSimdAvailability();
        
        // Create thread pool
        this.executor = new ForkJoinPool(numThreads);
    }
    
    /**
     * Get the singleton instance of AccelerationConfig.
     * 
     * @return the singleton instance
     */
    public static AccelerationConfig getInstance() {
        if (instance == null) {
            synchronized (lock) {
                if (instance == null) {
                    instance = new AccelerationConfig();
                }
            }
        }
        return instance;
    }
    
    /**
     * Reset configuration to defaults. Useful for testing.
     * This method reconfigures the current instance instead of creating a new one.
     */
    public synchronized void reset() {
        // Shutdown existing executor if running
        if (executor != null && !executor.isShutdown()) {
            executor.shutdown();
        }
        
        // Reset to default values
        this.numThreads = Math.max(1, availableProcessors - 1);
        this.useParallelization = availableProcessors > 1;
        this.memoryPoolingEnabled = true;
        this.maxPooledArrays = 1000;
        this.maxPooledMemoryBytes = maxMemory / 4;
        this.blockSize = 64;
        this.vectorSize = 8;
        this.parallelThreshold = 10000;
        this.matrixMultiplyThreshold = 64;
        this.convolutionThreshold = 32;
        
        // Create new executor
        this.executor = new ForkJoinPool(numThreads);
    }
    
    /**
     * Check if SIMD instructions are likely available.
     * Java doesn't provide direct SIMD access, but we can optimize for vectorization.
     */
    private boolean checkSimdAvailability() {
        // JVM may auto-vectorize loops on modern processors
        String arch = System.getProperty("os.arch", "");
        return arch.contains("64") || arch.contains("amd64") || arch.contains("x86_64");
    }
    
    // ==================== Thread Pool Configuration ====================
    
    /**
     * Set the number of threads for parallel operations.
     * 
     * @param numThreads number of threads (must be >= 1)
     */
    public synchronized void setNumThreads(int numThreads) {
        if (numThreads < 1) {
            throw new IllegalArgumentException("Number of threads must be at least 1");
        }
        this.numThreads = numThreads;
        
        // Recreate thread pool
        if (executor != null && !executor.isShutdown()) {
            executor.shutdown();
        }
        this.executor = new ForkJoinPool(numThreads);
    }
    
    /**
     * Get the number of threads for parallel operations.
     * 
     * @return number of threads
     */
    public int getNumThreads() {
        return numThreads;
    }
    
    /**
     * Get the ForkJoinPool executor for parallel tasks.
     * 
     * @return the executor
     */
    public ForkJoinPool getExecutor() {
        return executor;
    }
    
    /**
     * Enable or disable parallelization.
     * 
     * @param enabled true to enable parallel execution
     */
    public void setUseParallelization(boolean enabled) {
        this.useParallelization = enabled;
    }
    
    /**
     * Check if parallelization is enabled.
     * 
     * @return true if parallel execution is enabled
     */
    public boolean isParallelizationEnabled() {
        return useParallelization && numThreads > 1;
    }
    
    // ==================== Memory Configuration ====================
    
    /**
     * Enable or disable memory pooling.
     * 
     * @param enabled true to enable memory pooling
     */
    public void enableMemoryPooling(boolean enabled) {
        this.memoryPoolingEnabled = enabled;
    }
    
    /**
     * Check if memory pooling is enabled.
     * 
     * @return true if memory pooling is enabled
     */
    public boolean isMemoryPoolingEnabled() {
        return memoryPoolingEnabled;
    }
    
    /**
     * Set maximum number of arrays to pool.
     * 
     * @param maxArrays maximum arrays per size bucket
     */
    public void setMaxPooledArrays(int maxArrays) {
        this.maxPooledArrays = maxArrays;
    }
    
    /**
     * Get maximum number of pooled arrays.
     * 
     * @return maximum arrays per size bucket
     */
    public int getMaxPooledArrays() {
        return maxPooledArrays;
    }
    
    /**
     * Set maximum memory for pooled arrays.
     * 
     * @param bytes maximum bytes for pooling
     */
    public void setMaxPooledMemoryBytes(long bytes) {
        this.maxPooledMemoryBytes = bytes;
    }
    
    /**
     * Get maximum memory for pooled arrays.
     * 
     * @return maximum bytes for pooling
     */
    public long getMaxPooledMemoryBytes() {
        return maxPooledMemoryBytes;
    }
    
    // ==================== Block Size Configuration ====================
    
    /**
     * Set block size for cache-optimized operations.
     * 
     * @param blockSize block size (should be power of 2, typically 32-128)
     */
    public void setBlockSize(int blockSize) {
        if (blockSize < 1) {
            throw new IllegalArgumentException("Block size must be positive");
        }
        this.blockSize = blockSize;
    }
    
    /**
     * Get block size for cache-optimized operations.
     * 
     * @return block size
     */
    public int getBlockSize() {
        return blockSize;
    }
    
    /**
     * Set vector size for loop unrolling.
     * 
     * @param vectorSize vector size (typically 4 or 8)
     */
    public void setVectorSize(int vectorSize) {
        if (vectorSize < 1) {
            throw new IllegalArgumentException("Vector size must be positive");
        }
        this.vectorSize = vectorSize;
    }
    
    /**
     * Get vector size for loop unrolling.
     * 
     * @return vector size
     */
    public int getVectorSize() {
        return vectorSize;
    }
    
    // ==================== Threshold Configuration ====================
    
    /**
     * Set threshold for parallel execution.
     * Operations with fewer elements will run sequentially.
     * 
     * @param threshold minimum elements for parallel execution
     */
    public void setParallelThreshold(int threshold) {
        this.parallelThreshold = threshold;
    }
    
    /**
     * Get threshold for parallel execution.
     * 
     * @return minimum elements for parallel execution
     */
    public int getParallelThreshold() {
        return parallelThreshold;
    }
    
    /**
     * Set threshold for parallel matrix multiplication.
     * 
     * @param threshold minimum matrix dimension for parallel multiplication
     */
    public void setMatrixMultiplyThreshold(int threshold) {
        this.matrixMultiplyThreshold = threshold;
    }
    
    /**
     * Get threshold for parallel matrix multiplication.
     * 
     * @return minimum matrix dimension for parallel multiplication
     */
    public int getMatrixMultiplyThreshold() {
        return matrixMultiplyThreshold;
    }
    
    /**
     * Set threshold for parallel convolution.
     * 
     * @param threshold minimum output size for parallel convolution
     */
    public void setConvolutionThreshold(int threshold) {
        this.convolutionThreshold = threshold;
    }
    
    /**
     * Get threshold for parallel convolution.
     * 
     * @return minimum output size for parallel convolution
     */
    public int getConvolutionThreshold() {
        return convolutionThreshold;
    }
    
    // ==================== Hardware Information ====================
    
    /**
     * Get number of available processors.
     * 
     * @return number of processors
     */
    public int getAvailableProcessors() {
        return availableProcessors;
    }
    
    /**
     * Get maximum available memory.
     * 
     * @return maximum memory in bytes
     */
    public long getMaxMemory() {
        return maxMemory;
    }
    
    /**
     * Check if SIMD optimization is available.
     * 
     * @return true if SIMD may be available
     */
    public boolean isSimdAvailable() {
        return simdAvailable;
    }
    
    /**
     * Check if an operation should be parallelized based on size.
     * 
     * @param elements number of elements in the operation
     * @return true if operation should be parallelized
     */
    public boolean shouldParallelize(int elements) {
        return useParallelization && elements >= parallelThreshold;
    }
    
    /**
     * Check if matrix multiplication should be parallelized.
     * 
     * @param m rows of first matrix
     * @param n columns of second matrix
     * @param k common dimension
     * @return true if should be parallelized
     */
    public boolean shouldParallelizeMatMul(int m, int n, int k) {
        return useParallelization && 
               (m >= matrixMultiplyThreshold || n >= matrixMultiplyThreshold);
    }
    
    /**
     * Shutdown the executor. Call this when done using acceleration.
     */
    public void shutdown() {
        if (executor != null && !executor.isShutdown()) {
            executor.shutdown();
        }
    }
    
    @Override
    public String toString() {
        return String.format(
            "AccelerationConfig{threads=%d, parallelization=%s, memoryPooling=%s, " +
            "blockSize=%d, vectorSize=%d, processors=%d, maxMemory=%dMB, simd=%s}",
            numThreads, useParallelization, memoryPoolingEnabled,
            blockSize, vectorSize, availableProcessors, maxMemory / (1024 * 1024), simdAvailable
        );
    }
}
