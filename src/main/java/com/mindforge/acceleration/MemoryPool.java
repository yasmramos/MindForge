package com.mindforge.acceleration;

import java.util.Arrays;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Memory pool for reusing arrays to reduce garbage collection overhead.
 * 
 * Provides efficient allocation and deallocation of double arrays for
 * computational operations. Arrays are pooled by size bucket and reused
 * to minimize GC pressure during intensive computations.
 * 
 * <p>This pool is thread-safe and designed for concurrent access.</p>
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * MemoryPool pool = MemoryPool.getInstance();
 * 
 * // Acquire an array
 * double[] temp = pool.acquire(1000);
 * 
 * // Use the array...
 * 
 * // Release back to pool
 * pool.release(temp);
 * }</pre>
 * 
 * @author MindForge Team
 * @version 1.2.0
 */
public class MemoryPool {
    
    private static volatile MemoryPool instance;
    private static final Object lock = new Object();
    
    // Size buckets: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, etc.
    private static final int MIN_BUCKET_SIZE = 64;
    private static final int MAX_BUCKET_SIZE = 1 << 20; // 1M elements
    private static final int BUCKET_COUNT = 15; // log2(MAX/MIN) + 1
    
    // Pool storage: bucketIndex -> queue of arrays
    private final Map<Integer, Queue<double[]>> pools;
    private final Map<Integer, Queue<double[][]>> pools2D;
    private final Map<Integer, Queue<int[]>> intPools;
    
    // Statistics
    private final AtomicLong totalAllocations;
    private final AtomicLong poolHits;
    private final AtomicLong poolMisses;
    private final AtomicLong currentPooledBytes;
    
    // Configuration
    private final AccelerationConfig config;
    private volatile boolean enabled;
    
    /**
     * Private constructor for singleton pattern.
     */
    private MemoryPool() {
        this.pools = new ConcurrentHashMap<>();
        this.pools2D = new ConcurrentHashMap<>();
        this.intPools = new ConcurrentHashMap<>();
        
        this.totalAllocations = new AtomicLong(0);
        this.poolHits = new AtomicLong(0);
        this.poolMisses = new AtomicLong(0);
        this.currentPooledBytes = new AtomicLong(0);
        
        this.config = AccelerationConfig.getInstance();
        this.enabled = config.isMemoryPoolingEnabled();
        
        // Initialize bucket queues
        for (int i = 0; i < BUCKET_COUNT; i++) {
            int bucketKey = getBucketSize(i);
            pools.put(bucketKey, new ConcurrentLinkedQueue<>());
            pools2D.put(bucketKey, new ConcurrentLinkedQueue<>());
            intPools.put(bucketKey, new ConcurrentLinkedQueue<>());
        }
    }
    
    /**
     * Get the singleton instance of MemoryPool.
     * 
     * @return the singleton instance
     */
    public static MemoryPool getInstance() {
        if (instance == null) {
            synchronized (lock) {
                if (instance == null) {
                    instance = new MemoryPool();
                }
            }
        }
        return instance;
    }
    
    /**
     * Reset the memory pool. Clears all pooled arrays.
     */
    public synchronized void reset() {
        pools.values().forEach(Queue::clear);
        pools2D.values().forEach(Queue::clear);
        intPools.values().forEach(Queue::clear);
        
        totalAllocations.set(0);
        poolHits.set(0);
        poolMisses.set(0);
        currentPooledBytes.set(0);
    }
    
    /**
     * Get the bucket size for a given bucket index.
     */
    private int getBucketSize(int bucketIndex) {
        return MIN_BUCKET_SIZE << bucketIndex;
    }
    
    /**
     * Get the bucket index for a given size.
     */
    private int getBucketIndex(int size) {
        if (size <= MIN_BUCKET_SIZE) {
            return 0;
        }
        if (size > MAX_BUCKET_SIZE) {
            return -1; // Too large for pooling
        }
        
        // Find the next power of 2 >= size
        int highBit = 32 - Integer.numberOfLeadingZeros(size - 1);
        int bucketSize = 1 << highBit;
        
        // Calculate bucket index
        return highBit - 6; // MIN_BUCKET_SIZE = 64 = 2^6
    }
    
    /**
     * Get the actual bucket size for a requested size.
     */
    private int getActualBucketSize(int requestedSize) {
        if (requestedSize <= MIN_BUCKET_SIZE) {
            return MIN_BUCKET_SIZE;
        }
        if (requestedSize > MAX_BUCKET_SIZE) {
            return requestedSize; // Not pooled
        }
        
        // Round up to next power of 2
        int highBit = 32 - Integer.numberOfLeadingZeros(requestedSize - 1);
        return 1 << highBit;
    }
    
    // ==================== Double Array Operations ====================
    
    /**
     * Acquire a double array of at least the specified size.
     * May return a larger array from the pool.
     * 
     * @param size minimum required size
     * @return a double array (may be larger than requested)
     */
    public double[] acquire(int size) {
        totalAllocations.incrementAndGet();
        
        if (!enabled || size > MAX_BUCKET_SIZE) {
            poolMisses.incrementAndGet();
            return new double[size];
        }
        
        int bucketSize = getActualBucketSize(size);
        Queue<double[]> bucket = pools.get(bucketSize);
        
        if (bucket != null) {
            double[] array = bucket.poll();
            if (array != null) {
                poolHits.incrementAndGet();
                currentPooledBytes.addAndGet(-array.length * 8L);
                Arrays.fill(array, 0, size, 0.0); // Clear used portion
                return array;
            }
        }
        
        poolMisses.incrementAndGet();
        return new double[bucketSize];
    }
    
    /**
     * Acquire a double array and guarantee it's exactly the specified size.
     * 
     * @param size exact required size
     * @return a double array of exact size (new allocation if pooled is larger)
     */
    public double[] acquireExact(int size) {
        double[] array = acquire(size);
        if (array.length == size) {
            return array;
        }
        // Return a properly sized array if pool gave us a larger one
        release(array);
        return new double[size];
    }
    
    /**
     * Release a double array back to the pool for reuse.
     * 
     * @param array the array to release
     */
    public void release(double[] array) {
        if (!enabled || array == null || array.length > MAX_BUCKET_SIZE) {
            return;
        }
        
        // Check memory limits
        long currentBytes = currentPooledBytes.get();
        long arrayBytes = array.length * 8L;
        
        if (currentBytes + arrayBytes > config.getMaxPooledMemoryBytes()) {
            return; // Don't pool, let GC handle it
        }
        
        int bucketSize = getActualBucketSize(array.length);
        Queue<double[]> bucket = pools.get(bucketSize);
        
        if (bucket != null && bucket.size() < config.getMaxPooledArrays()) {
            bucket.offer(array);
            currentPooledBytes.addAndGet(arrayBytes);
        }
    }
    
    // ==================== 2D Double Array Operations ====================
    
    /**
     * Acquire a 2D double array.
     * 
     * @param rows number of rows
     * @param cols number of columns
     * @return a 2D double array
     */
    public double[][] acquire2D(int rows, int cols) {
        totalAllocations.incrementAndGet();
        
        int totalSize = rows * cols;
        if (!enabled || totalSize > MAX_BUCKET_SIZE) {
            poolMisses.incrementAndGet();
            return new double[rows][cols];
        }
        
        int bucketSize = getActualBucketSize(totalSize);
        Queue<double[][]> bucket = pools2D.get(bucketSize);
        
        if (bucket != null) {
            double[][] array = bucket.poll();
            if (array != null && array.length >= rows && array[0].length >= cols) {
                poolHits.incrementAndGet();
                currentPooledBytes.addAndGet(-array.length * array[0].length * 8L);
                // Clear the used portion
                for (int i = 0; i < rows; i++) {
                    Arrays.fill(array[i], 0, cols, 0.0);
                }
                return array;
            } else if (array != null) {
                // Wrong dimensions, put back and allocate new
                bucket.offer(array);
            }
        }
        
        poolMisses.incrementAndGet();
        return new double[rows][cols];
    }
    
    /**
     * Release a 2D double array back to the pool.
     * 
     * @param array the 2D array to release
     */
    public void release2D(double[][] array) {
        if (!enabled || array == null || array.length == 0) {
            return;
        }
        
        int totalSize = array.length * array[0].length;
        if (totalSize > MAX_BUCKET_SIZE) {
            return;
        }
        
        long arrayBytes = totalSize * 8L;
        long currentBytes = currentPooledBytes.get();
        
        if (currentBytes + arrayBytes > config.getMaxPooledMemoryBytes()) {
            return;
        }
        
        int bucketSize = getActualBucketSize(totalSize);
        Queue<double[][]> bucket = pools2D.get(bucketSize);
        
        if (bucket != null && bucket.size() < config.getMaxPooledArrays()) {
            bucket.offer(array);
            currentPooledBytes.addAndGet(arrayBytes);
        }
    }
    
    // ==================== Int Array Operations ====================
    
    /**
     * Acquire an int array of at least the specified size.
     * 
     * @param size minimum required size
     * @return an int array
     */
    public int[] acquireInt(int size) {
        totalAllocations.incrementAndGet();
        
        if (!enabled || size > MAX_BUCKET_SIZE) {
            poolMisses.incrementAndGet();
            return new int[size];
        }
        
        int bucketSize = getActualBucketSize(size);
        Queue<int[]> bucket = intPools.get(bucketSize);
        
        if (bucket != null) {
            int[] array = bucket.poll();
            if (array != null) {
                poolHits.incrementAndGet();
                currentPooledBytes.addAndGet(-array.length * 4L);
                Arrays.fill(array, 0, size, 0);
                return array;
            }
        }
        
        poolMisses.incrementAndGet();
        return new int[bucketSize];
    }
    
    /**
     * Release an int array back to the pool.
     * 
     * @param array the array to release
     */
    public void releaseInt(int[] array) {
        if (!enabled || array == null || array.length > MAX_BUCKET_SIZE) {
            return;
        }
        
        long arrayBytes = array.length * 4L;
        long currentBytes = currentPooledBytes.get();
        
        if (currentBytes + arrayBytes > config.getMaxPooledMemoryBytes()) {
            return;
        }
        
        int bucketSize = getActualBucketSize(array.length);
        Queue<int[]> bucket = intPools.get(bucketSize);
        
        if (bucket != null && bucket.size() < config.getMaxPooledArrays()) {
            bucket.offer(array);
            currentPooledBytes.addAndGet(arrayBytes);
        }
    }
    
    // ==================== Statistics ====================
    
    /**
     * Get total number of allocations requested.
     * 
     * @return total allocations
     */
    public long getTotalAllocations() {
        return totalAllocations.get();
    }
    
    /**
     * Get number of successful pool hits.
     * 
     * @return pool hits
     */
    public long getPoolHits() {
        return poolHits.get();
    }
    
    /**
     * Get number of pool misses (new allocations).
     * 
     * @return pool misses
     */
    public long getPoolMisses() {
        return poolMisses.get();
    }
    
    /**
     * Get the pool hit rate as a percentage.
     * 
     * @return hit rate (0.0 to 1.0)
     */
    public double getHitRate() {
        long total = totalAllocations.get();
        if (total == 0) {
            return 0.0;
        }
        return (double) poolHits.get() / total;
    }
    
    /**
     * Get current bytes stored in pool.
     * 
     * @return current pooled bytes
     */
    public long getCurrentPooledBytes() {
        return currentPooledBytes.get();
    }
    
    /**
     * Enable or disable memory pooling.
     * 
     * @param enabled true to enable pooling
     */
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
    
    /**
     * Check if memory pooling is enabled.
     * 
     * @return true if enabled
     */
    public boolean isEnabled() {
        return enabled;
    }
    
    /**
     * Get statistics as a formatted string.
     * 
     * @return statistics string
     */
    public String getStatistics() {
        return String.format(
            "MemoryPool{allocations=%d, hits=%d, misses=%d, hitRate=%.2f%%, pooledMB=%.2f}",
            totalAllocations.get(),
            poolHits.get(),
            poolMisses.get(),
            getHitRate() * 100,
            currentPooledBytes.get() / (1024.0 * 1024.0)
        );
    }
    
    @Override
    public String toString() {
        return getStatistics();
    }
}
