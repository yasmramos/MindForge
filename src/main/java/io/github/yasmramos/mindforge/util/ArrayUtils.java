package io.github.yasmramos.mindforge.util;

import java.util.Arrays;
import java.util.Random;

/**
 * Utility functions for array operations.
 */
public class ArrayUtils {
    
    private ArrayUtils() {
        // Utility class, prevent instantiation
    }
    
    /**
     * Calculate the mean of an array.
     * 
     * @param array input array
     * @return mean value
     */
    public static double mean(double[] array) {
        if (array.length == 0) return 0;
        double sum = 0;
        for (double v : array) {
            sum += v;
        }
        return sum / array.length;
    }
    
    /**
     * Calculate the standard deviation of an array.
     * 
     * @param array input array
     * @return standard deviation
     */
    public static double std(double[] array) {
        if (array.length == 0) return 0;
        double mean = mean(array);
        double sumSq = 0;
        for (double v : array) {
            sumSq += (v - mean) * (v - mean);
        }
        return Math.sqrt(sumSq / array.length);
    }
    
    /**
     * Calculate the variance of an array.
     * 
     * @param array input array
     * @return variance
     */
    public static double variance(double[] array) {
        double s = std(array);
        return s * s;
    }
    
    /**
     * Find the minimum value in an array.
     * 
     * @param array input array
     * @return minimum value
     */
    public static double min(double[] array) {
        if (array.length == 0) throw new IllegalArgumentException("Empty array");
        double min = array[0];
        for (double v : array) {
            if (v < min) min = v;
        }
        return min;
    }
    
    /**
     * Find the maximum value in an array.
     * 
     * @param array input array
     * @return maximum value
     */
    public static double max(double[] array) {
        if (array.length == 0) throw new IllegalArgumentException("Empty array");
        double max = array[0];
        for (double v : array) {
            if (v > max) max = v;
        }
        return max;
    }
    
    /**
     * Find the index of the maximum value.
     * 
     * @param array input array
     * @return index of maximum
     */
    public static int argmax(double[] array) {
        if (array.length == 0) throw new IllegalArgumentException("Empty array");
        int maxIdx = 0;
        double maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    /**
     * Find the index of the minimum value.
     * 
     * @param array input array
     * @return index of minimum
     */
    public static int argmin(double[] array) {
        if (array.length == 0) throw new IllegalArgumentException("Empty array");
        int minIdx = 0;
        double minVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] < minVal) {
                minVal = array[i];
                minIdx = i;
            }
        }
        return minIdx;
    }
    
    /**
     * Calculate the sum of an array.
     * 
     * @param array input array
     * @return sum
     */
    public static double sum(double[] array) {
        double sum = 0;
        for (double v : array) {
            sum += v;
        }
        return sum;
    }
    
    /**
     * Calculate the dot product of two arrays.
     * 
     * @param a first array
     * @param b second array
     * @return dot product
     */
    public static double dot(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    /**
     * Add two arrays element-wise.
     * 
     * @param a first array
     * @param b second array
     * @return result array
     */
    public static double[] add(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
    
    /**
     * Subtract two arrays element-wise.
     * 
     * @param a first array
     * @param b second array
     * @return result array
     */
    public static double[] subtract(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }
    
    /**
     * Multiply two arrays element-wise.
     * 
     * @param a first array
     * @param b second array
     * @return result array
     */
    public static double[] multiply(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }
    
    /**
     * Scale an array by a constant.
     * 
     * @param array input array
     * @param scalar scale factor
     * @return scaled array
     */
    public static double[] scale(double[] array, double scalar) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i] * scalar;
        }
        return result;
    }
    
    /**
     * Normalize an array to have zero mean and unit variance.
     * 
     * @param array input array
     * @return normalized array
     */
    public static double[] normalize(double[] array) {
        double mean = mean(array);
        double std = std(array);
        if (std == 0) std = 1;
        
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = (array[i] - mean) / std;
        }
        return result;
    }
    
    /**
     * Normalize an array to [0, 1] range.
     * 
     * @param array input array
     * @return normalized array
     */
    public static double[] minMaxNormalize(double[] array) {
        double min = min(array);
        double max = max(array);
        double range = max - min;
        if (range == 0) range = 1;
        
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = (array[i] - min) / range;
        }
        return result;
    }
    
    /**
     * Shuffle an array in place.
     * 
     * @param array array to shuffle
     * @param random random number generator
     */
    public static void shuffle(double[] array, Random random) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            double temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    
    /**
     * Shuffle an int array in place.
     * 
     * @param array array to shuffle
     * @param random random number generator
     */
    public static void shuffle(int[] array, Random random) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    
    /**
     * Create an array of evenly spaced values.
     * 
     * @param start start value
     * @param stop end value
     * @param num number of values
     * @return array of values
     */
    public static double[] linspace(double start, double stop, int num) {
        double[] result = new double[num];
        double step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            result[i] = start + i * step;
        }
        return result;
    }
    
    /**
     * Create an array of zeros.
     * 
     * @param size array size
     * @return array of zeros
     */
    public static double[] zeros(int size) {
        return new double[size];
    }
    
    /**
     * Create an array of ones.
     * 
     * @param size array size
     * @return array of ones
     */
    public static double[] ones(int size) {
        double[] result = new double[size];
        Arrays.fill(result, 1.0);
        return result;
    }
    
    /**
     * Create a 2D array of zeros.
     * 
     * @param rows number of rows
     * @param cols number of columns
     * @return 2D array of zeros
     */
    public static double[][] zeros2D(int rows, int cols) {
        return new double[rows][cols];
    }
    
    /**
     * Flatten a 2D array to 1D.
     * 
     * @param array 2D array
     * @return flattened 1D array
     */
    public static double[] flatten(double[][] array) {
        int total = 0;
        for (double[] row : array) {
            total += row.length;
        }
        
        double[] result = new double[total];
        int idx = 0;
        for (double[] row : array) {
            for (double v : row) {
                result[idx++] = v;
            }
        }
        return result;
    }
    
    /**
     * Reshape a 1D array to 2D.
     * 
     * @param array 1D array
     * @param rows number of rows
     * @param cols number of columns
     * @return 2D array
     */
    public static double[][] reshape(double[] array, int rows, int cols) {
        if (array.length != rows * cols) {
            throw new IllegalArgumentException("Array size does not match dimensions");
        }
        
        double[][] result = new double[rows][cols];
        int idx = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = array[idx++];
            }
        }
        return result;
    }
    
    /**
     * Transpose a 2D array.
     * 
     * @param array 2D array
     * @return transposed array
     */
    public static double[][] transpose(double[][] array) {
        if (array.length == 0) return new double[0][0];
        
        int rows = array.length;
        int cols = array[0].length;
        double[][] result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = array[i][j];
            }
        }
        return result;
    }
    
    /**
     * Concatenate two arrays.
     * 
     * @param a first array
     * @param b second array
     * @return concatenated array
     */
    public static double[] concatenate(double[] a, double[] b) {
        double[] result = new double[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }
    
    /**
     * Get a slice of an array.
     * 
     * @param array input array
     * @param start start index (inclusive)
     * @param end end index (exclusive)
     * @return sliced array
     */
    public static double[] slice(double[] array, int start, int end) {
        return Arrays.copyOfRange(array, start, end);
    }
    
    /**
     * Clone a 2D array.
     * 
     * @param array 2D array to clone
     * @return cloned array
     */
    public static double[][] clone2D(double[][] array) {
        double[][] result = new double[array.length][];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i].clone();
        }
        return result;
    }
    
    /**
     * Check if two arrays are approximately equal.
     * 
     * @param a first array
     * @param b second array
     * @param tolerance tolerance for comparison
     * @return true if approximately equal
     */
    public static boolean allClose(double[] a, double[] b, double tolerance) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > tolerance) return false;
        }
        return true;
    }
    
    /**
     * Count unique values in an array.
     * 
     * @param array input array
     * @return number of unique values
     */
    public static int countUnique(int[] array) {
        if (array.length == 0) return 0;
        int[] sorted = array.clone();
        Arrays.sort(sorted);
        int count = 1;
        for (int i = 1; i < sorted.length; i++) {
            if (sorted[i] != sorted[i - 1]) count++;
        }
        return count;
    }
    
    /**
     * Print a 2D array in a formatted way.
     * 
     * @param array 2D array to print
     */
    public static void print2D(double[][] array) {
        for (double[] row : array) {
            System.out.println(Arrays.toString(row));
        }
    }
}
