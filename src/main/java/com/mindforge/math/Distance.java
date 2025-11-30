package com.mindforge.math;

/**
 * Common distance metrics for machine learning algorithms.
 */
public class Distance {
    
    /**
     * Calculates the Euclidean distance between two points.
     * 
     * @param a first point
     * @param b second point
     * @return Euclidean distance
     */
    public static double euclidean(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Calculates the Manhattan distance between two points.
     * 
     * @param a first point
     * @param b second point
     * @return Manhattan distance
     */
    public static double manhattan(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.abs(a[i] - b[i]);
        }
        return sum;
    }
    
    /**
     * Calculates the Cosine similarity between two vectors.
     * Returns 1 - cosine similarity as distance.
     * 
     * @param a first vector
     * @param b second vector
     * @return Cosine distance (1 - similarity)
     */
    public static double cosine(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        
        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        if (normA == 0.0 || normB == 0.0) {
            return 1.0;
        }
        
        double similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
        return 1.0 - similarity;
    }
    
    /**
     * Calculates the Minkowski distance between two points.
     * 
     * @param a first point
     * @param b second point
     * @param p order of the Minkowski distance
     * @return Minkowski distance
     */
    public static double minkowski(double[] a, double[] b, double p) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        if (p <= 0) {
            throw new IllegalArgumentException("p must be positive");
        }
        
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(Math.abs(a[i] - b[i]), p);
        }
        return Math.pow(sum, 1.0 / p);
    }
}
