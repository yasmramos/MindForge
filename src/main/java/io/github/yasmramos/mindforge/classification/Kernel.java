package io.github.yasmramos.mindforge.classification;

/**
 * Kernel functions for Support Vector Machines.
 * 
 * Kernels enable SVMs to find non-linear decision boundaries by implicitly
 * mapping data to higher-dimensional feature spaces without explicitly
 * computing the transformation (the "kernel trick").
 * 
 * Supported kernels:
 * - LINEAR: K(x, y) = x · y
 * - RBF (Radial Basis Function): K(x, y) = exp(-gamma * ||x - y||²)
 * - POLYNOMIAL: K(x, y) = (gamma * x · y + coef0)^degree
 * - SIGMOID: K(x, y) = tanh(gamma * x · y + coef0)
 * 
 * Example usage:
 * <pre>
 * // RBF kernel with gamma=0.5
 * Kernel rbf = new Kernel(Kernel.Type.RBF, 0.5);
 * double similarity = rbf.compute(x1, x2);
 * 
 * // Polynomial kernel with degree=3
 * Kernel poly = new Kernel(Kernel.Type.POLYNOMIAL, 1.0, 0.0, 3);
 * double similarity = poly.compute(x1, x2);
 * </pre>
 * 
 * @author MindForge Team
 */
public class Kernel {
    
    /**
     * Types of kernel functions available.
     */
    public enum Type {
        /** Linear kernel: K(x, y) = x · y */
        LINEAR,
        
        /** Radial Basis Function: K(x, y) = exp(-gamma * ||x - y||²) */
        RBF,
        
        /** Polynomial: K(x, y) = (gamma * x · y + coef0)^degree */
        POLYNOMIAL,
        
        /** Sigmoid: K(x, y) = tanh(gamma * x · y + coef0) */
        SIGMOID
    }
    
    private final Type type;
    private final double gamma;
    private final double coef0;
    private final int degree;
    
    /**
     * Creates a linear kernel.
     */
    public Kernel() {
        this(Type.LINEAR, 1.0, 0.0, 3);
    }
    
    /**
     * Creates a kernel of the specified type with default parameters.
     * 
     * @param type The kernel type
     */
    public Kernel(Type type) {
        this(type, 1.0, 0.0, 3);
    }
    
    /**
     * Creates a kernel with specified gamma (for RBF, POLYNOMIAL, SIGMOID).
     * 
     * @param type The kernel type
     * @param gamma The gamma parameter (default: 1.0)
     */
    public Kernel(Type type, double gamma) {
        this(type, gamma, 0.0, 3);
    }
    
    /**
     * Creates a kernel with full parameter specification.
     * 
     * @param type The kernel type
     * @param gamma Kernel coefficient for RBF, POLYNOMIAL, SIGMOID
     * @param coef0 Independent term in POLYNOMIAL and SIGMOID kernels
     * @param degree Degree of the polynomial kernel
     */
    public Kernel(Type type, double gamma, double coef0, int degree) {
        if (gamma <= 0 && type != Type.LINEAR) {
            throw new IllegalArgumentException("gamma must be positive for non-linear kernels");
        }
        if (degree < 1 && type == Type.POLYNOMIAL) {
            throw new IllegalArgumentException("degree must be at least 1 for polynomial kernel");
        }
        
        this.type = type;
        this.gamma = gamma;
        this.coef0 = coef0;
        this.degree = degree;
    }
    
    /**
     * Computes the kernel value between two vectors.
     * 
     * @param x First vector
     * @param y Second vector
     * @return The kernel value K(x, y)
     */
    public double compute(double[] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException(
                String.format("Vector dimensions don't match: %d vs %d", x.length, y.length)
            );
        }
        
        switch (type) {
            case LINEAR:
                return linearKernel(x, y);
            case RBF:
                return rbfKernel(x, y);
            case POLYNOMIAL:
                return polynomialKernel(x, y);
            case SIGMOID:
                return sigmoidKernel(x, y);
            default:
                throw new IllegalStateException("Unknown kernel type: " + type);
        }
    }
    
    /**
     * Computes the kernel matrix (Gram matrix) for a dataset.
     * 
     * @param X The data matrix (n_samples x n_features)
     * @return The kernel matrix K where K[i][j] = K(X[i], X[j])
     */
    public double[][] computeMatrix(double[][] X) {
        int n = X.length;
        double[][] K = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double value = compute(X[i], X[j]);
                K[i][j] = value;
                K[j][i] = value; // Kernel matrices are symmetric
            }
        }
        
        return K;
    }
    
    /**
     * Computes the kernel values between a single sample and a dataset.
     * 
     * @param x The single sample
     * @param X The dataset (n_samples x n_features)
     * @return Array of kernel values K(x, X[i]) for all i
     */
    public double[] computeRow(double[] x, double[][] X) {
        double[] row = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            row[i] = compute(x, X[i]);
        }
        return row;
    }
    
    /**
     * Linear kernel: K(x, y) = x · y
     */
    private double linearKernel(double[] x, double[] y) {
        double dot = 0.0;
        for (int i = 0; i < x.length; i++) {
            dot += x[i] * y[i];
        }
        return dot;
    }
    
    /**
     * RBF (Gaussian) kernel: K(x, y) = exp(-gamma * ||x - y||²)
     */
    private double rbfKernel(double[] x, double[] y) {
        double squaredDistance = 0.0;
        for (int i = 0; i < x.length; i++) {
            double diff = x[i] - y[i];
            squaredDistance += diff * diff;
        }
        return Math.exp(-gamma * squaredDistance);
    }
    
    /**
     * Polynomial kernel: K(x, y) = (gamma * x · y + coef0)^degree
     */
    private double polynomialKernel(double[] x, double[] y) {
        double dot = 0.0;
        for (int i = 0; i < x.length; i++) {
            dot += x[i] * y[i];
        }
        return Math.pow(gamma * dot + coef0, degree);
    }
    
    /**
     * Sigmoid kernel: K(x, y) = tanh(gamma * x · y + coef0)
     */
    private double sigmoidKernel(double[] x, double[] y) {
        double dot = 0.0;
        for (int i = 0; i < x.length; i++) {
            dot += x[i] * y[i];
        }
        return Math.tanh(gamma * dot + coef0);
    }
    
    // Getters
    
    /**
     * Returns the kernel type.
     */
    public Type getType() {
        return type;
    }
    
    /**
     * Returns the gamma parameter.
     */
    public double getGamma() {
        return gamma;
    }
    
    /**
     * Returns the coef0 parameter.
     */
    public double getCoef0() {
        return coef0;
    }
    
    /**
     * Returns the polynomial degree.
     */
    public int getDegree() {
        return degree;
    }
    
    /**
     * Computes the default gamma value based on number of features.
     * Default is 1 / n_features (scale gamma).
     * 
     * @param nFeatures Number of features
     * @return The default gamma value
     */
    public static double defaultGamma(int nFeatures) {
        return 1.0 / nFeatures;
    }
    
    /**
     * Creates an RBF kernel with automatic gamma based on number of features.
     * 
     * @param nFeatures Number of features in the dataset
     * @return An RBF kernel with gamma = 1/n_features
     */
    public static Kernel rbf(int nFeatures) {
        return new Kernel(Type.RBF, defaultGamma(nFeatures));
    }
    
    /**
     * Creates an RBF kernel with specified gamma.
     * 
     * @param gamma The gamma parameter
     * @return An RBF kernel
     */
    public static Kernel rbf(double gamma) {
        return new Kernel(Type.RBF, gamma);
    }
    
    /**
     * Creates a polynomial kernel with specified degree.
     * 
     * @param degree The polynomial degree
     * @return A polynomial kernel with gamma=1, coef0=0
     */
    public static Kernel polynomial(int degree) {
        return new Kernel(Type.POLYNOMIAL, 1.0, 0.0, degree);
    }
    
    /**
     * Creates a polynomial kernel with full parameters.
     * 
     * @param degree The polynomial degree
     * @param gamma The gamma parameter
     * @param coef0 The independent term
     * @return A polynomial kernel
     */
    public static Kernel polynomial(int degree, double gamma, double coef0) {
        return new Kernel(Type.POLYNOMIAL, gamma, coef0, degree);
    }
    
    /**
     * Creates a linear kernel.
     * 
     * @return A linear kernel
     */
    public static Kernel linear() {
        return new Kernel(Type.LINEAR);
    }
    
    /**
     * Creates a sigmoid kernel with specified parameters.
     * 
     * @param gamma The gamma parameter
     * @param coef0 The independent term
     * @return A sigmoid kernel
     */
    public static Kernel sigmoid(double gamma, double coef0) {
        return new Kernel(Type.SIGMOID, gamma, coef0, 1);
    }
    
    @Override
    public String toString() {
        switch (type) {
            case LINEAR:
                return "Kernel(type=LINEAR)";
            case RBF:
                return String.format("Kernel(type=RBF, gamma=%.4f)", gamma);
            case POLYNOMIAL:
                return String.format("Kernel(type=POLYNOMIAL, degree=%d, gamma=%.4f, coef0=%.4f)", 
                    degree, gamma, coef0);
            case SIGMOID:
                return String.format("Kernel(type=SIGMOID, gamma=%.4f, coef0=%.4f)", gamma, coef0);
            default:
                return "Kernel(type=UNKNOWN)";
        }
    }
}
