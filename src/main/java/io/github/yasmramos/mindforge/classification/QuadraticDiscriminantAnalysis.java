package io.github.yasmramos.mindforge.classification;

import java.io.Serializable;
import java.util.*;

/**
 * Quadratic Discriminant Analysis (QDA).
 * 
 * A generative classifier that models each class with a Gaussian distribution
 * with its own covariance matrix (unlike LDA which assumes shared covariance).
 * 
 * @author MindForge
 */
public class QuadraticDiscriminantAnalysis implements Classifier<double[]>, ProbabilisticClassifier<double[]>, Serializable {
    private static final long serialVersionUID = 1L;
    
    private double regParam;
    private boolean storeCov;
    
    private int[] classes;
    private int numClasses;
    private int nFeatures;
    private double[] priors;
    private double[][] means;
    private double[][][] covariances;
    private double[][][] covarianceInverses;
    private double[] covarianceDets;
    private boolean trained;
    
    /**
     * Creates a QDA classifier with default parameters.
     */
    public QuadraticDiscriminantAnalysis() {
        this(0.0, false);
    }
    
    /**
     * Creates a QDA classifier with regularization.
     * 
     * @param regParam Regularization parameter (added to diagonal of covariance)
     */
    public QuadraticDiscriminantAnalysis(double regParam) {
        this(regParam, false);
    }
    
    /**
     * Creates a QDA classifier with full configuration.
     * 
     * @param regParam Regularization parameter
     * @param storeCov Whether to store covariance matrices
     */
    public QuadraticDiscriminantAnalysis(double regParam, boolean storeCov) {
        if (regParam < 0) {
            throw new IllegalArgumentException("regParam must be non-negative");
        }
        
        this.regParam = regParam;
        this.storeCov = storeCov;
        this.trained = false;
    }
    
    @Override
    public void train(double[][] X, int[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("X and y cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have same length");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("X cannot be empty");
        }
        
        int n = X.length;
        nFeatures = X[0].length;
        
        // Find unique classes
        Set<Integer> classSet = new TreeSet<>();
        for (int label : y) {
            classSet.add(label);
        }
        numClasses = classSet.size();
        classes = new int[numClasses];
        int idx = 0;
        for (int c : classSet) {
            classes[idx++] = c;
        }
        
        // Compute priors, means, and covariances for each class
        priors = new double[numClasses];
        means = new double[numClasses][nFeatures];
        covariances = new double[numClasses][nFeatures][nFeatures];
        covarianceInverses = new double[numClasses][nFeatures][nFeatures];
        covarianceDets = new double[numClasses];
        
        // Group samples by class
        List<List<Integer>> classIndices = new ArrayList<>();
        for (int c = 0; c < numClasses; c++) {
            classIndices.add(new ArrayList<>());
        }
        
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < numClasses; c++) {
                if (y[i] == classes[c]) {
                    classIndices.get(c).add(i);
                    break;
                }
            }
        }
        
        for (int c = 0; c < numClasses; c++) {
            List<Integer> indices = classIndices.get(c);
            int nC = indices.size();
            priors[c] = (double) nC / n;
            
            // Compute mean
            for (int idx2 : indices) {
                for (int j = 0; j < nFeatures; j++) {
                    means[c][j] += X[idx2][j];
                }
            }
            for (int j = 0; j < nFeatures; j++) {
                means[c][j] /= nC;
            }
            
            // Compute covariance
            for (int idx2 : indices) {
                for (int j = 0; j < nFeatures; j++) {
                    for (int k = 0; k < nFeatures; k++) {
                        covariances[c][j][k] += (X[idx2][j] - means[c][j]) * 
                                                 (X[idx2][k] - means[c][k]);
                    }
                }
            }
            
            for (int j = 0; j < nFeatures; j++) {
                for (int k = 0; k < nFeatures; k++) {
                    covariances[c][j][k] /= nC;
                }
                // Add regularization
                covariances[c][j][j] += regParam;
            }
            
            // Compute inverse and determinant
            covarianceInverses[c] = invertMatrix(covariances[c]);
            covarianceDets[c] = determinant(covariances[c]);
            
            // Ensure positive determinant
            if (covarianceDets[c] <= 0) {
                covarianceDets[c] = 1e-10;
            }
        }
        
        trained = true;
    }
    
    @Override
    public int predict(double[] x) {
        if (!trained) {
            throw new IllegalStateException("Model not trained");
        }
        if (x == null || x.length != nFeatures) {
            throw new IllegalArgumentException("Invalid input dimensions");
        }
        
        double[] proba = predictProba(x);
        int maxIdx = 0;
        for (int i = 1; i < numClasses; i++) {
            if (proba[i] > proba[maxIdx]) {
                maxIdx = i;
            }
        }
        return classes[maxIdx];
    }
    
    @Override
    public int[] predict(double[][] X) {
        if (X == null) {
            throw new IllegalArgumentException("X cannot be null");
        }
        
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    @Override
    public int getNumClasses() {
        return numClasses;
    }
    
    @Override
    public double[] predictProba(double[] x) {
        if (!trained) {
            throw new IllegalStateException("Model not trained");
        }
        
        double[] logProba = new double[numClasses];
        
        for (int c = 0; c < numClasses; c++) {
            // Log discriminant function
            double[] diff = new double[nFeatures];
            for (int j = 0; j < nFeatures; j++) {
                diff[j] = x[j] - means[c][j];
            }
            
            // Compute diff^T * Sigma^{-1} * diff
            double mahalanobis = 0;
            for (int j = 0; j < nFeatures; j++) {
                for (int k = 0; k < nFeatures; k++) {
                    mahalanobis += diff[j] * covarianceInverses[c][j][k] * diff[k];
                }
            }
            
            logProba[c] = Math.log(priors[c]) - 0.5 * Math.log(covarianceDets[c]) - 0.5 * mahalanobis;
        }
        
        // Convert to probabilities using softmax
        double maxLog = logProba[0];
        for (int c = 1; c < numClasses; c++) {
            maxLog = Math.max(maxLog, logProba[c]);
        }
        
        double[] proba = new double[numClasses];
        double sum = 0;
        for (int c = 0; c < numClasses; c++) {
            proba[c] = Math.exp(logProba[c] - maxLog);
            sum += proba[c];
        }
        
        for (int c = 0; c < numClasses; c++) {
            proba[c] /= sum;
        }
        
        return proba;
    }
    
    public double score(double[][] X, int[] y) {
        if (!trained) {
            throw new IllegalStateException("Model not trained");
        }
        
        int[] predictions = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) correct++;
        }
        return (double) correct / y.length;
    }
    
    /**
     * Computes matrix inverse using Gauss-Jordan elimination.
     */
    private double[][] invertMatrix(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];
        
        // Create augmented matrix [A | I]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = matrix[i][j];
            }
            augmented[i][n + i] = 1;
        }
        
        // Forward elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            double[] temp = augmented[i];
            augmented[i] = augmented[maxRow];
            augmented[maxRow] = temp;
            
            if (Math.abs(augmented[i][i]) < 1e-10) {
                augmented[i][i] = 1e-10;
            }
            
            // Scale row
            double scale = augmented[i][i];
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= scale;
            }
            
            // Eliminate column
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // Extract inverse
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = augmented[i][n + j];
            }
        }
        
        return inverse;
    }
    
    /**
     * Computes matrix determinant using LU decomposition.
     */
    private double determinant(double[][] matrix) {
        int n = matrix.length;
        double[][] lu = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, lu[i], 0, n);
        }
        
        double det = 1;
        
        for (int i = 0; i < n; i++) {
            // Partial pivoting
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(lu[k][i]) > Math.abs(lu[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            if (maxRow != i) {
                double[] temp = lu[i];
                lu[i] = lu[maxRow];
                lu[maxRow] = temp;
                det *= -1;
            }
            
            if (Math.abs(lu[i][i]) < 1e-10) {
                return 0;
            }
            
            det *= lu[i][i];
            
            for (int k = i + 1; k < n; k++) {
                double factor = lu[k][i] / lu[i][i];
                for (int j = i + 1; j < n; j++) {
                    lu[k][j] -= factor * lu[i][j];
                }
            }
        }
        
        return det;
    }
    
    public boolean isTrained() {
        return trained;
    }
    
    // Getters
    public int[] getClasses() {
        return classes != null ? classes.clone() : null;
    }
    
    public double[] getPriors() {
        return priors != null ? priors.clone() : null;
    }
    
    public double[][] getMeans() {
        if (means == null) return null;
        double[][] copy = new double[means.length][];
        for (int i = 0; i < means.length; i++) {
            copy[i] = means[i].clone();
        }
        return copy;
    }
    
    public double[][][] getCovariances() {
        if (!storeCov || covariances == null) return null;
        // Deep copy
        double[][][] copy = new double[covariances.length][][];
        for (int i = 0; i < covariances.length; i++) {
            copy[i] = new double[covariances[i].length][];
            for (int j = 0; j < covariances[i].length; j++) {
                copy[i][j] = covariances[i][j].clone();
            }
        }
        return copy;
    }
    
    public double getRegParam() {
        return regParam;
    }
}
