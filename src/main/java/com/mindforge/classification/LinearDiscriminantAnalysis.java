package com.mindforge.classification;

import java.util.*;

/**
 * Linear Discriminant Analysis (LDA) classifier and dimensionality reducer.
 * 
 * LDA finds a linear combination of features that characterizes or separates
 * two or more classes. It can be used for both classification and 
 * dimensionality reduction (supervised PCA alternative).
 * 
 * The algorithm maximizes the ratio of between-class variance to within-class
 * variance to achieve maximum class separability.
 * 
 * Key features:
 * - Multi-class classification
 * - Dimensionality reduction (project to n_classes - 1 dimensions)
 * - Prior probabilities (learned or specified)
 * - Regularization for numerical stability
 * 
 * Solvers:
 * - SVD: Singular Value Decomposition (default, recommended)
 * - EIGEN: Eigenvalue decomposition
 * 
 * Example usage:
 * <pre>
 * // Classification
 * LinearDiscriminantAnalysis lda = new LinearDiscriminantAnalysis.Builder()
 *     .solver(LinearDiscriminantAnalysis.Solver.SVD)
 *     .build();
 * 
 * lda.train(X_train, y_train);
 * int[] predictions = lda.predict(X_test);
 * 
 * // Dimensionality reduction
 * double[][] X_transformed = lda.transform(X_test);
 * 
 * // Combined
 * double[][] X_transformed = lda.fitTransform(X_train, y_train);
 * </pre>
 * 
 * @author MindForge Team
 * @since 2.0.0
 */
public class LinearDiscriminantAnalysis implements Classifier<double[]>, ProbabilisticClassifier<double[]> {
    
    /**
     * Solver type for LDA.
     */
    public enum Solver {
        /** Singular Value Decomposition (recommended for most cases) */
        SVD,
        /** Eigenvalue decomposition */
        EIGEN
    }
    
    // Configuration
    private final Solver solver;
    private final int nComponents;      // Number of components for dimensionality reduction
    private final double shrinkage;     // Shrinkage parameter for regularization
    private final boolean storeCov;     // Whether to store covariance matrices
    private final double[] priors;      // Prior probabilities (null = estimate from data)
    private final double tol;           // Tolerance for rank estimation
    
    // Learned parameters
    private int numFeatures;
    private int numClasses;
    private int[] classes;
    private double[][] classMeans;      // Mean vector for each class [class][feature]
    private double[] overallMean;       // Overall mean vector
    private double[][] scalings;        // Projection matrix [feature][component]
    private double[][] coef;            // Coefficient matrix for classification
    private double[] intercept;         // Intercept for each class
    private double[] classPriors;       // Learned or specified priors
    private double[][] withinClassCov;  // Within-class covariance (if storeCov=true)
    private double[][] betweenClassCov; // Between-class covariance (if storeCov=true)
    private double[] explainedVarianceRatio;
    
    private boolean fitted = false;
    
    /**
     * Private constructor - use Builder to create instances.
     */
    private LinearDiscriminantAnalysis(Solver solver, int nComponents, double shrinkage,
                                        boolean storeCov, double[] priors, double tol) {
        this.solver = solver;
        this.nComponents = nComponents;
        this.shrinkage = shrinkage;
        this.storeCov = storeCov;
        this.priors = priors;
        this.tol = tol;
    }
    
    @Override
    public void train(double[][] X, int[] y) {
        fit(X, y);
    }
    
    /**
     * Fits the LDA model.
     * 
     * @param X Training data of shape (n_samples, n_features)
     * @param y Target labels
     */
    public void fit(double[][] X, int[] y) {
        validateInput(X, y);
        
        numFeatures = X[0].length;
        
        // Find unique classes
        Set<Integer> classSet = new TreeSet<>();
        for (int label : y) {
            classSet.add(label);
        }
        numClasses = classSet.size();
        classes = new int[numClasses];
        int idx = 0;
        for (int cls : classSet) {
            classes[idx++] = cls;
        }
        
        // Compute class statistics
        computeClassStatistics(X, y);
        
        // Compute within-class and between-class scatter matrices
        double[][] Sw = computeWithinClassScatter(X, y);
        double[][] Sb = computeBetweenClassScatter();
        
        if (storeCov) {
            withinClassCov = Sw;
            betweenClassCov = Sb;
        }
        
        // Solve the generalized eigenvalue problem
        if (solver == Solver.SVD) {
            solveSVD(Sw, Sb);
        } else {
            solveEigen(Sw, Sb);
        }
        
        // Compute classification coefficients
        computeCoefficients();
        
        fitted = true;
    }
    
    /**
     * Validates input data.
     */
    private void validateInput(double[][] X, int[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Input data cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException(
                "X and y must have the same number of samples"
            );
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        for (int i = 0; i < X.length; i++) {
            if (X[i] == null) {
                throw new IllegalArgumentException("Sample at index " + i + " is null");
            }
            if (i > 0 && X[i].length != X[0].length) {
                throw new IllegalArgumentException(
                    "All samples must have the same number of features"
                );
            }
        }
    }
    
    /**
     * Computes class-wise statistics (means, priors).
     */
    private void computeClassStatistics(double[][] X, int[] y) {
        int n = X.length;
        
        // Initialize class means
        classMeans = new double[numClasses][numFeatures];
        int[] classCounts = new int[numClasses];
        
        // Map class labels to indices
        Map<Integer, Integer> classToIdx = new HashMap<>();
        for (int c = 0; c < numClasses; c++) {
            classToIdx.put(classes[c], c);
        }
        
        // Compute class sums
        for (int i = 0; i < n; i++) {
            int classIdx = classToIdx.get(y[i]);
            classCounts[classIdx]++;
            for (int f = 0; f < numFeatures; f++) {
                classMeans[classIdx][f] += X[i][f];
            }
        }
        
        // Compute class means
        for (int c = 0; c < numClasses; c++) {
            for (int f = 0; f < numFeatures; f++) {
                classMeans[c][f] /= classCounts[c];
            }
        }
        
        // Compute overall mean
        overallMean = new double[numFeatures];
        for (int i = 0; i < n; i++) {
            for (int f = 0; f < numFeatures; f++) {
                overallMean[f] += X[i][f];
            }
        }
        for (int f = 0; f < numFeatures; f++) {
            overallMean[f] /= n;
        }
        
        // Compute or use provided priors
        if (priors != null) {
            classPriors = Arrays.copyOf(priors, priors.length);
        } else {
            classPriors = new double[numClasses];
            for (int c = 0; c < numClasses; c++) {
                classPriors[c] = (double) classCounts[c] / n;
            }
        }
    }
    
    /**
     * Computes the within-class scatter matrix.
     */
    private double[][] computeWithinClassScatter(double[][] X, int[] y) {
        double[][] Sw = new double[numFeatures][numFeatures];
        
        Map<Integer, Integer> classToIdx = new HashMap<>();
        for (int c = 0; c < numClasses; c++) {
            classToIdx.put(classes[c], c);
        }
        
        // Compute scatter
        for (int i = 0; i < X.length; i++) {
            int classIdx = classToIdx.get(y[i]);
            double[] mean = classMeans[classIdx];
            
            for (int j = 0; j < numFeatures; j++) {
                double diffJ = X[i][j] - mean[j];
                for (int k = 0; k < numFeatures; k++) {
                    double diffK = X[i][k] - mean[k];
                    Sw[j][k] += diffJ * diffK;
                }
            }
        }
        
        // Regularization (shrinkage)
        if (shrinkage > 0) {
            double trace = 0.0;
            for (int i = 0; i < numFeatures; i++) {
                trace += Sw[i][i];
            }
            double mu = trace / numFeatures;
            
            for (int i = 0; i < numFeatures; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    if (i == j) {
                        Sw[i][j] = (1 - shrinkage) * Sw[i][j] + shrinkage * mu;
                    } else {
                        Sw[i][j] = (1 - shrinkage) * Sw[i][j];
                    }
                }
            }
        }
        
        return Sw;
    }
    
    /**
     * Computes the between-class scatter matrix.
     */
    private double[][] computeBetweenClassScatter() {
        double[][] Sb = new double[numFeatures][numFeatures];
        
        for (int c = 0; c < numClasses; c++) {
            double[] diff = new double[numFeatures];
            for (int f = 0; f < numFeatures; f++) {
                diff[f] = classMeans[c][f] - overallMean[f];
            }
            
            for (int j = 0; j < numFeatures; j++) {
                for (int k = 0; k < numFeatures; k++) {
                    Sb[j][k] += classPriors[c] * diff[j] * diff[k];
                }
            }
        }
        
        return Sb;
    }
    
    /**
     * Solves LDA using SVD approach.
     */
    private void solveSVD(double[][] Sw, double[][] Sb) {
        int effectiveComponents = Math.min(nComponents > 0 ? nComponents : numClasses - 1,
                                           Math.min(numFeatures, numClasses - 1));
        
        // Compute Sw^(-1/2)
        double[][] SwInvSqrt = computeInverseSquareRoot(Sw);
        
        // Compute Sw^(-1/2) * Sb * Sw^(-1/2)
        double[][] temp = matrixMultiply(SwInvSqrt, Sb);
        double[][] M = matrixMultiply(temp, SwInvSqrt);
        
        // Eigendecomposition
        double[][] eigenvectors = new double[numFeatures][numFeatures];
        double[] eigenvalues = new double[numFeatures];
        computeEigendecomposition(M, eigenvectors, eigenvalues);
        
        // Sort eigenvalues and eigenvectors in descending order
        Integer[] indices = new Integer[numFeatures];
        for (int i = 0; i < numFeatures; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(eigenvalues[b], eigenvalues[a]));
        
        // Compute scalings (transformation matrix)
        scalings = new double[numFeatures][effectiveComponents];
        explainedVarianceRatio = new double[effectiveComponents];
        
        double totalVariance = 0.0;
        for (double ev : eigenvalues) {
            if (ev > 0) totalVariance += ev;
        }
        
        for (int j = 0; j < effectiveComponents; j++) {
            int sortedIdx = indices[j];
            
            // Transform eigenvector: Sw^(-1/2) * v
            for (int i = 0; i < numFeatures; i++) {
                for (int k = 0; k < numFeatures; k++) {
                    scalings[i][j] += SwInvSqrt[i][k] * eigenvectors[k][sortedIdx];
                }
            }
            
            if (totalVariance > 0) {
                explainedVarianceRatio[j] = eigenvalues[sortedIdx] / totalVariance;
            }
        }
    }
    
    /**
     * Solves LDA using eigenvalue decomposition approach.
     */
    private void solveEigen(double[][] Sw, double[][] Sb) {
        int effectiveComponents = Math.min(nComponents > 0 ? nComponents : numClasses - 1,
                                           Math.min(numFeatures, numClasses - 1));
        
        // Compute Sw^(-1) * Sb
        double[][] SwInv = computeInverse(Sw);
        double[][] M = matrixMultiply(SwInv, Sb);
        
        // Eigendecomposition
        double[][] eigenvectors = new double[numFeatures][numFeatures];
        double[] eigenvalues = new double[numFeatures];
        computeEigendecomposition(M, eigenvectors, eigenvalues);
        
        // Sort eigenvalues and eigenvectors in descending order
        Integer[] indices = new Integer[numFeatures];
        for (int i = 0; i < numFeatures; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(eigenvalues[b], eigenvalues[a]));
        
        // Extract top components
        scalings = new double[numFeatures][effectiveComponents];
        explainedVarianceRatio = new double[effectiveComponents];
        
        double totalVariance = 0.0;
        for (double ev : eigenvalues) {
            if (ev > 0) totalVariance += ev;
        }
        
        for (int j = 0; j < effectiveComponents; j++) {
            int sortedIdx = indices[j];
            for (int i = 0; i < numFeatures; i++) {
                scalings[i][j] = eigenvectors[i][sortedIdx];
            }
            if (totalVariance > 0) {
                explainedVarianceRatio[j] = eigenvalues[sortedIdx] / totalVariance;
            }
        }
    }
    
    /**
     * Computes matrix inverse using Gauss-Jordan elimination.
     */
    private double[][] computeInverse(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];
        
        // Create augmented matrix [A | I]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = matrix[i][j];
            }
            augmented[i][n + i] = 1.0;
        }
        
        // Gauss-Jordan elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            // Swap rows
            double[] temp = augmented[i];
            augmented[i] = augmented[maxRow];
            augmented[maxRow] = temp;
            
            // Check for singular matrix
            if (Math.abs(augmented[i][i]) < 1e-10) {
                // Add small regularization
                augmented[i][i] = 1e-10;
            }
            
            // Scale pivot row
            double pivot = augmented[i][i];
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
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
     * Computes the inverse square root of a positive semi-definite matrix.
     */
    private double[][] computeInverseSquareRoot(double[][] matrix) {
        int n = matrix.length;
        
        // Eigendecomposition
        double[][] V = new double[n][n];
        double[] D = new double[n];
        computeEigendecomposition(matrix, V, D);
        
        // Compute inverse square root of eigenvalues
        double[] DInvSqrt = new double[n];
        for (int i = 0; i < n; i++) {
            if (D[i] > tol) {
                DInvSqrt[i] = 1.0 / Math.sqrt(D[i]);
            } else {
                DInvSqrt[i] = 0.0;
            }
        }
        
        // Reconstruct: V * D^(-1/2) * V^T
        double[][] result = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    result[i][j] += V[i][k] * DInvSqrt[k] * V[j][k];
                }
            }
        }
        
        return result;
    }
    
    /**
     * Computes eigendecomposition using power iteration.
     */
    private void computeEigendecomposition(double[][] matrix, double[][] eigenvectors, double[] eigenvalues) {
        int n = matrix.length;
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) {
            A[i] = Arrays.copyOf(matrix[i], n);
        }
        
        Random random = new Random(42);
        
        for (int k = 0; k < n; k++) {
            // Initialize random vector
            double[] v = new double[n];
            for (int i = 0; i < n; i++) {
                v[i] = random.nextGaussian();
            }
            normalize(v);
            
            // Power iteration
            for (int iter = 0; iter < 100; iter++) {
                double[] Av = new double[n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        Av[i] += A[i][j] * v[j];
                    }
                }
                
                // Deflate previous eigenvectors
                for (int prev = 0; prev < k; prev++) {
                    double dot = 0.0;
                    for (int i = 0; i < n; i++) {
                        dot += Av[i] * eigenvectors[i][prev];
                    }
                    for (int i = 0; i < n; i++) {
                        Av[i] -= dot * eigenvectors[i][prev];
                    }
                }
                
                double norm = normalize(Av);
                
                // Check convergence
                double diff = 0.0;
                for (int i = 0; i < n; i++) {
                    diff += Math.abs(Av[i] - v[i]);
                }
                
                v = Av;
                
                if (diff < tol) break;
            }
            
            // Compute eigenvalue
            double[] Av = new double[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    Av[i] += A[i][j] * v[j];
                }
            }
            
            double eigenvalue = 0.0;
            for (int i = 0; i < n; i++) {
                eigenvalue += v[i] * Av[i];
            }
            
            eigenvalues[k] = eigenvalue;
            for (int i = 0; i < n; i++) {
                eigenvectors[i][k] = v[i];
            }
        }
    }
    
    /**
     * Normalizes a vector in place and returns the original norm.
     */
    private double normalize(double[] v) {
        double norm = 0.0;
        for (double val : v) {
            norm += val * val;
        }
        norm = Math.sqrt(norm);
        
        if (norm > 0) {
            for (int i = 0; i < v.length; i++) {
                v[i] /= norm;
            }
        }
        return norm;
    }
    
    /**
     * Multiplies two matrices.
     */
    private double[][] matrixMultiply(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        int p = B[0].length;
        
        double[][] C = new double[m][p];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }
    
    /**
     * Computes coefficients for classification.
     */
    private void computeCoefficients() {
        // For each class, compute the projection of class mean
        coef = new double[numClasses][scalings[0].length];
        intercept = new double[numClasses];
        
        for (int c = 0; c < numClasses; c++) {
            // Project class mean onto discriminant space
            for (int j = 0; j < scalings[0].length; j++) {
                for (int f = 0; f < numFeatures; f++) {
                    coef[c][j] += classMeans[c][f] * scalings[f][j];
                }
            }
            
            // Compute intercept (log prior - 0.5 * mean^T * cov^-1 * mean)
            intercept[c] = Math.log(classPriors[c]);
            for (int j = 0; j < coef[c].length; j++) {
                intercept[c] -= 0.5 * coef[c][j] * coef[c][j];
            }
        }
    }
    
    @Override
    public int predict(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        if (x == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, x.length)
            );
        }
        
        // Project sample
        double[] projected = new double[scalings[0].length];
        for (int j = 0; j < scalings[0].length; j++) {
            for (int f = 0; f < numFeatures; f++) {
                projected[j] += x[f] * scalings[f][j];
            }
        }
        
        // Compute decision function for each class
        double maxScore = Double.NEGATIVE_INFINITY;
        int predictedClass = classes[0];
        
        for (int c = 0; c < numClasses; c++) {
            double score = intercept[c];
            for (int j = 0; j < projected.length; j++) {
                score += projected[j] * coef[c][j];
            }
            
            if (score > maxScore) {
                maxScore = score;
                predictedClass = classes[c];
            }
        }
        
        return predictedClass;
    }
    
    @Override
    public int[] predict(double[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    @Override
    public double[] predictProbabilities(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        
        // Project sample
        double[] projected = new double[scalings[0].length];
        for (int j = 0; j < scalings[0].length; j++) {
            for (int f = 0; f < numFeatures; f++) {
                projected[j] += x[f] * scalings[f][j];
            }
        }
        
        // Compute log-likelihood for each class
        double[] logLikelihood = new double[numClasses];
        double maxLog = Double.NEGATIVE_INFINITY;
        
        for (int c = 0; c < numClasses; c++) {
            logLikelihood[c] = intercept[c];
            for (int j = 0; j < projected.length; j++) {
                logLikelihood[c] += projected[j] * coef[c][j];
            }
            maxLog = Math.max(maxLog, logLikelihood[c]);
        }
        
        // Softmax normalization
        double[] probabilities = new double[numClasses];
        double sum = 0.0;
        for (int c = 0; c < numClasses; c++) {
            probabilities[c] = Math.exp(logLikelihood[c] - maxLog);
            sum += probabilities[c];
        }
        
        for (int c = 0; c < numClasses; c++) {
            probabilities[c] /= sum;
        }
        
        return probabilities;
    }
    
    /**
     * Transforms the data to the discriminant space.
     * 
     * @param X Data to transform
     * @return Transformed data
     */
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before transform");
        }
        if (X == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        
        int nComponents = scalings[0].length;
        double[][] result = new double[X.length][nComponents];
        
        for (int i = 0; i < X.length; i++) {
            if (X[i].length != numFeatures) {
                throw new IllegalArgumentException(
                    "Sample " + i + " has " + X[i].length + 
                    " features, expected " + numFeatures
                );
            }
            
            for (int j = 0; j < nComponents; j++) {
                for (int f = 0; f < numFeatures; f++) {
                    result[i][j] += X[i][f] * scalings[f][j];
                }
            }
        }
        
        return result;
    }
    
    /**
     * Fits the model and transforms the data.
     * 
     * @param X Training data
     * @param y Target labels
     * @return Transformed data
     */
    public double[][] fitTransform(double[][] X, int[] y) {
        fit(X, y);
        return transform(X);
    }
    
    /**
     * Computes the accuracy on the given data.
     * 
     * @param X Test features
     * @param y True labels
     * @return Accuracy score
     */
    public double score(double[][] X, int[] y) {
        int[] predictions = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) {
                correct++;
            }
        }
        return (double) correct / y.length;
    }
    
    @Override
    public int getNumClasses() {
        return numClasses;
    }
    
    /**
     * Returns the class labels.
     * 
     * @return Class labels
     */
    public int[] getClasses() {
        return Arrays.copyOf(classes, classes.length);
    }
    
    /**
     * Returns the class means.
     * 
     * @return Class means [class][feature]
     */
    public double[][] getClassMeans() {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted");
        }
        double[][] copy = new double[numClasses][];
        for (int c = 0; c < numClasses; c++) {
            copy[c] = Arrays.copyOf(classMeans[c], numFeatures);
        }
        return copy;
    }
    
    /**
     * Returns the scaling/projection matrix.
     * 
     * @return Scaling matrix [feature][component]
     */
    public double[][] getScalings() {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted");
        }
        double[][] copy = new double[numFeatures][];
        for (int f = 0; f < numFeatures; f++) {
            copy[f] = Arrays.copyOf(scalings[f], scalings[f].length);
        }
        return copy;
    }
    
    /**
     * Returns the explained variance ratio.
     * 
     * @return Explained variance ratio for each component
     */
    public double[] getExplainedVarianceRatio() {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted");
        }
        return Arrays.copyOf(explainedVarianceRatio, explainedVarianceRatio.length);
    }
    
    /**
     * Returns the class priors.
     * 
     * @return Class priors
     */
    public double[] getPriors() {
        if (!fitted) {
            throw new IllegalStateException("Model has not been fitted");
        }
        return Arrays.copyOf(classPriors, classPriors.length);
    }
    
    /**
     * Returns whether the model has been fitted.
     * 
     * @return true if fitted
     */
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Builder class for LinearDiscriminantAnalysis.
     */
    public static class Builder {
        private Solver solver = Solver.SVD;
        private int nComponents = -1; // -1 means min(n_classes - 1, n_features)
        private double shrinkage = 0.0;
        private boolean storeCov = false;
        private double[] priors = null;
        private double tol = 1e-4;
        
        /**
         * Sets the solver type.
         * 
         * @param solver The solver (SVD or EIGEN)
         * @return This builder
         */
        public Builder solver(Solver solver) {
            if (solver == null) {
                throw new IllegalArgumentException("solver cannot be null");
            }
            this.solver = solver;
            return this;
        }
        
        /**
         * Sets the number of components for dimensionality reduction.
         * 
         * @param nComponents Number of components (must be positive)
         * @return This builder
         */
        public Builder nComponents(int nComponents) {
            if (nComponents <= 0) {
                throw new IllegalArgumentException("nComponents must be positive");
            }
            this.nComponents = nComponents;
            return this;
        }
        
        /**
         * Sets the shrinkage parameter for regularization.
         * 
         * @param shrinkage Shrinkage parameter in [0, 1]
         * @return This builder
         */
        public Builder shrinkage(double shrinkage) {
            if (shrinkage < 0 || shrinkage > 1) {
                throw new IllegalArgumentException("shrinkage must be in [0, 1]");
            }
            this.shrinkage = shrinkage;
            return this;
        }
        
        /**
         * Sets whether to store covariance matrices.
         * 
         * @param storeCov Whether to store covariance
         * @return This builder
         */
        public Builder storeCov(boolean storeCov) {
            this.storeCov = storeCov;
            return this;
        }
        
        /**
         * Sets the class priors.
         * 
         * @param priors Prior probabilities (must sum to 1)
         * @return This builder
         */
        public Builder priors(double[] priors) {
            if (priors != null) {
                double sum = 0.0;
                for (double p : priors) {
                    if (p < 0) {
                        throw new IllegalArgumentException("priors must be non-negative");
                    }
                    sum += p;
                }
                if (Math.abs(sum - 1.0) > 1e-6) {
                    throw new IllegalArgumentException("priors must sum to 1");
                }
            }
            this.priors = priors != null ? Arrays.copyOf(priors, priors.length) : null;
            return this;
        }
        
        /**
         * Sets the tolerance for numerical stability.
         * 
         * @param tol Tolerance (must be positive)
         * @return This builder
         */
        public Builder tol(double tol) {
            if (tol <= 0) {
                throw new IllegalArgumentException("tol must be positive");
            }
            this.tol = tol;
            return this;
        }
        
        /**
         * Builds the LinearDiscriminantAnalysis instance.
         * 
         * @return A new LinearDiscriminantAnalysis instance
         */
        public LinearDiscriminantAnalysis build() {
            return new LinearDiscriminantAnalysis(solver, nComponents, shrinkage, 
                                                   storeCov, priors, tol);
        }
    }
}
