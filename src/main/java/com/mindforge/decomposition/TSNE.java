package com.mindforge.decomposition;

import java.util.Arrays;
import java.util.Random;

/**
 * t-Distributed Stochastic Neighbor Embedding (t-SNE) for visualization.
 * 
 * t-SNE is a nonlinear dimensionality reduction technique particularly
 * well-suited for visualizing high-dimensional data in 2D or 3D.
 * 
 * Key features:
 * - Preserves local structure of data
 * - Uses Student's t-distribution in low-dimensional space
 * - Adaptive learning rate with momentum
 * - Early exaggeration for better cluster separation
 * 
 * Example usage:
 * <pre>
 * TSNE tsne = new TSNE.Builder()
 *     .nComponents(2)
 *     .perplexity(30.0)
 *     .maxIter(1000)
 *     .build();
 * 
 * double[][] embedding = tsne.fitTransform(X);
 * </pre>
 * 
 * @author MindForge Team
 * @since 2.0.0
 */
public class TSNE {
    
    private final int nComponents;
    private final double perplexity;
    private final int maxIter;
    private final double learningRate;
    private final double earlyExaggeration;
    private final int earlyExaggerationIter;
    private final double minGradNorm;
    private final long randomSeed;
    
    private double[][] embedding;
    private boolean fitted = false;
    private int nSamples;
    private int nFeatures;
    private double klDivergence;
    
    /**
     * Private constructor - use Builder.
     */
    private TSNE(int nComponents, double perplexity, int maxIter,
                 double learningRate, double earlyExaggeration,
                 int earlyExaggerationIter, double minGradNorm, long randomSeed) {
        this.nComponents = nComponents;
        this.perplexity = perplexity;
        this.maxIter = maxIter;
        this.learningRate = learningRate;
        this.earlyExaggeration = earlyExaggeration;
        this.earlyExaggerationIter = earlyExaggerationIter;
        this.minGradNorm = minGradNorm;
        this.randomSeed = randomSeed;
    }
    
    /**
     * Fits the model and returns the embedding.
     * 
     * @param X Data matrix (n_samples x n_features)
     * @return Low-dimensional embedding (n_samples x n_components)
     */
    public double[][] fitTransform(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Data cannot be null or empty");
        }
        
        nSamples = X.length;
        nFeatures = X[0].length;
        
        if (nSamples < 4) {
            throw new IllegalArgumentException("Need at least 4 samples for t-SNE");
        }
        
        // Compute pairwise distances
        double[][] distances = computePairwiseDistances(X);
        
        // Compute joint probabilities P (symmetric)
        double[][] P = computeJointProbabilities(distances);
        
        // Apply early exaggeration
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nSamples; j++) {
                P[i][j] *= earlyExaggeration;
            }
        }
        
        // Initialize embedding randomly
        Random random = new Random(randomSeed);
        embedding = new double[nSamples][nComponents];
        for (int i = 0; i < nSamples; i++) {
            for (int d = 0; d < nComponents; d++) {
                embedding[i][d] = random.nextGaussian() * 1e-4;
            }
        }
        
        // Gradient descent with momentum
        double[][] velocity = new double[nSamples][nComponents];
        double[][] gains = new double[nSamples][nComponents];
        for (int i = 0; i < nSamples; i++) {
            Arrays.fill(gains[i], 1.0);
        }
        
        double momentum = 0.5;
        
        for (int iter = 0; iter < maxIter; iter++) {
            // Remove early exaggeration after specified iterations
            if (iter == earlyExaggerationIter) {
                for (int i = 0; i < nSamples; i++) {
                    for (int j = 0; j < nSamples; j++) {
                        P[i][j] /= earlyExaggeration;
                    }
                }
                momentum = 0.8;
            }
            
            // Compute Q (low-dimensional affinities)
            double[][] Q = computeLowDimAffinities(embedding);
            
            // Compute gradients
            double[][] gradient = computeGradient(P, Q, embedding);
            
            // Update with adaptive gains and momentum
            double gradNorm = 0.0;
            for (int i = 0; i < nSamples; i++) {
                for (int d = 0; d < nComponents; d++) {
                    // Adaptive gains
                    boolean sameSign = (gradient[i][d] * velocity[i][d]) >= 0;
                    gains[i][d] = sameSign ? gains[i][d] * 0.8 : gains[i][d] + 0.2;
                    gains[i][d] = Math.max(gains[i][d], 0.01);
                    
                    // Update velocity
                    velocity[i][d] = momentum * velocity[i][d] - 
                                     learningRate * gains[i][d] * gradient[i][d];
                    
                    // Update embedding
                    embedding[i][d] += velocity[i][d];
                    
                    gradNorm += gradient[i][d] * gradient[i][d];
                }
            }
            
            gradNorm = Math.sqrt(gradNorm);
            
            // Center embedding
            centerEmbedding();
            
            // Check convergence
            if (gradNorm < minGradNorm) {
                break;
            }
        }
        
        // Compute final KL divergence
        double[][] Q = computeLowDimAffinities(embedding);
        klDivergence = computeKLDivergence(P, Q);
        
        fitted = true;
        return copyEmbedding();
    }
    
    /**
     * Computes pairwise squared Euclidean distances.
     */
    private double[][] computePairwiseDistances(double[][] X) {
        double[][] D = new double[nSamples][nSamples];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = i + 1; j < nSamples; j++) {
                double dist = 0.0;
                for (int d = 0; d < nFeatures; d++) {
                    double diff = X[i][d] - X[j][d];
                    dist += diff * diff;
                }
                D[i][j] = dist;
                D[j][i] = dist;
            }
        }
        
        return D;
    }
    
    /**
     * Computes joint probabilities using binary search for perplexity.
     */
    private double[][] computeJointProbabilities(double[][] D) {
        double[][] P = new double[nSamples][nSamples];
        double targetEntropy = Math.log(perplexity);
        
        // Compute conditional probabilities for each point
        for (int i = 0; i < nSamples; i++) {
            // Binary search for sigma
            double sigmaMin = 1e-20;
            double sigmaMax = 1e20;
            double sigma = 1.0;
            
            for (int attempt = 0; attempt < 50; attempt++) {
                // Compute probabilities with current sigma
                double sumP = 0.0;
                for (int j = 0; j < nSamples; j++) {
                    if (i != j) {
                        P[i][j] = Math.exp(-D[i][j] / (2 * sigma * sigma));
                        sumP += P[i][j];
                    }
                }
                
                // Normalize
                sumP = Math.max(sumP, 1e-12);
                for (int j = 0; j < nSamples; j++) {
                    P[i][j] /= sumP;
                }
                
                // Compute entropy
                double entropy = 0.0;
                for (int j = 0; j < nSamples; j++) {
                    if (P[i][j] > 1e-12) {
                        entropy -= P[i][j] * Math.log(P[i][j]);
                    }
                }
                
                // Adjust sigma based on entropy
                double entropyDiff = entropy - targetEntropy;
                if (Math.abs(entropyDiff) < 1e-5) {
                    break;
                }
                
                if (entropyDiff > 0) {
                    sigmaMax = sigma;
                    sigma = (sigma + sigmaMin) / 2;
                } else {
                    sigmaMin = sigma;
                    sigma = (sigma + sigmaMax) / 2;
                }
            }
        }
        
        // Symmetrize and normalize
        double[][] Psym = new double[nSamples][nSamples];
        double sumP = 0.0;
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = i + 1; j < nSamples; j++) {
                Psym[i][j] = (P[i][j] + P[j][i]) / (2.0 * nSamples);
                Psym[j][i] = Psym[i][j];
                sumP += 2 * Psym[i][j];
            }
        }
        
        // Ensure minimum probability
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nSamples; j++) {
                Psym[i][j] = Math.max(Psym[i][j] / sumP, 1e-12);
            }
        }
        
        return Psym;
    }
    
    /**
     * Computes low-dimensional affinities using t-distribution.
     */
    private double[][] computeLowDimAffinities(double[][] Y) {
        double[][] Q = new double[nSamples][nSamples];
        double sumQ = 0.0;
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = i + 1; j < nSamples; j++) {
                double dist = 0.0;
                for (int d = 0; d < nComponents; d++) {
                    double diff = Y[i][d] - Y[j][d];
                    dist += diff * diff;
                }
                // Student's t-distribution with 1 degree of freedom
                double qij = 1.0 / (1.0 + dist);
                Q[i][j] = qij;
                Q[j][i] = qij;
                sumQ += 2 * qij;
            }
        }
        
        // Normalize
        sumQ = Math.max(sumQ, 1e-12);
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nSamples; j++) {
                Q[i][j] = Math.max(Q[i][j] / sumQ, 1e-12);
            }
        }
        
        return Q;
    }
    
    /**
     * Computes the gradient of the KL divergence.
     */
    private double[][] computeGradient(double[][] P, double[][] Q, double[][] Y) {
        double[][] gradient = new double[nSamples][nComponents];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nSamples; j++) {
                if (i != j) {
                    double dist = 0.0;
                    for (int d = 0; d < nComponents; d++) {
                        double diff = Y[i][d] - Y[j][d];
                        dist += diff * diff;
                    }
                    double qij = 1.0 / (1.0 + dist);
                    double mult = 4.0 * (P[i][j] - Q[i][j]) * qij;
                    
                    for (int d = 0; d < nComponents; d++) {
                        gradient[i][d] += mult * (Y[i][d] - Y[j][d]);
                    }
                }
            }
        }
        
        return gradient;
    }
    
    /**
     * Centers the embedding to have zero mean.
     */
    private void centerEmbedding() {
        double[] mean = new double[nComponents];
        
        for (int i = 0; i < nSamples; i++) {
            for (int d = 0; d < nComponents; d++) {
                mean[d] += embedding[i][d];
            }
        }
        
        for (int d = 0; d < nComponents; d++) {
            mean[d] /= nSamples;
        }
        
        for (int i = 0; i < nSamples; i++) {
            for (int d = 0; d < nComponents; d++) {
                embedding[i][d] -= mean[d];
            }
        }
    }
    
    /**
     * Computes KL divergence between P and Q.
     */
    private double computeKLDivergence(double[][] P, double[][] Q) {
        double kl = 0.0;
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nSamples; j++) {
                if (P[i][j] > 1e-12 && Q[i][j] > 1e-12) {
                    kl += P[i][j] * Math.log(P[i][j] / Q[i][j]);
                }
            }
        }
        return kl;
    }
    
    /**
     * Returns a copy of the embedding.
     */
    private double[][] copyEmbedding() {
        double[][] copy = new double[nSamples][];
        for (int i = 0; i < nSamples; i++) {
            copy[i] = Arrays.copyOf(embedding[i], nComponents);
        }
        return copy;
    }
    
    /**
     * Returns the embedding.
     */
    public double[][] getEmbedding() {
        if (!fitted) {
            throw new IllegalStateException("Model not fitted");
        }
        return copyEmbedding();
    }
    
    /**
     * Returns the KL divergence of the fitted model.
     */
    public double getKLDivergence() {
        return klDivergence;
    }
    
    /**
     * Returns whether the model is fitted.
     */
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Returns the number of components.
     */
    public int getNComponents() {
        return nComponents;
    }
    
    /**
     * Builder class for TSNE.
     */
    public static class Builder {
        private int nComponents = 2;
        private double perplexity = 30.0;
        private int maxIter = 1000;
        private double learningRate = 200.0;
        private double earlyExaggeration = 12.0;
        private int earlyExaggerationIter = 250;
        private double minGradNorm = 1e-7;
        private long randomSeed = 42;
        
        /**
         * Sets the number of components (dimensions) for embedding.
         */
        public Builder nComponents(int nComponents) {
            if (nComponents < 1) {
                throw new IllegalArgumentException("nComponents must be >= 1");
            }
            this.nComponents = nComponents;
            return this;
        }
        
        /**
         * Sets the perplexity (related to number of nearest neighbors).
         * Typical values: 5-50.
         */
        public Builder perplexity(double perplexity) {
            if (perplexity <= 0) {
                throw new IllegalArgumentException("perplexity must be positive");
            }
            this.perplexity = perplexity;
            return this;
        }
        
        /**
         * Sets the maximum number of iterations.
         */
        public Builder maxIter(int maxIter) {
            if (maxIter < 1) {
                throw new IllegalArgumentException("maxIter must be >= 1");
            }
            this.maxIter = maxIter;
            return this;
        }
        
        /**
         * Sets the learning rate.
         */
        public Builder learningRate(double learningRate) {
            if (learningRate <= 0) {
                throw new IllegalArgumentException("learningRate must be positive");
            }
            this.learningRate = learningRate;
            return this;
        }
        
        /**
         * Sets the early exaggeration factor.
         */
        public Builder earlyExaggeration(double earlyExaggeration) {
            if (earlyExaggeration < 1) {
                throw new IllegalArgumentException("earlyExaggeration must be >= 1");
            }
            this.earlyExaggeration = earlyExaggeration;
            return this;
        }
        
        /**
         * Sets the number of iterations for early exaggeration.
         */
        public Builder earlyExaggerationIter(int earlyExaggerationIter) {
            this.earlyExaggerationIter = earlyExaggerationIter;
            return this;
        }
        
        /**
         * Sets the minimum gradient norm for convergence.
         */
        public Builder minGradNorm(double minGradNorm) {
            this.minGradNorm = minGradNorm;
            return this;
        }
        
        /**
         * Sets the random seed.
         */
        public Builder randomSeed(long randomSeed) {
            this.randomSeed = randomSeed;
            return this;
        }
        
        /**
         * Builds the TSNE instance.
         */
        public TSNE build() {
            return new TSNE(nComponents, perplexity, maxIter, learningRate,
                           earlyExaggeration, earlyExaggerationIter, 
                           minGradNorm, randomSeed);
        }
    }
}
