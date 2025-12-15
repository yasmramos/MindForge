package io.github.yasmramos.mindforge.interpret;

import io.github.yasmramos.mindforge.classification.Classifier;
import io.github.yasmramos.mindforge.regression.Regressor;
import io.github.yasmramos.mindforge.validation.Metrics;

import java.io.Serializable;
import java.util.Random;

/**
 * Feature importance calculation using permutation importance.
 * Works with any model type.
 */
public class FeatureImportance implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int nRepeats;
    private final long randomSeed;
    
    private double[] importances;
    private double[] importancesStd;
    private String[] featureNames;
    
    public FeatureImportance(int nRepeats, long randomSeed) {
        this.nRepeats = nRepeats;
        this.randomSeed = randomSeed;
    }
    
    public FeatureImportance() {
        this(10, 42);
    }
    
    /**
     * Compute permutation importance for a classifier.
     */
    public void compute(Classifier model, double[][] X, int[] y, String[] featureNames) {
        this.featureNames = featureNames;
        int nFeatures = X[0].length;
        importances = new double[nFeatures];
        importancesStd = new double[nFeatures];
        
        Random random = new Random(randomSeed);
        
        // Baseline score
        int[] basePred = model.predict(X);
        double baseScore = Metrics.accuracy(y, basePred);
        
        for (int f = 0; f < nFeatures; f++) {
            double[] scores = new double[nRepeats];
            
            for (int r = 0; r < nRepeats; r++) {
                double[][] XPermuted = copyMatrix(X);
                permuteColumn(XPermuted, f, random);
                
                int[] pred = model.predict(XPermuted);
                scores[r] = baseScore - Metrics.accuracy(y, pred);
            }
            
            importances[f] = mean(scores);
            importancesStd[f] = std(scores);
        }
    }
    
    /**
     * Compute permutation importance for a regressor.
     */
    public void compute(Regressor model, double[][] X, double[] y, String[] featureNames) {
        this.featureNames = featureNames;
        int nFeatures = X[0].length;
        importances = new double[nFeatures];
        importancesStd = new double[nFeatures];
        
        Random random = new Random(randomSeed);
        
        // Baseline score (R2)
        double[] basePred = model.predict(X);
        double baseScore = Metrics.r2Score(y, basePred);
        
        for (int f = 0; f < nFeatures; f++) {
            double[] scores = new double[nRepeats];
            
            for (int r = 0; r < nRepeats; r++) {
                double[][] XPermuted = copyMatrix(X);
                permuteColumn(XPermuted, f, random);
                
                double[] pred = model.predict(XPermuted);
                scores[r] = baseScore - Metrics.r2Score(y, pred);
            }
            
            importances[f] = mean(scores);
            importancesStd[f] = std(scores);
        }
    }
    
    private double[][] copyMatrix(double[][] X) {
        double[][] copy = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            copy[i] = X[i].clone();
        }
        return copy;
    }
    
    private void permuteColumn(double[][] X, int col, Random random) {
        int n = X.length;
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            double temp = X[i][col];
            X[i][col] = X[j][col];
            X[j][col] = temp;
        }
    }
    
    private double mean(double[] values) {
        double sum = 0;
        for (double v : values) sum += v;
        return sum / values.length;
    }
    
    private double std(double[] values) {
        double m = mean(values);
        double sum = 0;
        for (double v : values) {
            double diff = v - m;
            sum += diff * diff;
        }
        return Math.sqrt(sum / values.length);
    }
    
    /**
     * Get sorted feature indices by importance (descending).
     */
    public int[] getSortedIndices() {
        int[] indices = new int[importances.length];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        
        for (int i = 0; i < indices.length - 1; i++) {
            for (int j = i + 1; j < indices.length; j++) {
                if (importances[indices[j]] > importances[indices[i]]) {
                    int temp = indices[i];
                    indices[i] = indices[j];
                    indices[j] = temp;
                }
            }
        }
        return indices;
    }
    
    public double[] getImportances() { return importances; }
    public double[] getImportancesStd() { return importancesStd; }
    public String[] getFeatureNames() { return featureNames; }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Feature Importances:\n");
        int[] sorted = getSortedIndices();
        for (int idx : sorted) {
            String name = (featureNames != null && idx < featureNames.length) ? 
                         featureNames[idx] : "Feature " + idx;
            sb.append(String.format("  %s: %.4f (+/- %.4f)%n", name, importances[idx], importancesStd[idx]));
        }
        return sb.toString();
    }
}
