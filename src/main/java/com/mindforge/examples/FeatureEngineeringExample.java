package com.mindforge.examples;

import com.mindforge.feature.*;
import com.mindforge.decomposition.PCA;
import com.mindforge.preprocessing.PolynomialFeatures;
import com.mindforge.data.Dataset;
import com.mindforge.data.DatasetLoader;
import com.mindforge.util.ArrayUtils;

/**
 * Demonstrates feature engineering techniques in MindForge.
 * 
 * This example shows:
 * - Feature Selection (SelectKBest, VarianceThreshold, RFE)
 * - Dimensionality Reduction (PCA)
 * - Feature Transformation (Polynomial Features)
 * 
 * @author MindForge Team
 * @version 1.2.0-alpha
 */
public class FeatureEngineeringExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("MindForge Feature Engineering Example");
        System.out.println("=".repeat(60));
        
        // Load dataset
        System.out.println("\n1. Loading Wine Dataset...");
        Dataset wine = DatasetLoader.loadWine();
        double[][] X = wine.getFeatures();
        int[] y = wine.getLabels();
        
        System.out.println("   Samples: " + X.length);
        System.out.println("   Original Features: " + X[0].length);
        
        // 2. Variance Threshold Feature Selection
        System.out.println("\n2. Variance Threshold Selection:");
        System.out.println("-".repeat(40));
        
        VarianceThreshold varSelector = new VarianceThreshold(0.1);
        varSelector.fit(X);
        double[][] X_var = varSelector.transform(X);
        
        System.out.println("   Threshold: 0.1");
        System.out.println("   Features before: " + X[0].length);
        System.out.println("   Features after: " + X_var[0].length);
        System.out.println("   (Low variance features removed)");
        
        // Get variances
        double[] variances = varSelector.getVariances();
        System.out.println("\n   Feature Variances (sample):");
        for (int i = 0; i < Math.min(5, variances.length); i++) {
            System.out.println("     Feature " + i + ": " + String.format("%.4f", variances[i]));
        }
        
        // 3. SelectKBest Feature Selection - using enum instead of String
        System.out.println("\n3. SelectKBest Selection:");
        System.out.println("-".repeat(40));
        
        int k = 5;
        SelectKBest kBest = new SelectKBest(SelectKBest.ScoreFunction.F_CLASSIF, k);
        kBest.fit(X, y);
        double[][] X_kbest = kBest.transform(X);
        
        System.out.println("   Method: F-Classif (ANOVA F-value)");
        System.out.println("   K (features to select): " + k);
        System.out.println("   Features before: " + X[0].length);
        System.out.println("   Features after: " + X_kbest[0].length);
        
        double[] scores = kBest.getScores();
        int[] selectedIndices = kBest.getSelectedFeatureIndices();
        
        System.out.println("\n   Top " + k + " Features (by F-score):");
        for (int i = 0; i < k; i++) {
            int idx = selectedIndices[i];
            System.out.println("     Feature " + idx + ": score = " + String.format("%.4f", scores[idx]));
        }
        
        // 4. Recursive Feature Elimination (RFE)
        System.out.println("\n4. Recursive Feature Elimination (RFE):");
        System.out.println("-".repeat(40));
        
        int nFeatures = 6;
        RFE rfe = new RFE(nFeatures);
        rfe.fit(X, y);
        double[][] X_rfe = rfe.transform(X);
        
        System.out.println("   Features to select: " + nFeatures);
        System.out.println("   Features before: " + X[0].length);
        System.out.println("   Features after: " + X_rfe[0].length);
        
        int[] rankings = rfe.getRanking();
        System.out.println("\n   Feature Rankings (1 = selected):");
        for (int i = 0; i < Math.min(8, rankings.length); i++) {
            System.out.println("     Feature " + i + ": rank " + rankings[i]);
        }
        
        // 5. PCA Dimensionality Reduction
        System.out.println("\n5. Principal Component Analysis (PCA):");
        System.out.println("-".repeat(40));
        
        int nComponents = 3;
        PCA pca = new PCA(nComponents);
        pca.fit(X);
        double[][] X_pca = pca.transform(X);
        
        System.out.println("   Components: " + nComponents);
        System.out.println("   Features before: " + X[0].length);
        System.out.println("   Features after: " + X_pca[0].length);
        
        double[] explainedVariance = pca.getExplainedVarianceRatio();
        double totalVariance = 0;
        System.out.println("\n   Explained Variance Ratio:");
        for (int i = 0; i < nComponents; i++) {
            totalVariance += explainedVariance[i];
            System.out.println("     PC" + (i+1) + ": " + String.format("%.4f", explainedVariance[i]) + 
                             " (cumulative: " + String.format("%.4f", totalVariance) + ")");
        }
        
        // Inverse transform
        System.out.println("\n   Inverse Transform (reconstruction):");
        double[][] X_reconstructed = pca.inverseTransform(X_pca);
        
        double mse = 0;
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[0].length; j++) {
                mse += Math.pow(X[i][j] - X_reconstructed[i][j], 2);
            }
        }
        mse /= (X.length * X[0].length);
        System.out.println("     Reconstruction MSE: " + String.format("%.4f", mse));
        
        // 6. Polynomial Features
        System.out.println("\n6. Polynomial Features:");
        System.out.println("-".repeat(40));
        
        // Use small sample for demonstration
        double[][] smallX = new double[5][2];
        for (int i = 0; i < 5; i++) {
            smallX[i][0] = X[i][0];
            smallX[i][1] = X[i][1];
        }
        
        System.out.println("   Original features: 2 (using first 2 features)");
        System.out.println("   Sample: [" + String.format("%.2f", smallX[0][0]) + ", " + 
                          String.format("%.2f", smallX[0][1]) + "]");
        
        PolynomialFeatures poly = new PolynomialFeatures(2, true, true);
        poly.fit(smallX);
        double[][] X_poly = poly.transform(smallX);
        
        System.out.println("\n   Polynomial degree: 2 (with bias and interaction)");
        System.out.println("   Features after: " + X_poly[0].length);
        System.out.println("   Generated: [1, x1, x2, x1^2, x1*x2, x2^2]");
        
        System.out.print("   Transformed sample: [");
        for (int i = 0; i < X_poly[0].length; i++) {
            System.out.print(String.format("%.2f", X_poly[0][i]));
            if (i < X_poly[0].length - 1) System.out.print(", ");
        }
        System.out.println("]");
        
        // 7. Feature Engineering Pipeline Summary
        System.out.println("\n7. Feature Engineering Summary:");
        System.out.println("-".repeat(40));
        System.out.println("   Original Dataset: " + X.length + " x " + X[0].length);
        System.out.println("\n   Techniques Applied:");
        System.out.println("   +-- VarianceThreshold -> " + X_var[0].length + " features");
        System.out.println("   +-- SelectKBest (k=" + k + ") -> " + X_kbest[0].length + " features");
        System.out.println("   +-- RFE (n=" + nFeatures + ") -> " + X_rfe[0].length + " features");
        System.out.println("   +-- PCA (n=" + nComponents + ") -> " + X_pca[0].length + " components");
        System.out.println("   +-- Polynomial (d=2) -> " + X_poly[0].length + " features (from 2)");
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Example completed successfully!");
        System.out.println("=".repeat(60));
    }
}
