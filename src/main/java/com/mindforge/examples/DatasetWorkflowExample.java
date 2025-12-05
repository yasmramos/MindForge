package com.mindforge.examples;

import com.mindforge.data.*;
import com.mindforge.preprocessing.*;
import com.mindforge.util.ArrayUtils;

/**
 * Demonstrates complete dataset workflow in MindForge.
 * 
 * This example shows:
 * - Loading built-in datasets (Iris, Wine, Breast Cancer)
 * - Creating synthetic datasets (classification, circles)
 * - Train/test splitting with stratification
 * - Data preprocessing and normalization
 * - Dataset manipulation operations
 * 
 * @author MindForge Team
 * @version 1.2.0-alpha
 */
public class DatasetWorkflowExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("MindForge Dataset Workflow Example");
        System.out.println("=".repeat(60));
        
        // 1. Loading Built-in Datasets
        System.out.println("\n1. Loading Built-in Datasets:");
        System.out.println("-".repeat(40));
        
        // Iris Dataset
        Dataset iris = DatasetLoader.loadIris();
        System.out.println("   Iris Dataset:");
        System.out.println("     Samples: " + iris.getFeatures().length);
        System.out.println("     Features: " + iris.getFeatures()[0].length);
        System.out.println("     Classes: 3 (Setosa, Versicolor, Virginica)");
        
        // Wine Dataset
        Dataset wine = DatasetLoader.loadWine();
        System.out.println("\n   Wine Dataset:");
        System.out.println("     Samples: " + wine.getFeatures().length);
        System.out.println("     Features: " + wine.getFeatures()[0].length);
        System.out.println("     Classes: 3");
        
        // Breast Cancer Dataset
        Dataset cancer = DatasetLoader.loadBreastCancer();
        System.out.println("\n   Breast Cancer Dataset:");
        System.out.println("     Samples: " + cancer.getFeatures().length);
        System.out.println("     Features: " + cancer.getFeatures()[0].length);
        System.out.println("     Classes: 2 (Malignant, Benign)");
        
        // 2. Creating Synthetic Datasets
        System.out.println("\n2. Creating Synthetic Datasets:");
        System.out.println("-".repeat(40));
        
        // Classification dataset (similar to blobs)
        Dataset syntheticClassif = DatasetLoader.makeClassification(200, 2, 3, 42L);
        System.out.println("   Synthetic Classification Dataset:");
        System.out.println("     Samples: " + syntheticClassif.getFeatures().length);
        System.out.println("     Features: " + syntheticClassif.getFeatures()[0].length);
        System.out.println("     Classes: 3");
        
        // Circles for non-linear classification
        Dataset circles = DatasetLoader.makeCircles(75, 0.1, 42L);
        System.out.println("\n   Circles Dataset:");
        System.out.println("     Samples: " + circles.getFeatures().length);
        System.out.println("     Features: 2");
        System.out.println("     Pattern: Concentric circles");
        
        // 3. Train/Test Split
        System.out.println("\n3. Train/Test Split:");
        System.out.println("-".repeat(40));
        
        // trainTestSplit requires (testSize, seed)
        Dataset[] split = iris.trainTestSplit(0.2, 42L);
        Dataset trainSet = split[0];
        Dataset testSet = split[1];
        
        System.out.println("   Original samples: " + iris.getFeatures().length);
        System.out.println("   Training samples: " + trainSet.getFeatures().length + " (80%)");
        System.out.println("   Test samples: " + testSet.getFeatures().length + " (20%)");
        
        // Stratified split using DataSplit
        System.out.println("\n   Stratified Split:");
        DataSplit.TrainTestSplit stratifiedSplit = DataSplit.stratifiedTrainTestSplit(
            iris.getFeatures(), iris.getLabels(), 0.2, 42
        );
        
        // Access public fields directly instead of methods
        System.out.println("     Training samples: " + stratifiedSplit.XTrain.length);
        System.out.println("     Test samples: " + stratifiedSplit.XTest.length);
        System.out.println("     (Class distribution preserved)");
        
        // 4. Data Preprocessing
        System.out.println("\n4. Data Preprocessing:");
        System.out.println("-".repeat(40));
        
        double[][] features = trainSet.getFeatures();
        
        // Standard Scaling
        System.out.println("   Standard Scaling (zero mean, unit variance):");
        StandardScaler stdScaler = new StandardScaler();
        stdScaler.fit(features);
        double[][] stdScaled = stdScaler.transform(features);
        
        System.out.println("     Before - Feature 0 mean: " + String.format("%.4f", ArrayUtils.mean(getColumn(features, 0))));
        System.out.println("     After  - Feature 0 mean: " + String.format("%.4f", ArrayUtils.mean(getColumn(stdScaled, 0))));
        System.out.println("     After  - Feature 0 std:  " + String.format("%.4f", ArrayUtils.std(getColumn(stdScaled, 0))));
        
        // Min-Max Scaling
        System.out.println("\n   Min-Max Scaling (range [0, 1]):");
        MinMaxScaler mmScaler = new MinMaxScaler();
        mmScaler.fit(features);
        double[][] mmScaled = mmScaler.transform(features);
        
        double[] col0 = getColumn(mmScaled, 0);
        double minVal = Double.MAX_VALUE, maxVal = Double.MIN_VALUE;
        for (double v : col0) {
            minVal = Math.min(minVal, v);
            maxVal = Math.max(maxVal, v);
        }
        System.out.println("     Feature 0 min: " + String.format("%.4f", minVal));
        System.out.println("     Feature 0 max: " + String.format("%.4f", maxVal));
        
        // 5. Dataset Operations
        System.out.println("\n5. Dataset Operations:");
        System.out.println("-".repeat(40));
        
        // Subset
        System.out.println("   Creating subset (first 50 samples)...");
        Dataset subset = iris.subset(0, 50);
        System.out.println("     Subset size: " + subset.getFeatures().length);
        
        // 6. Label Encoding
        System.out.println("\n6. Label Encoding:");
        System.out.println("-".repeat(40));
        
        String[] categories = {"cat", "dog", "bird", "cat", "bird", "dog", "cat"};
        System.out.println("   Original: [cat, dog, bird, cat, bird, dog, cat]");
        
        LabelEncoder encoder = new LabelEncoder();
        int[] encoded = encoder.fitTransform(categories);
        System.out.print("   Encoded:  [");
        for (int i = 0; i < encoded.length; i++) {
            System.out.print(encoded[i]);
            if (i < encoded.length - 1) System.out.print(", ");
        }
        System.out.println("]");
        
        String[] decoded = encoder.inverseTransform(new int[]{0, 1, 2});
        System.out.println("   Decoded [0,1,2]: [" + String.join(", ", decoded) + "]");
        
        // 7. One-Hot Encoding
        System.out.println("\n7. One-Hot Encoding:");
        System.out.println("-".repeat(40));
        
        String[][] catData = {{"red"}, {"green"}, {"blue"}, {"red"}};
        System.out.println("   Original: [[red], [green], [blue], [red]]");
        
        OneHotEncoder ohEncoder = new OneHotEncoder();
        ohEncoder.fit(catData);
        double[][] oneHot = ohEncoder.transform(catData);
        
        System.out.println("   One-Hot Encoded:");
        for (int i = 0; i < oneHot.length; i++) {
            System.out.print("     ");
            for (int j = 0; j < oneHot[i].length; j++) {
                System.out.print((int) oneHot[i][j] + " ");
            }
            System.out.println();
        }
        
        // 8. Missing Value Imputation
        System.out.println("\n8. Missing Value Imputation:");
        System.out.println("-".repeat(40));
        
        double[][] dataWithNaN = {
            {1.0, 2.0, Double.NaN},
            {4.0, Double.NaN, 6.0},
            {7.0, 8.0, 9.0}
        };
        System.out.println("   Original (with NaN):");
        printMatrix(dataWithNaN, "     ");
        
        SimpleImputer imputer = new SimpleImputer(SimpleImputer.ImputeStrategy.MEAN);
        imputer.fit(dataWithNaN);
        double[][] imputed = imputer.transform(dataWithNaN);
        
        System.out.println("\n   Imputed (mean strategy):");
        printMatrix(imputed, "     ");
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Example completed successfully!");
        System.out.println("=".repeat(60));
    }
    
    private static double[] getColumn(double[][] matrix, int col) {
        double[] column = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            column[i] = matrix[i][col];
        }
        return column;
    }
    
    private static void printMatrix(double[][] matrix, String indent) {
        for (double[] row : matrix) {
            System.out.print(indent + "[");
            for (int j = 0; j < row.length; j++) {
                if (Double.isNaN(row[j])) {
                    System.out.print(" NaN ");
                } else {
                    System.out.print(String.format("%5.2f", row[j]));
                }
                if (j < row.length - 1) System.out.print(", ");
            }
            System.out.println("]");
        }
    }
}
