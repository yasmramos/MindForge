package io.github.yasmramos.mindforge.interpret;

import java.util.*;
import java.util.function.Function;

/**
 * Partial Dependence Plot (PDP) calculator.
 * Shows the marginal effect of features on predictions.
 */
public class PartialDependence {
    
    private final Function<double[][], double[]> predictor;
    private final int numGridPoints;
    
    public PartialDependence(Function<double[][], double[]> predictor) {
        this(predictor, 50);
    }
    
    public PartialDependence(Function<double[][], double[]> predictor, int numGridPoints) {
        this.predictor = predictor;
        this.numGridPoints = numGridPoints;
    }
    
    /**
     * Calculate partial dependence for a single feature.
     */
    public PDPResult calculate(double[][] X, int featureIndex) {
        double[] featureValues = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            featureValues[i] = X[i][featureIndex];
        }
        
        double min = Arrays.stream(featureValues).min().orElse(0);
        double max = Arrays.stream(featureValues).max().orElse(1);
        
        double[] gridValues = new double[numGridPoints];
        double[] pdpValues = new double[numGridPoints];
        
        for (int g = 0; g < numGridPoints; g++) {
            gridValues[g] = min + (max - min) * g / (numGridPoints - 1);
            
            double[][] XModified = new double[X.length][X[0].length];
            for (int i = 0; i < X.length; i++) {
                System.arraycopy(X[i], 0, XModified[i], 0, X[i].length);
                XModified[i][featureIndex] = gridValues[g];
            }
            
            double[] predictions = predictor.apply(XModified);
            pdpValues[g] = Arrays.stream(predictions).average().orElse(0);
        }
        
        return new PDPResult(featureIndex, gridValues, pdpValues);
    }
    
    /**
     * Calculate partial dependence for two features (2D PDP).
     */
    public PDP2DResult calculate2D(double[][] X, int feature1, int feature2, int gridSize) {
        double[] feat1Values = new double[X.length];
        double[] feat2Values = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            feat1Values[i] = X[i][feature1];
            feat2Values[i] = X[i][feature2];
        }
        
        double min1 = Arrays.stream(feat1Values).min().orElse(0);
        double max1 = Arrays.stream(feat1Values).max().orElse(1);
        double min2 = Arrays.stream(feat2Values).min().orElse(0);
        double max2 = Arrays.stream(feat2Values).max().orElse(1);
        
        double[] grid1 = new double[gridSize];
        double[] grid2 = new double[gridSize];
        double[][] pdpValues = new double[gridSize][gridSize];
        
        for (int i = 0; i < gridSize; i++) {
            grid1[i] = min1 + (max1 - min1) * i / (gridSize - 1);
            grid2[i] = min2 + (max2 - min2) * i / (gridSize - 1);
        }
        
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                double[][] XModified = new double[X.length][X[0].length];
                for (int k = 0; k < X.length; k++) {
                    System.arraycopy(X[k], 0, XModified[k], 0, X[k].length);
                    XModified[k][feature1] = grid1[i];
                    XModified[k][feature2] = grid2[j];
                }
                
                double[] predictions = predictor.apply(XModified);
                pdpValues[i][j] = Arrays.stream(predictions).average().orElse(0);
            }
        }
        
        return new PDP2DResult(feature1, feature2, grid1, grid2, pdpValues);
    }
    
    public static class PDPResult {
        public final int featureIndex;
        public final double[] gridValues;
        public final double[] pdpValues;
        
        public PDPResult(int featureIndex, double[] gridValues, double[] pdpValues) {
            this.featureIndex = featureIndex;
            this.gridValues = gridValues;
            this.pdpValues = pdpValues;
        }
    }
    
    public static class PDP2DResult {
        public final int feature1Index;
        public final int feature2Index;
        public final double[] grid1;
        public final double[] grid2;
        public final double[][] pdpValues;
        
        public PDP2DResult(int feature1Index, int feature2Index, 
                          double[] grid1, double[] grid2, double[][] pdpValues) {
            this.feature1Index = feature1Index;
            this.feature2Index = feature2Index;
            this.grid1 = grid1;
            this.grid2 = grid2;
            this.pdpValues = pdpValues;
        }
    }
}
