package io.github.yasmramos.mindforge.interpret;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.function.Function;

/**
 * Tests for SHAP and LIME interpretability classes.
 */
public class SHAPLIMETest {
    
    // Simple linear model for testing: y = 2*x0 + 3*x1 + 1
    private Function<double[], Double> linearModel = x -> 2.0 * x[0] + 3.0 * x[1] + 1.0;
    
    @Test
    public void testSHAPExplain() {
        double[][] background = {
            {0, 0}, {1, 0}, {0, 1}, {1, 1},
            {0.5, 0.5}, {0.2, 0.8}, {0.8, 0.2}
        };
        
        SHAP shap = new SHAP.Builder()
            .nSamples(200)
            .seed(42)
            .build();
        
        shap.setBackground(background);
        
        double[] instance = {1.0, 1.0};
        double[] shapValues = shap.explain(linearModel, instance);
        
        assertNotNull(shapValues);
        assertEquals(2, shapValues.length);
        
        // SHAP values should roughly correspond to feature contributions
        // For linear model, x0 contributes ~2 and x1 contributes ~3
        assertTrue(shapValues[1] > shapValues[0]); // x1 should have higher contribution
    }
    
    @Test
    public void testSHAPExplainBatch() {
        double[][] background = {{0, 0}, {1, 1}, {0.5, 0.5}};
        
        SHAP shap = new SHAP.Builder()
            .nSamples(100)
            .build();
        
        shap.setBackground(background);
        
        double[][] instances = {{1.0, 0.0}, {0.0, 1.0}};
        double[][] shapValues = shap.explainBatch(linearModel, instances);
        
        assertNotNull(shapValues);
        assertEquals(2, shapValues.length);
        assertEquals(2, shapValues[0].length);
    }
    
    @Test
    public void testSHAPMeanAbsolute() {
        double[][] background = {{0, 0}, {1, 1}};
        
        SHAP shap = new SHAP.Builder()
            .nSamples(100)
            .build();
        
        shap.setBackground(background);
        
        double[][] instances = {
            {1.0, 0.5}, {0.5, 1.0}, {0.8, 0.2}
        };
        
        double[] meanAbs = shap.meanAbsoluteShap(linearModel, instances);
        
        assertNotNull(meanAbs);
        assertEquals(2, meanAbs.length);
        assertTrue(meanAbs[0] >= 0);
        assertTrue(meanAbs[1] >= 0);
    }
    
    @Test
    public void testSHAPExpectedValue() {
        double[][] background = {{0, 0}, {1, 1}, {2, 2}};
        
        SHAP shap = new SHAP.Builder().build();
        shap.setBackground(background);
        
        shap.explain(linearModel, new double[]{1, 1});
        
        // Expected value should be the average prediction on background
        // (1 + 6 + 11) / 3 = 6
        assertEquals(6.0, shap.getExpectedValue(), 0.001);
    }
    
    @Test
    public void testSHAPWithoutBackground() {
        SHAP shap = new SHAP.Builder().build();
        
        assertThrows(IllegalStateException.class, () -> 
            shap.explain(linearModel, new double[]{1, 1}));
    }
    
    @Test
    public void testLIMEExplain() {
        double[][] trainingData = {
            {0, 0}, {1, 0}, {0, 1}, {1, 1},
            {0.5, 0.5}, {0.2, 0.8}, {0.8, 0.2}
        };
        
        LIME lime = new LIME.Builder()
            .nSamples(500)
            .kernelWidth(0.5)
            .seed(42)
            .build();
        
        lime.fitFeatureStd(trainingData);
        
        double[] instance = {0.5, 0.5};
        LIME.Explanation explanation = lime.explain(linearModel, instance);
        
        assertNotNull(explanation);
        
        double[] weights = explanation.getFeatureWeights();
        assertEquals(2, weights.length);
        
        // For linear model, LIME should recover approximately the true coefficients
        // x0 coefficient ~2, x1 coefficient ~3
        assertTrue(weights[1] > weights[0]);
    }
    
    @Test
    public void testLIMEExplanationPrediction() {
        double[][] trainingData = {{0, 0}, {1, 1}};
        
        LIME lime = new LIME.Builder()
            .nSamples(100)
            .build();
        
        lime.fitFeatureStd(trainingData);
        
        double[] instance = {1.0, 1.0};
        LIME.Explanation explanation = lime.explain(linearModel, instance);
        
        // Prediction should match the model output
        assertEquals(linearModel.apply(instance), explanation.getPrediction(), 0.001);
    }
    
    @Test
    public void testLIMETopFeatures() {
        double[][] trainingData = {{0, 0, 0}, {1, 1, 1}};
        
        // Model where x2 has highest contribution
        Function<double[], Double> model = x -> x[0] + 2*x[1] + 5*x[2];
        
        LIME lime = new LIME.Builder()
            .nSamples(500)
            .build();
        
        lime.fitFeatureStd(trainingData);
        
        double[] instance = {1.0, 1.0, 1.0};
        LIME.Explanation explanation = lime.explain(model, instance);
        
        int[] topFeatures = explanation.getTopFeatures(2);
        
        assertEquals(2, topFeatures.length);
        // x2 (index 2) should be the most important
        assertEquals(2, topFeatures[0]);
    }
    
    @Test
    public void testLIMEWithoutFeatureStd() {
        LIME lime = new LIME.Builder().build();
        
        assertThrows(IllegalStateException.class, () -> 
            lime.explain(linearModel, new double[]{1, 1}));
    }
    
    @Test
    public void testLIMESetFeatureStd() {
        LIME lime = new LIME.Builder()
            .nSamples(100)
            .build();
        
        lime.setFeatureStd(new double[]{1.0, 1.0});
        
        // Should not throw
        LIME.Explanation explanation = lime.explain(linearModel, new double[]{0.5, 0.5});
        assertNotNull(explanation);
    }
    
    @Test
    public void testLIMEIntercept() {
        double[][] trainingData = {{0, 0}, {1, 1}};
        
        LIME lime = new LIME.Builder()
            .nSamples(200)
            .build();
        
        lime.fitFeatureStd(trainingData);
        
        double[] instance = {0.5, 0.5};
        LIME.Explanation explanation = lime.explain(linearModel, instance);
        
        // Check that intercept + weights*instance ≈ prediction
        double[] weights = explanation.getFeatureWeights();
        double reconstructed = explanation.getIntercept() + 
            weights[0] * instance[0] + weights[1] * instance[1];
        
        assertEquals(explanation.getPrediction(), reconstructed, 0.5);
    }
}
