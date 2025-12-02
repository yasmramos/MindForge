package com.mindforge.pipeline;

import com.mindforge.classification.Classifier;
import java.io.Serializable;
import java.util.*;

/**
 * Pipeline for chaining transformers and estimators.
 * 
 * Allows sequential application of transforms and a final estimator.
 * 
 * @author MindForge
 */
public class Pipeline implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private List<Step> steps;
    private boolean fitted;
    
    /**
     * Interface for transformers.
     */
    public interface Transformer extends Serializable {
        void fit(double[][] X, int[] y);
        double[][] transform(double[][] X);
        default double[][] fitTransform(double[][] X, int[] y) {
            fit(X, y);
            return transform(X);
        }
    }
    
    /**
     * Named step in the pipeline.
     */
    public static class Step implements Serializable {
        private static final long serialVersionUID = 1L;
        public final String name;
        public final Object component;
        
        public Step(String name, Object component) {
            this.name = name;
            this.component = component;
        }
    }
    
    /**
     * Creates an empty Pipeline.
     */
    public Pipeline() {
        this.steps = new ArrayList<>();
        this.fitted = false;
    }
    
    /**
     * Creates a Pipeline with specified steps.
     */
    public Pipeline(List<Step> steps) {
        this.steps = new ArrayList<>(steps);
        this.fitted = false;
    }
    
    /**
     * Builder pattern for Pipeline.
     */
    public static class Builder {
        private List<Step> steps = new ArrayList<>();
        
        public Builder addStep(String name, Object component) {
            steps.add(new Step(name, component));
            return this;
        }
        
        public Builder addTransformer(String name, Transformer transformer) {
            steps.add(new Step(name, transformer));
            return this;
        }
        
        public Builder addClassifier(String name, Classifier<double[]> classifier) {
            steps.add(new Step(name, classifier));
            return this;
        }
        
        public Pipeline build() {
            return new Pipeline(steps);
        }
    }
    
    /**
     * Adds a step to the pipeline.
     */
    public Pipeline addStep(String name, Object component) {
        steps.add(new Step(name, component));
        return this;
    }
    
    /**
     * Fits all transformers and the final estimator.
     */
    public Pipeline fit(double[][] X, int[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("X and y cannot be null");
        }
        if (steps.isEmpty()) {
            throw new IllegalStateException("Pipeline has no steps");
        }
        
        double[][] current = X;
        
        for (int i = 0; i < steps.size() - 1; i++) {
            Step step = steps.get(i);
            
            if (step.component instanceof Transformer) {
                Transformer transformer = (Transformer) step.component;
                current = transformer.fitTransform(current, y);
            } else {
                throw new IllegalStateException("Non-final step must be a Transformer: " + step.name);
            }
        }
        
        // Fit final estimator
        Step finalStep = steps.get(steps.size() - 1);
        if (finalStep.component instanceof Classifier) {
            @SuppressWarnings("unchecked")
            Classifier<double[]> classifier = (Classifier<double[]>) finalStep.component;
            classifier.train(current, y);
        } else if (finalStep.component instanceof Transformer) {
            ((Transformer) finalStep.component).fit(current, y);
        }
        
        fitted = true;
        return this;
    }
    
    /**
     * Transforms data through all transformers.
     */
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline not fitted");
        }
        
        double[][] current = X;
        
        for (int i = 0; i < steps.size() - 1; i++) {
            Step step = steps.get(i);
            if (step.component instanceof Transformer) {
                current = ((Transformer) step.component).transform(current);
            }
        }
        
        // If final step is also a transformer
        Step finalStep = steps.get(steps.size() - 1);
        if (finalStep.component instanceof Transformer) {
            current = ((Transformer) finalStep.component).transform(current);
        }
        
        return current;
    }
    
    /**
     * Predicts using the final estimator.
     */
    public int[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline not fitted");
        }
        
        double[][] current = X;
        
        // Transform through all but final step
        for (int i = 0; i < steps.size() - 1; i++) {
            Step step = steps.get(i);
            if (step.component instanceof Transformer) {
                current = ((Transformer) step.component).transform(current);
            }
        }
        
        // Predict with final estimator
        Step finalStep = steps.get(steps.size() - 1);
        if (finalStep.component instanceof Classifier) {
            @SuppressWarnings("unchecked")
            Classifier<double[]> classifier = (Classifier<double[]>) finalStep.component;
            return classifier.predict(current);
        }
        
        throw new IllegalStateException("Final step is not a classifier");
    }
    
    /**
     * Fits and predicts in one step.
     */
    public int[] fitPredict(double[][] X, int[] y) {
        fit(X, y);
        return predict(X);
    }
    
    /**
     * Computes accuracy score.
     */
    public double score(double[][] X, int[] y) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline not fitted");
        }
        
        int[] predictions = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) correct++;
        }
        return (double) correct / y.length;
    }
    
    /**
     * Gets a step by name.
     */
    public Object getStep(String name) {
        for (Step step : steps) {
            if (step.name.equals(name)) {
                return step.component;
            }
        }
        return null;
    }
    
    /**
     * Gets all step names.
     */
    public List<String> getStepNames() {
        List<String> names = new ArrayList<>();
        for (Step step : steps) {
            names.add(step.name);
        }
        return names;
    }
    
    public boolean isFitted() {
        return fitted;
    }
    
    public int getNumSteps() {
        return steps.size();
    }
}
