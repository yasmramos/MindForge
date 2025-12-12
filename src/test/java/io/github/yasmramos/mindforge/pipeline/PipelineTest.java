package io.github.yasmramos.mindforge.pipeline;

import io.github.yasmramos.mindforge.classification.*;
import io.github.yasmramos.mindforge.preprocessing.*;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * Tests for Pipeline, GridSearchCV, and ColumnTransformer.
 */
class PipelineTest {
    
    private double[][] X;
    private int[] y;
    
    @BeforeEach
    void setUp() {
        X = new double[][]{
            {1, 100}, {2, 200}, {3, 300}, {4, 400},
            {10, 1000}, {11, 1100}, {12, 1200}, {13, 1300}
        };
        y = new int[]{0, 0, 0, 0, 1, 1, 1, 1};
    }
    
    @Nested
    @DisplayName("Pipeline Tests")
    class PipelineBasicTests {
        
        @Test
        @DisplayName("Empty pipeline")
        void testEmptyPipeline() {
            Pipeline pipe = new Pipeline();
            assertNotNull(pipe);
            assertEquals(0, pipe.getNumSteps());
        }
        
        @Test
        @DisplayName("Add steps manually")
        void testAddSteps() {
            Pipeline pipe = new Pipeline();
            pipe.addStep("scaler", new StandardScalerTransformer());
            pipe.addStep("classifier", new KNearestNeighbors(1));
            
            assertEquals(2, pipe.getNumSteps());
        }
        
        @Test
        @DisplayName("Fit and predict")
        void testFitAndPredict() {
            Pipeline pipe = new Pipeline.Builder()
                .addStep("scaler", new StandardScalerTransformer())
                .addClassifier("classifier", new KNearestNeighbors(1))
                .build();
            
            pipe.fit(X, y);
            
            assertTrue(pipe.isFitted());
            int[] predictions = pipe.predict(X);
            assertEquals(X.length, predictions.length);
        }
        
        @Test
        @DisplayName("Score computes accuracy")
        void testScore() {
            Pipeline pipe = new Pipeline.Builder()
                .addStep("scaler", new StandardScalerTransformer())
                .addClassifier("classifier", new KNearestNeighbors(1))
                .build();
            
            pipe.fit(X, y);
            double score = pipe.score(X, y);
            
            assertTrue(score >= 0 && score <= 1);
        }
        
        @Test
        @DisplayName("Get step by name")
        void testGetStep() {
            StandardScalerTransformer scaler = new StandardScalerTransformer();
            Pipeline pipe = new Pipeline.Builder()
                .addStep("scaler", scaler)
                .addClassifier("classifier", new KNearestNeighbors(1))
                .build();
            
            Object retrieved = pipe.getStep("scaler");
            assertSame(scaler, retrieved);
        }
    }
    
    @Nested
    @DisplayName("GridSearchCV Tests")
    class GridSearchCVTests {
        
        @Test
        @DisplayName("Grid search finds best params")
        void testGridSearch() {
            Map<String, Object[]> paramGrid = new HashMap<>();
            paramGrid.put("k", new Object[]{1, 3, 5});
            
            GridSearchCV gs = new GridSearchCV(
                () -> new KNearestNeighbors(1),
                paramGrid,
                3,
                "accuracy",
                true
            );
            
            gs.fit(X, y);
            
            assertTrue(gs.isFitted());
            assertNotNull(gs.getBestParams());
            assertTrue(gs.getBestScore() >= 0);
        }
        
        @Test
        @DisplayName("Predict with best estimator")
        void testPredict() {
            Map<String, Object[]> paramGrid = new HashMap<>();
            paramGrid.put("k", new Object[]{1, 3});
            
            GridSearchCV gs = new GridSearchCV(
                () -> new KNearestNeighbors(1),
                paramGrid
            );
            
            gs.fit(X, y);
            
            int[] predictions = gs.predict(X);
            assertEquals(X.length, predictions.length);
        }
        
        @Test
        @DisplayName("CV results contain all combinations")
        void testCvResults() {
            Map<String, Object[]> paramGrid = new HashMap<>();
            paramGrid.put("p1", new Object[]{1, 2});
            paramGrid.put("p2", new Object[]{"a", "b"});
            
            GridSearchCV gs = new GridSearchCV(
                () -> new KNearestNeighbors(1),
                paramGrid
            );
            
            gs.fit(X, y);
            
            List<Map<String, Object>> results = gs.getCvResults();
            assertEquals(4, results.size()); // 2 x 2 combinations
        }
    }
    
    @Nested
    @DisplayName("ColumnTransformer Tests")
    class ColumnTransformerTests {
        
        @Test
        @DisplayName("Empty transformer")
        void testEmptyTransformer() {
            ColumnTransformer ct = new ColumnTransformer();
            assertNotNull(ct);
            assertEquals(0, ct.getNumTransformers());
        }
        
        @Test
        @DisplayName("Add and apply transformers")
        void testAddTransformers() {
            ColumnTransformer ct = new ColumnTransformer.Builder()
                .addTransformer("first", new IdentityTransformer(), 0)
                .remainder("passthrough")
                .build();
            
            ct.fit(X, y);
            
            assertTrue(ct.isFitted());
            double[][] transformed = ct.transform(X);
            assertNotNull(transformed);
        }
        
        @Test
        @DisplayName("Get transformer by name")
        void testGetTransformer() {
            IdentityTransformer id = new IdentityTransformer();
            ColumnTransformer ct = new ColumnTransformer.Builder()
                .addTransformer("identity", id, 0, 1)
                .build();
            
            Pipeline.Transformer retrieved = ct.getTransformer("identity");
            assertSame(id, retrieved);
        }
        
        @Test
        @DisplayName("Remainder passthrough")
        void testRemainderPassthrough() {
            ColumnTransformer ct = new ColumnTransformer.Builder()
                .addTransformer("first", new IdentityTransformer(), 0)
                .remainder("passthrough")
                .build();
            
            ct.fit(X, y);
            double[][] transformed = ct.transform(X);
            
            // Should have both columns
            assertEquals(2, transformed[0].length);
        }
    }
    
    // Helper transformers for testing
    static class StandardScalerTransformer implements Pipeline.Transformer {
        private static final long serialVersionUID = 1L;
        private double[] means;
        private double[] stds;
        
        @Override
        public void fit(double[][] X, int[] y) {
            int m = X[0].length;
            means = new double[m];
            stds = new double[m];
            
            for (int j = 0; j < m; j++) {
                for (double[] row : X) means[j] += row[j];
                means[j] /= X.length;
                
                for (double[] row : X) stds[j] += Math.pow(row[j] - means[j], 2);
                stds[j] = Math.sqrt(stds[j] / X.length);
                if (stds[j] == 0) stds[j] = 1;
            }
        }
        
        @Override
        public double[][] transform(double[][] X) {
            double[][] result = new double[X.length][X[0].length];
            for (int i = 0; i < X.length; i++) {
                for (int j = 0; j < X[0].length; j++) {
                    result[i][j] = (X[i][j] - means[j]) / stds[j];
                }
            }
            return result;
        }
    }
    
    static class IdentityTransformer implements Pipeline.Transformer {
        private static final long serialVersionUID = 1L;
        
        @Override
        public void fit(double[][] X, int[] y) {}
        
        @Override
        public double[][] transform(double[][] X) {
            double[][] result = new double[X.length][];
            for (int i = 0; i < X.length; i++) {
                result[i] = X[i].clone();
            }
            return result;
        }
    }
}
