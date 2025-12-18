package io.github.yasmramos.mindforge.interpret;

import io.github.yasmramos.mindforge.classification.LogisticRegression;
import io.github.yasmramos.mindforge.regression.LinearRegression;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class InterpretTest {

    @Test
    void testFeatureImportanceClassifier() {
        double[][] X = {
            {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4},
            {5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}
        };
        int[] y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
        
        LogisticRegression model = new LogisticRegression();
        model.fit(X, y);
        
        FeatureImportance fi = new FeatureImportance(5, 42);
        fi.compute(model, X, y, new String[]{"feature1", "feature2"});
        
        assertNotNull(fi.getImportances());
        assertNotNull(fi.getImportancesStd());
        assertNotNull(fi.getFeatureNames());
        assertEquals(2, fi.getImportances().length);
    }

    @Test
    void testFeatureImportanceRegressor() {
        double[][] X = {
            {1, 0}, {2, 1}, {3, 2}, {4, 3}, {5, 4}
        };
        double[] y = {2, 4, 6, 8, 10};
        
        LinearRegression model = new LinearRegression();
        model.fit(X, y);
        
        FeatureImportance fi = new FeatureImportance();
        fi.compute(model, X, y, null);
        
        assertNotNull(fi.getImportances());
        assertEquals(2, fi.getImportances().length);
    }

    @Test
    void testFeatureImportanceSortedIndices() {
        double[][] X = {
            {0, 5}, {1, 4}, {2, 3}, {3, 2}, {4, 1},
            {5, 0}, {6, 1}, {7, 2}, {8, 3}, {9, 4}
        };
        int[] y = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
        
        LogisticRegression model = new LogisticRegression();
        model.fit(X, y);
        
        FeatureImportance fi = new FeatureImportance();
        fi.compute(model, X, y, new String[]{"f1", "f2"});
        
        int[] sorted = fi.getSortedIndices();
        assertNotNull(sorted);
        assertEquals(2, sorted.length);
    }

    @Test
    void testFeatureImportanceToString() {
        double[][] X = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}};
        int[] y = {0, 0, 1, 1, 1};
        
        LogisticRegression model = new LogisticRegression();
        model.fit(X, y);
        
        FeatureImportance fi = new FeatureImportance();
        fi.compute(model, X, y, new String[]{"a", "b"});
        
        String str = fi.toString();
        assertNotNull(str);
        assertTrue(str.contains("Feature Importances"));
    }

    @Test
    void testPartialDependence1D() {
        double[][] X = {
            {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}
        };
        
        // Simple predictor that returns sum of features
        PartialDependence pdp = new PartialDependence(
            data -> {
                double[] result = new double[data.length];
                for (int i = 0; i < data.length; i++) {
                    result[i] = data[i][0] + data[i][1];
                }
                return result;
            }, 10);
        
        PartialDependence.PDPResult result = pdp.calculate(X, 0);
        
        assertNotNull(result);
        assertNotNull(result.gridValues);
        assertNotNull(result.pdpValues);
        assertEquals(10, result.gridValues.length);
        assertEquals(10, result.pdpValues.length);
    }

    @Test
    void testPartialDependence2D() {
        double[][] X = {
            {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}
        };
        
        PartialDependence pdp = new PartialDependence(
            data -> {
                double[] result = new double[data.length];
                for (int i = 0; i < data.length; i++) {
                    result[i] = data[i][0] * data[i][1];
                }
                return result;
            });
        
        PartialDependence.PDP2DResult result = pdp.calculate2D(X, 0, 1, 5);
        
        assertNotNull(result);
        assertNotNull(result.grid1);
        assertNotNull(result.grid2);
        assertNotNull(result.pdpValues);
        assertEquals(5, result.grid1.length);
        assertEquals(5, result.pdpValues.length);
    }
}
