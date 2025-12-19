package io.github.yasmramos.mindforge.interpret;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class InterpretTest {

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

    @Test
    void testPartialDependenceDefaultGridPoints() {
        double[][] X = {{0, 0}, {1, 1}, {2, 2}};
        
        PartialDependence pdp = new PartialDependence(
            data -> {
                double[] result = new double[data.length];
                for (int i = 0; i < data.length; i++) {
                    result[i] = data[i][0];
                }
                return result;
            });
        
        PartialDependence.PDPResult result = pdp.calculate(X, 0);
        assertEquals(50, result.gridValues.length); // Default is 50
    }
}
