package io.github.yasmramos.mindforge.decomposition;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class ICATest {
    
    @Test
    void testBasicICA() {
        Random random = new Random(42);
        int n = 100;
        int m = 4;
        
        double[][] data = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                data[i][j] = random.nextGaussian();
            }
        }
        
        ICA ica = new ICA(2, 200, 1e-4, new Random(123));
        double[][] sources = ica.fitTransform(data);
        
        assertEquals(n, sources.length);
        assertEquals(2, sources[0].length);
        assertNotNull(ica.getUnmixingMatrix());
        assertNotNull(ica.getMixingMatrix());
    }
    
    @Test
    void testTransformAfterFit() {
        Random random = new Random(42);
        double[][] train = new double[50][3];
        double[][] test = new double[20][3];
        
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < 3; j++) {
                train[i][j] = random.nextGaussian();
            }
        }
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 3; j++) {
                test[i][j] = random.nextGaussian();
            }
        }
        
        ICA ica = new ICA(2);
        ica.fit(train);
        double[][] result = ica.transform(test);
        
        assertEquals(20, result.length);
        assertEquals(2, result[0].length);
    }
    
    @Test
    void testSingleComponent() {
        Random random = new Random(42);
        double[][] data = new double[30][5];
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 5; j++) {
                data[i][j] = random.nextGaussian();
            }
        }
        
        ICA ica = new ICA(1);
        double[][] result = ica.fitTransform(data);
        
        assertEquals(30, result.length);
        assertEquals(1, result[0].length);
        assertEquals(1, ica.getNComponents());
    }
    
    @Test
    void testNotFittedThrowsException() {
        ICA ica = new ICA(2);
        double[][] data = new double[10][3];
        
        assertThrows(IllegalStateException.class, () -> ica.transform(data));
    }
}
