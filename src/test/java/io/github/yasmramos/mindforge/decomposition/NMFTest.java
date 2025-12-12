package io.github.yasmramos.mindforge.decomposition;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class NMFTest {
    
    @Test
    void testBasicNMF() {
        Random random = new Random(42);
        double[][] data = new double[50][10];
        for (int i = 0; i < 50; i++) {
            for (int j = 0; j < 10; j++) {
                data[i][j] = random.nextDouble() * 10;
            }
        }
        
        NMF nmf = new NMF(3, 200, 1e-4, new Random(123));
        double[][] W = nmf.fitTransform(data);
        
        assertEquals(50, W.length);
        assertEquals(3, W[0].length);
        assertNotNull(nmf.getH());
        assertEquals(3, nmf.getH().length);
        assertEquals(10, nmf.getH()[0].length);
    }
    
    @Test
    void testNonNegativity() {
        Random random = new Random(42);
        double[][] data = new double[30][5];
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 5; j++) {
                data[i][j] = random.nextDouble() * 5;
            }
        }
        
        NMF nmf = new NMF(2);
        nmf.fit(data);
        
        double[][] W = nmf.getW();
        double[][] H = nmf.getH();
        
        for (double[] row : W) {
            for (double val : row) {
                assertTrue(val >= 0, "W should be non-negative");
            }
        }
        for (double[] row : H) {
            for (double val : row) {
                assertTrue(val >= 0, "H should be non-negative");
            }
        }
    }
    
    @Test
    void testInverseTransform() {
        Random random = new Random(42);
        double[][] data = new double[20][8];
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 8; j++) {
                data[i][j] = random.nextDouble() * 3;
            }
        }
        
        NMF nmf = new NMF(4);
        double[][] W = nmf.fitTransform(data);
        double[][] reconstructed = nmf.inverseTransform(W);
        
        assertEquals(20, reconstructed.length);
        assertEquals(8, reconstructed[0].length);
    }
    
    @Test
    void testNotFittedThrowsException() {
        NMF nmf = new NMF(2);
        double[][] data = new double[10][5];
        
        assertThrows(IllegalStateException.class, () -> nmf.transform(data));
        assertThrows(IllegalStateException.class, () -> nmf.inverseTransform(data));
    }
}
