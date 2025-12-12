package io.github.yasmramos.mindforge.clustering;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class SpectralClusteringTest {
    
    @Test
    void testBasicClustering() {
        Random random = new Random(42);
        double[][] data = new double[60][2];
        
        for (int i = 0; i < 20; i++) {
            data[i][0] = random.nextGaussian() * 0.3;
            data[i][1] = random.nextGaussian() * 0.3;
        }
        for (int i = 20; i < 40; i++) {
            data[i][0] = 3 + random.nextGaussian() * 0.3;
            data[i][1] = random.nextGaussian() * 0.3;
        }
        for (int i = 40; i < 60; i++) {
            data[i][0] = 1.5 + random.nextGaussian() * 0.3;
            data[i][1] = 3 + random.nextGaussian() * 0.3;
        }
        
        SpectralClustering sc = new SpectralClustering(3, 1.0, 300, new Random(123));
        int[] labels = sc.cluster(data);
        
        assertEquals(60, labels.length);
        assertNotNull(sc.getEmbedding());
        
        for (int label : labels) {
            assertTrue(label >= 0 && label < 3);
        }
    }
    
    @Test
    void testTwoClusters() {
        double[][] data = new double[40][2];
        Random random = new Random(42);
        
        for (int i = 0; i < 20; i++) {
            data[i][0] = random.nextGaussian() * 0.2;
            data[i][1] = random.nextGaussian() * 0.2;
        }
        for (int i = 20; i < 40; i++) {
            data[i][0] = 5 + random.nextGaussian() * 0.2;
            data[i][1] = 5 + random.nextGaussian() * 0.2;
        }
        
        SpectralClustering sc = new SpectralClustering(2);
        int[] labels = sc.cluster(data);
        
        assertEquals(40, labels.length);
        assertEquals(2, sc.getNumClusters());
    }
    
    @Test
    void testEmbeddingDimensions() {
        double[][] data = new double[30][5];
        Random random = new Random(42);
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < 5; j++) {
                data[i][j] = random.nextGaussian();
            }
        }
        
        SpectralClustering sc = new SpectralClustering(3);
        sc.cluster(data);
        
        double[][] embedding = sc.getEmbedding();
        assertEquals(30, embedding.length);
        assertEquals(3, embedding[0].length);
    }
}
