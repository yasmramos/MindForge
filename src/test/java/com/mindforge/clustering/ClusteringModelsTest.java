package com.mindforge.clustering;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for clustering models: DBSCAN, HierarchicalClustering, MeanShift.
 */
class ClusteringModelsTest {
    
    private double[][] X;
    
    @BeforeEach
    void setUp() {
        // Three clear clusters
        X = new double[][]{
            {0, 0}, {0.1, 0.1}, {0.2, 0},
            {5, 5}, {5.1, 5.1}, {5, 5.2},
            {10, 0}, {10.1, 0.1}, {10, 0.2}
        };
    }
    
    @Nested
    @DisplayName("DBSCAN Tests")
    class DBSCANTests {
        
        @Test
        @DisplayName("Default constructor works")
        void testDefaultConstructor() {
            DBSCAN dbscan = new DBSCAN();
            assertNotNull(dbscan);
            assertEquals(0.5, dbscan.getEps());
            assertEquals(5, dbscan.getMinSamples());
        }
        
        @Test
        @DisplayName("Fit clusters data")
        void testFit() {
            DBSCAN dbscan = new DBSCAN(1.0, 2);
            int[] labels = dbscan.fit(X);
            
            assertTrue(dbscan.isFitted());
            assertEquals(X.length, labels.length);
            assertTrue(dbscan.getNClusters() > 0);
        }
        
        @Test
        @DisplayName("Identifies noise points")
        void testNoisePoints() {
            // Add an outlier
            double[][] XWithNoise = new double[][]{
                {0, 0}, {0.1, 0.1}, {0.2, 0},
                {5, 5}, {5.1, 5.1}, {5, 5.2},
                {100, 100} // Outlier
            };
            
            DBSCAN dbscan = new DBSCAN(1.0, 2);
            dbscan.fit(XWithNoise);
            
            assertTrue(dbscan.getNumNoise() >= 1);
        }
        
        @Test
        @DisplayName("Gets core indices")
        void testCoreIndices() {
            DBSCAN dbscan = new DBSCAN(1.0, 2);
            dbscan.fit(X);
            
            int[] coreIndices = dbscan.getCoreIndices();
            assertNotNull(coreIndices);
        }
        
        @Test
        @DisplayName("Invalid eps throws exception")
        void testInvalidEps() {
            assertThrows(IllegalArgumentException.class, () -> new DBSCAN(0, 5));
            assertThrows(IllegalArgumentException.class, () -> new DBSCAN(-1, 5));
        }
    }
    
    @Nested
    @DisplayName("HierarchicalClustering Tests")
    class HierarchicalClusteringTests {
        
        @Test
        @DisplayName("Default constructor works")
        void testDefaultConstructor() {
            HierarchicalClustering hc = new HierarchicalClustering();
            assertNotNull(hc);
            assertEquals(HierarchicalClustering.Linkage.WARD, hc.getLinkage());
        }
        
        @Test
        @DisplayName("Fit clusters data")
        void testFit() {
            HierarchicalClustering hc = new HierarchicalClustering(3);
            int[] labels = hc.fit(X);
            
            assertTrue(hc.isFitted());
            assertEquals(X.length, labels.length);
            assertEquals(3, hc.getNClusters());
        }
        
        @Test
        @DisplayName("Different linkage methods")
        void testLinkageMethods() {
            for (HierarchicalClustering.Linkage linkage : HierarchicalClustering.Linkage.values()) {
                HierarchicalClustering hc = new HierarchicalClustering(3, linkage);
                int[] labels = hc.fit(X);
                
                assertTrue(hc.isFitted());
                assertEquals(X.length, labels.length);
            }
        }
        
        @Test
        @DisplayName("Gets linkage matrix")
        void testLinkageMatrix() {
            HierarchicalClustering hc = new HierarchicalClustering(2);
            hc.fit(X);
            
            double[][] linkageMatrix = hc.getLinkageMatrix();
            assertNotNull(linkageMatrix);
        }
        
        @Test
        @DisplayName("Distance threshold mode")
        void testDistanceThreshold() {
            HierarchicalClustering hc = new HierarchicalClustering(2.0, HierarchicalClustering.Linkage.COMPLETE);
            hc.fit(X);
            
            assertTrue(hc.isFitted());
        }
    }
    
    @Nested
    @DisplayName("MeanShift Tests")
    class MeanShiftTests {
        
        @Test
        @DisplayName("Default constructor works")
        void testDefaultConstructor() {
            MeanShift ms = new MeanShift();
            assertNotNull(ms);
        }
        
        @Test
        @DisplayName("Fit clusters data")
        void testFit() {
            MeanShift ms = new MeanShift(2.0);
            int[] labels = ms.fit(X);
            
            assertTrue(ms.isFitted());
            assertEquals(X.length, labels.length);
            assertTrue(ms.getNClusters() > 0);
        }
        
        @Test
        @DisplayName("Auto bandwidth estimation")
        void testAutoBandwidth() {
            MeanShift ms = new MeanShift(-1);
            int[] labels = ms.fit(X);
            
            assertTrue(ms.isFitted());
        }
        
        @Test
        @DisplayName("Gets cluster centers")
        void testClusterCenters() {
            MeanShift ms = new MeanShift(2.0);
            ms.fit(X);
            
            double[][] centers = ms.getClusterCenters();
            assertNotNull(centers);
            assertEquals(ms.getNClusters(), centers.length);
        }
        
        @Test
        @DisplayName("Predict new points")
        void testPredict() {
            MeanShift ms = new MeanShift(2.0);
            ms.fit(X);
            
            double[][] newPoints = {{0, 0}, {5, 5}};
            int[] predictions = ms.predict(newPoints);
            
            assertEquals(2, predictions.length);
        }
        
        @Test
        @DisplayName("Binned seeding")
        void testBinnedSeeding() {
            MeanShift ms = new MeanShift(2.0, 300, 1e-4, true);
            int[] labels = ms.fit(X);
            
            assertTrue(ms.isFitted());
        }
    }
}
