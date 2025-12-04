package com.mindforge.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Dataset Loader Tests")
class DatasetLoaderTest {
    
    @Nested
    @DisplayName("Iris Dataset Tests")
    class IrisDatasetTests {
        
        @Test
        @DisplayName("Should load Iris dataset")
        void testLoadIris() {
            Dataset iris = DatasetLoader.loadIris();
            
            assertNotNull(iris);
            assertEquals(150, iris.getNumSamples(), "Iris should have 150 samples");
            assertEquals(4, iris.getNumFeatures(), "Iris should have 4 features");
        }
        
        @Test
        @DisplayName("Iris should have correct feature names")
        void testIrisFeatureNames() {
            Dataset iris = DatasetLoader.loadIris();
            String[] featureNames = iris.getFeatureNames();
            
            assertNotNull(featureNames);
            assertEquals(4, featureNames.length);
            assertTrue(featureNames[0].toLowerCase().contains("sepal"));
        }
        
        @Test
        @DisplayName("Iris should have 3 classes")
        void testIrisClasses() {
            Dataset iris = DatasetLoader.loadIris();
            
            assertEquals(3, iris.getNumClasses(), "Iris should have 3 classes");
            assertTrue(iris.isClassification());
        }
        
        @Test
        @DisplayName("Iris should have target names")
        void testIrisTargetNames() {
            Dataset iris = DatasetLoader.loadIris();
            String[] targetNames = iris.getTargetNames();
            
            assertNotNull(targetNames);
            assertEquals(3, targetNames.length);
        }
        
        @Test
        @DisplayName("Iris should have name and description")
        void testIrisMetadata() {
            Dataset iris = DatasetLoader.loadIris();
            
            assertNotNull(iris.getName());
            assertNotNull(iris.getDescription());
        }
    }
    
    @Nested
    @DisplayName("Wine Dataset Tests")
    class WineDatasetTests {
        
        @Test
        @DisplayName("Should load Wine dataset")
        void testLoadWine() {
            Dataset wine = DatasetLoader.loadWine();
            
            assertNotNull(wine);
            assertEquals(30, wine.getNumSamples(), "Wine subset should have 30 samples");
            assertEquals(13, wine.getNumFeatures(), "Wine should have 13 features");
        }
        
        @Test
        @DisplayName("Wine should have 2 classes (subset)")
        void testWineClasses() {
            Dataset wine = DatasetLoader.loadWine();
            
            assertEquals(2, wine.getNumClasses(), "Wine subset should have 2 classes");
            assertTrue(wine.isClassification());
        }
        
        @Test
        @DisplayName("Wine should have feature names")
        void testWineFeatureNames() {
            Dataset wine = DatasetLoader.loadWine();
            String[] featureNames = wine.getFeatureNames();
            
            assertNotNull(featureNames);
            assertEquals(13, featureNames.length);
        }
    }
    
    @Nested
    @DisplayName("Breast Cancer Dataset Tests")
    class BreastCancerTests {
        
        @Test
        @DisplayName("Should load Breast Cancer dataset")
        void testLoadBreastCancer() {
            Dataset cancer = DatasetLoader.loadBreastCancer();
            
            assertNotNull(cancer);
            assertEquals(20, cancer.getNumSamples(), "Breast Cancer subset should have 20 samples");
            assertEquals(10, cancer.getNumFeatures(), "Breast Cancer subset should have 10 features");
        }
        
        @Test
        @DisplayName("Breast Cancer should be binary classification")
        void testBreastCancerBinary() {
            Dataset cancer = DatasetLoader.loadBreastCancer();
            
            assertEquals(2, cancer.getNumClasses(), "Breast Cancer should have 2 classes");
            assertTrue(cancer.isClassification());
        }
    }
    
    @Nested
    @DisplayName("Synthetic Dataset Tests")
    class SyntheticDatasetTests {
        
        @Test
        @DisplayName("Should generate classification dataset")
        void testMakeClassification() {
            Dataset dataset = DatasetLoader.makeClassification(100, 5, 3, 42L);
            
            assertNotNull(dataset);
            assertEquals(100, dataset.getNumSamples());
            assertEquals(5, dataset.getNumFeatures());
            assertEquals(3, dataset.getNumClasses());
        }
        
        @Test
        @DisplayName("Should generate regression dataset")
        void testMakeRegression() {
            Dataset dataset = DatasetLoader.makeRegression(100, 5, 0.1, 42L);
            
            assertNotNull(dataset);
            assertEquals(100, dataset.getNumSamples());
            assertEquals(5, dataset.getNumFeatures());
            assertFalse(dataset.isClassification());
        }
        
        @Test
        @DisplayName("Should generate XOR dataset")
        void testMakeXOR() {
            Dataset dataset = DatasetLoader.makeXOR(25, 0.1, 42L);
            
            assertNotNull(dataset);
            assertEquals(100, dataset.getNumSamples()); // 25 per quadrant * 4
            assertEquals(2, dataset.getNumFeatures());
            assertEquals(2, dataset.getNumClasses());
        }
        
        @Test
        @DisplayName("Should generate circles dataset")
        void testMakeCircles() {
            Dataset dataset = DatasetLoader.makeCircles(50, 0.1, 42L);
            
            assertNotNull(dataset);
            assertEquals(100, dataset.getNumSamples()); // 50 per circle * 2
            assertEquals(2, dataset.getNumFeatures());
            assertEquals(2, dataset.getNumClasses());
        }
        
        @Test
        @DisplayName("Synthetic classification should be reproducible with seed")
        void testSyntheticReproducible() {
            Dataset d1 = DatasetLoader.makeClassification(50, 3, 2, 42L);
            Dataset d2 = DatasetLoader.makeClassification(50, 3, 2, 42L);
            
            assertArrayEquals(d1.getFeatures()[0], d2.getFeatures()[0], 0.001);
            assertArrayEquals(d1.getLabels(), d2.getLabels());
        }
    }
    
    @Nested
    @DisplayName("Dataset Splitting Tests")
    class SplittingTests {
        
        @Test
        @DisplayName("Should split loaded dataset")
        void testSplitLoadedDataset() {
            Dataset iris = DatasetLoader.loadIris();
            Dataset[] splits = iris.trainTestSplit(0.3, 42L);
            
            Dataset train = splits[0];
            Dataset test = splits[1];
            
            assertEquals(150, train.getNumSamples() + test.getNumSamples());
            assertEquals(45, test.getNumSamples()); // 30% of 150
        }
    }
    
    @Nested
    @DisplayName("Data Integrity Tests")
    class DataIntegrityTests {
        
        @Test
        @DisplayName("Iris features should not contain NaN")
        void testIrisNoNaN() {
            Dataset iris = DatasetLoader.loadIris();
            double[][] features = iris.getFeatures();
            
            for (double[] row : features) {
                for (double val : row) {
                    assertFalse(Double.isNaN(val), "Features should not contain NaN");
                    assertFalse(Double.isInfinite(val), "Features should not contain Inf");
                }
            }
        }
        
        @Test
        @DisplayName("Wine features should not contain NaN")
        void testWineNoNaN() {
            Dataset wine = DatasetLoader.loadWine();
            double[][] features = wine.getFeatures();
            
            for (double[] row : features) {
                for (double val : row) {
                    assertFalse(Double.isNaN(val), "Features should not contain NaN");
                    assertFalse(Double.isInfinite(val), "Features should not contain Inf");
                }
            }
        }
        
        @Test
        @DisplayName("Labels should be valid class indices")
        void testValidLabels() {
            Dataset iris = DatasetLoader.loadIris();
            int[] labels = iris.getLabels();
            int numClasses = iris.getNumClasses();
            
            for (int label : labels) {
                assertTrue(label >= 0 && label < numClasses, 
                    "Label should be valid class index");
            }
        }
    }
}
