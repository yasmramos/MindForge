package io.github.yasmramos.mindforge.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Dataset Tests")
class DatasetTest {
    
    private double[][] features;
    private int[] labels;
    private double[] targets;
    
    @BeforeEach
    void setUp() {
        features = new double[][] {
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0},
            {7.0, 8.0},
            {9.0, 10.0}
        };
        labels = new int[] {0, 1, 0, 1, 0};
        targets = new double[] {1.5, 2.5, 3.5, 4.5, 5.5};
    }
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Should create classification dataset with features and labels")
        void testCreateClassificationDataset() {
            Dataset dataset = new Dataset(features, labels);
            
            assertNotNull(dataset);
            assertEquals(5, dataset.getNumSamples());
            assertEquals(2, dataset.getNumFeatures());
            assertTrue(dataset.isClassification());
        }
        
        @Test
        @DisplayName("Should create regression dataset with features and targets")
        void testCreateRegressionDataset() {
            Dataset dataset = new Dataset(features, targets);
            
            assertNotNull(dataset);
            assertEquals(5, dataset.getNumSamples());
            assertEquals(2, dataset.getNumFeatures());
            assertFalse(dataset.isClassification());
        }
        
        @Test
        @DisplayName("Should store features correctly")
        void testFeatures() {
            Dataset dataset = new Dataset(features, labels);
            double[][] storedFeatures = dataset.getFeatures();
            
            assertArrayEquals(features[0], storedFeatures[0], 0.001);
            assertArrayEquals(features[4], storedFeatures[4], 0.001);
        }
        
        @Test
        @DisplayName("Should store labels correctly")
        void testLabels() {
            Dataset dataset = new Dataset(features, labels);
            int[] storedLabels = dataset.getLabels();
            
            assertArrayEquals(labels, storedLabels);
        }
        
        @Test
        @DisplayName("Should store targets correctly")
        void testTargets() {
            Dataset dataset = new Dataset(features, targets);
            double[] storedTargets = dataset.getTargets();
            
            assertArrayEquals(targets, storedTargets, 0.001);
        }
        
        @Test
        @DisplayName("Should count number of classes")
        void testNumClasses() {
            Dataset dataset = new Dataset(features, labels);
            
            assertEquals(2, dataset.getNumClasses());
        }
    }
    
    @Nested
    @DisplayName("Metadata Tests")
    class MetadataTests {
        
        @Test
        @DisplayName("Should set and get name")
        void testName() {
            Dataset dataset = new Dataset(features, labels);
            dataset.setName("TestDataset");
            
            assertEquals("TestDataset", dataset.getName());
        }
        
        @Test
        @DisplayName("Should set and get description")
        void testDescription() {
            Dataset dataset = new Dataset(features, labels);
            dataset.setDescription("A test dataset");
            
            assertEquals("A test dataset", dataset.getDescription());
        }
        
        @Test
        @DisplayName("Should set and get feature names")
        void testFeatureNames() {
            Dataset dataset = new Dataset(features, labels);
            String[] featureNames = {"Feature1", "Feature2"};
            dataset.setFeatureNames(featureNames);
            
            assertArrayEquals(featureNames, dataset.getFeatureNames());
        }
        
        @Test
        @DisplayName("Should set and get target names")
        void testTargetNames() {
            Dataset dataset = new Dataset(features, labels);
            String[] targetNames = {"Class0", "Class1"};
            dataset.setTargetNames(targetNames);
            
            assertArrayEquals(targetNames, dataset.getTargetNames());
        }
    }
    
    @Nested
    @DisplayName("Train-Test Split Tests")
    class TrainTestSplitTests {
        
        @Test
        @DisplayName("Should split dataset correctly")
        void testTrainTestSplit() {
            Dataset dataset = new Dataset(features, labels);
            Dataset[] splits = dataset.trainTestSplit(0.4, 42L);
            
            assertEquals(2, splits.length);
            assertNotNull(splits[0]); // Train
            assertNotNull(splits[1]); // Test
        }
        
        @Test
        @DisplayName("Should maintain correct proportions")
        void testSplitProportions() {
            Dataset dataset = new Dataset(features, labels);
            Dataset[] splits = dataset.trainTestSplit(0.4, 42L);
            
            Dataset train = splits[0];
            Dataset test = splits[1];
            
            assertEquals(5, train.getNumSamples() + test.getNumSamples());
            assertEquals(2, test.getNumSamples()); // 40% of 5 = 2
        }
        
        @Test
        @DisplayName("Should be reproducible with same seed")
        void testSplitReproducible() {
            Dataset dataset = new Dataset(features, labels);
            Dataset[] split1 = dataset.trainTestSplit(0.4, 42L);
            Dataset[] split2 = dataset.trainTestSplit(0.4, 42L);
            
            // Same seed should produce same split
            assertArrayEquals(split1[0].getLabels(), split2[0].getLabels());
        }
        
        @Test
        @DisplayName("Should split regression dataset")
        void testSplitRegressionDataset() {
            Dataset dataset = new Dataset(features, targets);
            Dataset[] splits = dataset.trainTestSplit(0.4, 42L);
            
            Dataset train = splits[0];
            Dataset test = splits[1];
            
            assertEquals(3, train.getNumSamples());
            assertEquals(2, test.getNumSamples());
            assertFalse(train.isClassification());
            assertFalse(test.isClassification());
        }
    }
    
    @Nested
    @DisplayName("Subset Tests")
    class SubsetTests {
        
        @Test
        @DisplayName("Should get subset by range")
        void testSubset() {
            Dataset dataset = new Dataset(features, labels);
            Dataset subset = dataset.subset(1, 4);
            
            assertEquals(3, subset.getNumSamples());
            assertEquals(2, subset.getNumFeatures());
        }
        
        @Test
        @DisplayName("Subset should maintain feature names")
        void testSubsetPreservesMetadata() {
            Dataset dataset = new Dataset(features, labels);
            String[] featureNames = {"F1", "F2"};
            dataset.setFeatureNames(featureNames);
            
            Dataset subset = dataset.subset(0, 2);
            
            assertArrayEquals(featureNames, subset.getFeatureNames());
        }
    }
    
    @Nested
    @DisplayName("String Representation Tests")
    class StringRepresentationTests {
        
        @Test
        @DisplayName("Should generate toString")
        void testToString() {
            Dataset dataset = new Dataset(features, labels);
            dataset.setName("TestDataset");
            
            String str = dataset.toString();
            
            assertNotNull(str);
            assertTrue(str.contains("5 samples"));
            assertTrue(str.contains("2 features"));
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle single sample")
        void testSingleSample() {
            double[][] singleFeature = {{1.0, 2.0}};
            int[] singleLabel = {0};
            
            Dataset dataset = new Dataset(singleFeature, singleLabel);
            assertEquals(1, dataset.getNumSamples());
        }
        
        @Test
        @DisplayName("Should handle single feature")
        void testSingleFeature() {
            double[][] singleCol = {{1.0}, {2.0}, {3.0}};
            int[] label = {0, 1, 0};
            
            Dataset dataset = new Dataset(singleCol, label);
            assertEquals(1, dataset.getNumFeatures());
        }
        
        @Test
        @DisplayName("Should return 0 classes for regression dataset")
        void testRegressionNoClasses() {
            Dataset dataset = new Dataset(features, targets);
            assertEquals(0, dataset.getNumClasses());
        }
    }
}
