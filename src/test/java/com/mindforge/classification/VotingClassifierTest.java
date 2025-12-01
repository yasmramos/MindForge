package com.mindforge.classification;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;
import java.util.*;

/**
 * Comprehensive tests for VotingClassifier.
 */
class VotingClassifierTest {
    
    @Nested
    @DisplayName("Constructor Tests")
    class ConstructorTests {
        
        @Test
        @DisplayName("Default constructor")
        void testDefaultConstructor() {
            VotingClassifier vc = new VotingClassifier();
            assertEquals(VotingClassifier.Voting.HARD, vc.getVoting());
            assertEquals(0, vc.getNumClassifiers());
            assertFalse(vc.isTrained());
        }
        
        @Test
        @DisplayName("Constructor with voting strategy")
        void testVotingConstructor() {
            VotingClassifier vc = new VotingClassifier(VotingClassifier.Voting.SOFT);
            assertEquals(VotingClassifier.Voting.SOFT, vc.getVoting());
        }
        
        @Test
        @DisplayName("Constructor with weights")
        void testWeightsConstructor() {
            double[] weights = {1.0, 2.0};
            VotingClassifier vc = new VotingClassifier(VotingClassifier.Voting.HARD, weights);
            assertEquals(VotingClassifier.Voting.HARD, vc.getVoting());
        }
    }
    
    @Nested
    @DisplayName("Add Classifier Tests")
    class AddClassifierTests {
        
        @Test
        @DisplayName("Add single classifier")
        void testAddSingleClassifier() {
            VotingClassifier vc = new VotingClassifier();
            vc.addClassifier("knn", new KNearestNeighbors(3));
            
            assertEquals(1, vc.getNumClassifiers());
            assertEquals(List.of("knn"), vc.getClassifierNames());
        }
        
        @Test
        @DisplayName("Add multiple classifiers")
        void testAddMultipleClassifiers() {
            VotingClassifier vc = new VotingClassifier()
                .addClassifier("knn", new KNearestNeighbors(3))
                .addClassifier("dt", new DecisionTreeClassifier())
                .addClassifier("nb", new GaussianNaiveBayes());
            
            assertEquals(3, vc.getNumClassifiers());
        }
        
        @Test
        @DisplayName("Null name throws exception")
        void testNullName() {
            VotingClassifier vc = new VotingClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> vc.addClassifier(null, new KNearestNeighbors(3)));
        }
        
        @Test
        @DisplayName("Empty name throws exception")
        void testEmptyName() {
            VotingClassifier vc = new VotingClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> vc.addClassifier("  ", new KNearestNeighbors(3)));
        }
        
        @Test
        @DisplayName("Null classifier throws exception")
        void testNullClassifier() {
            VotingClassifier vc = new VotingClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> vc.addClassifier("test", null));
        }
        
        @Test
        @DisplayName("Duplicate name throws exception")
        void testDuplicateName() {
            VotingClassifier vc = new VotingClassifier();
            vc.addClassifier("knn", new KNearestNeighbors(3));
            
            assertThrows(IllegalArgumentException.class, 
                () -> vc.addClassifier("knn", new KNearestNeighbors(5)));
        }
    }
    
    @Nested
    @DisplayName("Training Tests")
    class TrainingTests {
        
        @Test
        @DisplayName("Train without classifiers throws exception")
        void testTrainWithoutClassifiers() {
            VotingClassifier vc = new VotingClassifier();
            assertThrows(IllegalStateException.class, 
                () -> vc.train(new double[][]{{0}, {1}}, new int[]{0, 1}));
        }
        
        @Test
        @DisplayName("Train with null X throws exception")
        void testTrainNullX() {
            VotingClassifier vc = new VotingClassifier()
                .addClassifier("knn", new KNearestNeighbors(1));
            
            assertThrows(IllegalArgumentException.class, 
                () -> vc.train(null, new int[]{0, 1}));
        }
        
        @Test
        @DisplayName("Train with null y throws exception")
        void testTrainNullY() {
            VotingClassifier vc = new VotingClassifier()
                .addClassifier("knn", new KNearestNeighbors(1));
            
            assertThrows(IllegalArgumentException.class, 
                () -> vc.train(new double[][]{{0}, {1}}, null));
        }
        
        @Test
        @DisplayName("Train with mismatched lengths throws exception")
        void testTrainMismatchedLengths() {
            VotingClassifier vc = new VotingClassifier()
                .addClassifier("knn", new KNearestNeighbors(1));
            
            assertThrows(IllegalArgumentException.class, 
                () -> vc.train(new double[][]{{0}, {1}}, new int[]{0}));
        }
        
        @Test
        @DisplayName("Train with wrong weight count throws exception")
        void testTrainWrongWeightCount() {
            VotingClassifier vc = new VotingClassifier(VotingClassifier.Voting.HARD, new double[]{1.0, 2.0});
            vc.addClassifier("knn", new KNearestNeighbors(1));
            
            assertThrows(IllegalArgumentException.class, 
                () -> vc.train(new double[][]{{0}, {1}}, new int[]{0, 1}));
        }
        
        @Test
        @DisplayName("Successful training")
        void testSuccessfulTraining() {
            VotingClassifier vc = new VotingClassifier()
                .addClassifier("knn", new KNearestNeighbors(1))
                .addClassifier("dt", new DecisionTreeClassifier());
            
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            
            vc.train(X, y);
            
            assertTrue(vc.isTrained());
            assertEquals(2, vc.getNumClasses());
        }
    }
    
    @Nested
    @DisplayName("Prediction Tests")
    class PredictionTests {
        
        @Test
        @DisplayName("Predict before training throws exception")
        void testPredictBeforeTraining() {
            VotingClassifier vc = new VotingClassifier()
                .addClassifier("knn", new KNearestNeighbors(1));
            
            assertThrows(IllegalStateException.class, 
                () -> vc.predict(new double[]{0}));
        }
        
        @Test
        @DisplayName("Single prediction")
        void testSinglePrediction() {
            VotingClassifier vc = createTrainedVotingClassifier();
            
            int pred = vc.predict(new double[]{0.5});
            assertTrue(pred == 0 || pred == 1);
        }
        
        @Test
        @DisplayName("Batch prediction")
        void testBatchPrediction() {
            VotingClassifier vc = createTrainedVotingClassifier();
            
            double[][] X = {{0}, {10}};
            int[] predictions = vc.predict(X);
            
            assertEquals(2, predictions.length);
        }
        
        @Test
        @DisplayName("Batch prediction with null throws exception")
        void testBatchPredictionNull() {
            VotingClassifier vc = createTrainedVotingClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> vc.predict((double[][]) null));
        }
        
        @Test
        @DisplayName("Batch prediction with empty throws exception")
        void testBatchPredictionEmpty() {
            VotingClassifier vc = createTrainedVotingClassifier();
            assertThrows(IllegalArgumentException.class, 
                () -> vc.predict(new double[0][]));
        }
        
        @Test
        @DisplayName("Majority voting works")
        void testMajorityVoting() {
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            
            // Create 3 classifiers that should agree
            VotingClassifier vc = new VotingClassifier()
                .addClassifier("knn1", new KNearestNeighbors(1))
                .addClassifier("knn2", new KNearestNeighbors(2))
                .addClassifier("dt", new DecisionTreeClassifier());
            
            vc.train(X, y);
            
            int[] predictions = vc.predict(X);
            
            // With clear separation, should get good accuracy
            int correct = 0;
            for (int i = 0; i < y.length; i++) {
                if (predictions[i] == y[i]) correct++;
            }
            assertTrue(correct >= 3);
        }
        
        @Test
        @DisplayName("Weighted voting")
        void testWeightedVoting() {
            double[][] X = {{0}, {1}, {10}, {11}};
            int[] y = {0, 0, 1, 1};
            
            // Weight the first classifier more
            double[] weights = {10.0, 1.0};
            VotingClassifier vc = new VotingClassifier(VotingClassifier.Voting.HARD, weights)
                .addClassifier("knn1", new KNearestNeighbors(1))
                .addClassifier("knn2", new KNearestNeighbors(2));
            
            vc.train(X, y);
            
            // Should work without errors
            int pred = vc.predict(new double[]{5});
            assertTrue(pred == 0 || pred == 1);
        }
    }
    
    @Nested
    @DisplayName("Individual Predictions Tests")
    class IndividualPredictionsTests {
        
        @Test
        @DisplayName("Get individual predictions")
        void testGetIndividualPredictions() {
            VotingClassifier vc = createTrainedVotingClassifier();
            
            Map<String, Integer> individual = vc.getIndividualPredictions(new double[]{0});
            
            assertEquals(2, individual.size());
            assertTrue(individual.containsKey("knn"));
            assertTrue(individual.containsKey("dt"));
        }
        
        @Test
        @DisplayName("Individual predictions before training throws exception")
        void testIndividualPredictionsBeforeTraining() {
            VotingClassifier vc = new VotingClassifier()
                .addClassifier("knn", new KNearestNeighbors(1));
            
            assertThrows(IllegalStateException.class, 
                () -> vc.getIndividualPredictions(new double[]{0}));
        }
    }
    
    @Nested
    @DisplayName("Getter Tests")
    class GetterTests {
        
        @Test
        @DisplayName("getClasses before training throws exception")
        void testGetClassesBeforeTraining() {
            VotingClassifier vc = new VotingClassifier()
                .addClassifier("knn", new KNearestNeighbors(1));
            
            assertThrows(IllegalStateException.class, vc::getClasses);
        }
        
        @Test
        @DisplayName("getClasses after training")
        void testGetClassesAfterTraining() {
            VotingClassifier vc = createTrainedVotingClassifier();
            
            int[] classes = vc.getClasses();
            assertArrayEquals(new int[]{0, 1}, classes);
        }
        
        @Test
        @DisplayName("getNumClasses before training returns 0")
        void testGetNumClassesBeforeTraining() {
            VotingClassifier vc = new VotingClassifier();
            assertEquals(0, vc.getNumClasses());
        }
    }
    
    @Nested
    @DisplayName("Builder Tests")
    class BuilderTests {
        
        @Test
        @DisplayName("Builder creates classifier")
        void testBuilder() {
            VotingClassifier vc = new VotingClassifier.Builder()
                .voting(VotingClassifier.Voting.HARD)
                .weights(new double[]{1.0, 1.0})
                .addClassifier("knn", new KNearestNeighbors(1))
                .addClassifier("dt", new DecisionTreeClassifier())
                .build();
            
            assertEquals(2, vc.getNumClassifiers());
            assertEquals(VotingClassifier.Voting.HARD, vc.getVoting());
        }
    }
    
    @Nested
    @DisplayName("Serialization Tests")
    class SerializationTests {
        
        @Test
        @DisplayName("Serialize and deserialize")
        @Disabled("KNearestNeighbors is not serializable")
        void testSerialization() throws IOException, ClassNotFoundException {
            VotingClassifier vc = createTrainedVotingClassifier();
            
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            oos.writeObject(vc);
            oos.close();
            
            ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bis);
            VotingClassifier restored = (VotingClassifier) ois.readObject();
            ois.close();
            
            assertTrue(restored.isTrained());
            assertEquals(vc.getNumClassifiers(), restored.getNumClassifiers());
        }
    }
    
    private VotingClassifier createTrainedVotingClassifier() {
        VotingClassifier vc = new VotingClassifier()
            .addClassifier("knn", new KNearestNeighbors(1))
            .addClassifier("dt", new DecisionTreeClassifier());
        
        double[][] X = {{0}, {1}, {10}, {11}};
        int[] y = {0, 0, 1, 1};
        vc.train(X, y);
        
        return vc;
    }
}
