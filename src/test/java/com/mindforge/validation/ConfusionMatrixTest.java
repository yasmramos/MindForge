package com.mindforge.validation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Confusion Matrix Tests")
class ConfusionMatrixTest {
    
    @Nested
    @DisplayName("Binary Classification Tests")
    class BinaryClassificationTests {
        
        @Test
        @DisplayName("Should create confusion matrix for binary classification")
        void testBinaryConfusionMatrix() {
            int[] yTrue = {0, 0, 1, 1, 0, 1, 0, 1};
            int[] yPred = {0, 1, 1, 1, 0, 0, 0, 1};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            int[][] matrix = cm.getMatrix();
            
            assertNotNull(matrix);
            assertEquals(2, matrix.length);
            assertEquals(2, matrix[0].length);
        }
        
        @Test
        @DisplayName("Should calculate accuracy correctly")
        void testAccuracy() {
            int[] yTrue = {0, 0, 1, 1};
            int[] yPred = {0, 0, 1, 1};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            assertEquals(1.0, cm.getAccuracy(), 0.001, "Perfect predictions should have 100% accuracy");
        }
        
        @Test
        @DisplayName("Should calculate precision for a class")
        void testPrecision() {
            int[] yTrue = {0, 0, 1, 1, 1};
            int[] yPred = {0, 1, 1, 1, 0};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            double precision = cm.getPrecision(1);
            
            assertTrue(precision >= 0 && precision <= 1);
            // Class 1: TP=2, FP=1, precision = 2/3
            assertEquals(2.0/3.0, precision, 0.001);
        }
        
        @Test
        @DisplayName("Should calculate recall for a class")
        void testRecall() {
            int[] yTrue = {0, 0, 1, 1, 1};
            int[] yPred = {0, 1, 1, 1, 0};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            double recall = cm.getRecall(1);
            
            assertTrue(recall >= 0 && recall <= 1);
            // Class 1: TP=2, FN=1, recall = 2/3
            assertEquals(2.0/3.0, recall, 0.001);
        }
        
        @Test
        @DisplayName("Should calculate F1 score for a class")
        void testF1Score() {
            int[] yTrue = {0, 0, 1, 1};
            int[] yPred = {0, 0, 1, 1};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            double f1 = cm.getF1Score(1);
            
            assertEquals(1.0, f1, 0.001, "Perfect class 1 should have F1 = 1");
        }
        
        @Test
        @DisplayName("Should calculate true positives correctly")
        void testTruePositives() {
            int[] yTrue = {1, 1, 1, 0, 0};
            int[] yPred = {1, 1, 0, 0, 1};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            assertEquals(2, cm.getTruePositives(1), "Should have 2 true positives for class 1");
        }
        
        @Test
        @DisplayName("Should calculate false positives correctly")
        void testFalsePositives() {
            int[] yTrue = {0, 0, 0, 1, 1};
            int[] yPred = {1, 1, 0, 1, 1};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            assertEquals(2, cm.getFalsePositives(1), "Should have 2 false positives for class 1");
        }
        
        @Test
        @DisplayName("Should calculate false negatives correctly")
        void testFalseNegatives() {
            int[] yTrue = {1, 1, 1, 0, 0};
            int[] yPred = {0, 0, 1, 0, 0};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            assertEquals(2, cm.getFalseNegatives(1), "Should have 2 false negatives for class 1");
        }
        
        @Test
        @DisplayName("Should calculate true negatives correctly")
        void testTrueNegatives() {
            int[] yTrue = {0, 0, 1, 1};
            int[] yPred = {0, 0, 1, 1};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            assertEquals(2, cm.getTrueNegatives(1), "Should have 2 true negatives for class 1");
        }
    }
    
    @Nested
    @DisplayName("Multi-class Classification Tests")
    class MultiClassTests {
        
        @Test
        @DisplayName("Should handle 3-class classification")
        void testThreeClassConfusionMatrix() {
            int[] yTrue = {0, 0, 1, 1, 2, 2};
            int[] yPred = {0, 0, 1, 2, 2, 1};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            int[][] matrix = cm.getMatrix();
            
            assertEquals(3, matrix.length);
            assertEquals(3, matrix[0].length);
            assertEquals(3, cm.getNumClasses());
        }
        
        @Test
        @DisplayName("Should calculate macro-averaged precision")
        void testMacroPrecision() {
            int[] yTrue = {0, 0, 1, 1, 2, 2};
            int[] yPred = {0, 0, 1, 1, 2, 2};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            double macroPrecision = cm.getMacroPrecision();
            
            assertEquals(1.0, macroPrecision, 0.001, "Perfect predictions should have macro precision 1.0");
        }
        
        @Test
        @DisplayName("Should calculate macro-averaged recall")
        void testMacroRecall() {
            int[] yTrue = {0, 0, 1, 1, 2, 2};
            int[] yPred = {0, 0, 1, 1, 2, 2};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            double macroRecall = cm.getMacroRecall();
            
            assertEquals(1.0, macroRecall, 0.001, "Perfect predictions should have macro recall 1.0");
        }
        
        @Test
        @DisplayName("Should calculate macro-averaged F1")
        void testMacroF1Score() {
            int[] yTrue = {0, 0, 1, 1, 2, 2};
            int[] yPred = {0, 0, 1, 1, 2, 2};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            double macroF1 = cm.getMacroF1Score();
            
            assertEquals(1.0, macroF1, 0.001, "Perfect predictions should have macro F1 1.0");
        }
        
        @Test
        @DisplayName("Should calculate weighted F1 score")
        void testWeightedF1Score() {
            int[] yTrue = {0, 0, 1, 1, 2, 2};
            int[] yPred = {0, 0, 1, 1, 2, 2};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            double weightedF1 = cm.getWeightedF1Score();
            
            assertEquals(1.0, weightedF1, 0.001, "Perfect predictions should have weighted F1 1.0");
        }
    }
    
    @Nested
    @DisplayName("String Representation Tests")
    class StringRepresentationTests {
        
        @Test
        @DisplayName("Should generate string representation")
        void testToString() {
            int[] yTrue = {0, 1, 0, 1};
            int[] yPred = {0, 1, 0, 1};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            String str = cm.toString();
            
            assertNotNull(str);
            assertFalse(str.isEmpty());
        }
        
        @Test
        @DisplayName("Should generate classification report")
        void testClassificationReport() {
            int[] yTrue = {0, 0, 1, 1, 2, 2};
            int[] yPred = {0, 0, 1, 1, 2, 2};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            String report = cm.classificationReport();
            
            assertNotNull(report);
            assertTrue(report.contains("precision") || report.contains("Precision"));
            assertTrue(report.contains("recall") || report.contains("Recall"));
        }
        
        @Test
        @DisplayName("Should get and set class labels")
        void testClassLabels() {
            int[] yTrue = {0, 0, 1, 1};
            int[] yPred = {0, 0, 1, 1};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            cm.setClassLabels(new String[]{"Negative", "Positive"});
            
            String[] labels = cm.getClassLabels();
            assertEquals("Negative", labels[0]);
            assertEquals("Positive", labels[1]);
        }
    }
    
    @Nested
    @DisplayName("Constructor with Labels")
    class ConstructorWithLabelsTests {
        
        @Test
        @DisplayName("Should create confusion matrix with custom labels")
        void testWithLabels() {
            int[] yTrue = {0, 0, 1, 1};
            int[] yPred = {0, 0, 1, 1};
            String[] labels = {"Cat", "Dog"};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred, labels);
            
            String[] storedLabels = cm.getClassLabels();
            assertEquals("Cat", storedLabels[0]);
            assertEquals("Dog", storedLabels[1]);
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle all correct predictions")
        void testAllCorrect() {
            int[] yTrue = {0, 0, 0};
            int[] yPred = {0, 0, 0};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            assertEquals(1.0, cm.getAccuracy(), 0.001);
        }
        
        @Test
        @DisplayName("Should handle all wrong predictions")
        void testAllWrong() {
            int[] yTrue = {0, 0, 0};
            int[] yPred = {1, 1, 1};
            
            ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
            assertEquals(0.0, cm.getAccuracy(), 0.001);
        }
        
        @Test
        @DisplayName("Should throw exception for mismatched lengths")
        void testMismatchedLengths() {
            int[] yTrue = {0, 1, 2};
            int[] yPred = {0, 1};
            
            assertThrows(IllegalArgumentException.class, () -> {
                new ConfusionMatrix(yTrue, yPred);
            });
        }
    }
}
