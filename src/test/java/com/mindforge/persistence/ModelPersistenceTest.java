package com.mindforge.persistence;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.io.TempDir;

import java.io.*;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for ModelPersistence utility.
 */
@DisplayName("ModelPersistence Tests")
class ModelPersistenceTest {
    
    @TempDir
    Path tempDir;
    
    private String testFilePath;
    
    @BeforeEach
    void setUp() {
        testFilePath = tempDir.resolve("test_model.bin").toString();
    }
    
    /**
     * Simple serializable model for testing.
     */
    static class TestModel implements Serializable {
        private static final long serialVersionUID = 1L;
        
        private final String name;
        private final int[] parameters;
        private final double[][] weights;
        private boolean trained;
        
        public TestModel(String name, int[] parameters) {
            this.name = name;
            this.parameters = parameters;
            this.weights = new double[3][3];
            this.trained = false;
        }
        
        public void train() {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    weights[i][j] = (i + 1) * (j + 1) * 0.1;
                }
            }
            trained = true;
        }
        
        public String getName() { return name; }
        public int[] getParameters() { return parameters; }
        public double[][] getWeights() { return weights; }
        public boolean isTrained() { return trained; }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            TestModel testModel = (TestModel) o;
            return trained == testModel.trained &&
                   name.equals(testModel.name) &&
                   java.util.Arrays.equals(parameters, testModel.parameters);
        }
    }
    
    @Nested
    @DisplayName("Save Tests")
    class SaveTests {
        
        @Test
        @DisplayName("Should save model to file")
        void testSaveModel() {
            TestModel model = new TestModel("test", new int[]{1, 2, 3});
            model.train();
            
            ModelPersistence.save(model, testFilePath);
            
            File file = new File(testFilePath);
            assertTrue(file.exists());
            assertTrue(file.length() > 0);
        }
        
        @Test
        @DisplayName("Should reject null model")
        void testSaveNullModel() {
            assertThrows(IllegalArgumentException.class, () -> 
                ModelPersistence.save(null, testFilePath));
        }
        
        @Test
        @DisplayName("Should reject null path")
        void testSaveNullPath() {
            TestModel model = new TestModel("test", new int[]{1});
            assertThrows(IllegalArgumentException.class, () -> 
                ModelPersistence.save(model, null));
        }
        
        @Test
        @DisplayName("Should reject empty path")
        void testSaveEmptyPath() {
            TestModel model = new TestModel("test", new int[]{1});
            assertThrows(IllegalArgumentException.class, () -> 
                ModelPersistence.save(model, ""));
        }
        
        @Test
        @DisplayName("Should throw on invalid path")
        void testSaveInvalidPath() {
            TestModel model = new TestModel("test", new int[]{1});
            assertThrows(ModelPersistenceException.class, () -> 
                ModelPersistence.save(model, "/nonexistent/directory/model.bin"));
        }
    }
    
    @Nested
    @DisplayName("Load Tests")
    class LoadTests {
        
        @Test
        @DisplayName("Should load saved model")
        void testLoadModel() {
            TestModel original = new TestModel("myModel", new int[]{10, 20, 30});
            original.train();
            
            ModelPersistence.save(original, testFilePath);
            TestModel loaded = ModelPersistence.load(testFilePath);
            
            assertNotNull(loaded);
            assertEquals(original.getName(), loaded.getName());
            assertArrayEquals(original.getParameters(), loaded.getParameters());
            assertTrue(loaded.isTrained());
        }
        
        @Test
        @DisplayName("Should preserve model state after save/load")
        void testPreservesState() {
            TestModel original = new TestModel("trained", new int[]{5});
            original.train();
            
            ModelPersistence.save(original, testFilePath);
            TestModel loaded = ModelPersistence.load(testFilePath);
            
            double[][] originalWeights = original.getWeights();
            double[][] loadedWeights = loaded.getWeights();
            
            for (int i = 0; i < originalWeights.length; i++) {
                assertArrayEquals(originalWeights[i], loadedWeights[i], 1e-10);
            }
        }
        
        @Test
        @DisplayName("Should load with type checking")
        void testLoadWithType() {
            TestModel original = new TestModel("typed", new int[]{1});
            ModelPersistence.save(original, testFilePath);
            
            TestModel loaded = ModelPersistence.load(testFilePath, TestModel.class);
            assertNotNull(loaded);
            assertEquals("typed", loaded.getName());
        }
        
        @Test
        @DisplayName("Should throw on type mismatch")
        void testLoadTypeMismatch() {
            TestModel original = new TestModel("test", new int[]{1});
            ModelPersistence.save(original, testFilePath);
            
            assertThrows(ModelPersistenceException.class, () -> 
                ModelPersistence.load(testFilePath, String.class));
        }
        
        @Test
        @DisplayName("Should reject null path")
        void testLoadNullPath() {
            assertThrows(IllegalArgumentException.class, () -> 
                ModelPersistence.load(null));
        }
        
        @Test
        @DisplayName("Should throw on nonexistent file")
        void testLoadNonexistentFile() {
            assertThrows(ModelPersistenceException.class, () -> 
                ModelPersistence.load("/nonexistent/model.bin"));
        }
        
        @Test
        @DisplayName("Should throw on invalid format")
        void testLoadInvalidFormat() throws IOException {
            // Create a file with invalid format
            File invalidFile = tempDir.resolve("invalid.bin").toFile();
            try (FileOutputStream fos = new FileOutputStream(invalidFile)) {
                fos.write("not a valid model".getBytes());
            }
            
            assertThrows(ModelPersistenceException.class, () -> 
                ModelPersistence.load(invalidFile.getAbsolutePath()));
        }
    }
    
    @Nested
    @DisplayName("Metadata Tests")
    class MetadataTests {
        
        @Test
        @DisplayName("Should get model metadata")
        void testGetMetadata() {
            TestModel model = new TestModel("metaTest", new int[]{1, 2});
            ModelPersistence.save(model, testFilePath);
            
            ModelPersistence.ModelMetadata metadata = ModelPersistence.getMetadata(testFilePath);
            
            assertNotNull(metadata);
            assertTrue(metadata.getClassName().contains("TestModel"));
            assertTrue(metadata.getSimpleClassName().contains("TestModel"));
            assertTrue(metadata.getSavedTimestamp() > 0);
            assertTrue(metadata.getFileSize() > 0);
        }
        
        @Test
        @DisplayName("Should throw on nonexistent file")
        void testMetadataNonexistent() {
            assertThrows(ModelPersistenceException.class, () -> 
                ModelPersistence.getMetadata("/nonexistent/model.bin"));
        }
        
        @Test
        @DisplayName("Should have informative toString")
        void testMetadataToString() {
            TestModel model = new TestModel("test", new int[]{1});
            ModelPersistence.save(model, testFilePath);
            
            ModelPersistence.ModelMetadata metadata = ModelPersistence.getMetadata(testFilePath);
            String str = metadata.toString();
            
            assertTrue(str.contains("TestModel"));
            assertTrue(str.contains("bytes"));
        }
    }
    
    @Nested
    @DisplayName("Validation Tests")
    class ValidationTests {
        
        @Test
        @DisplayName("Should validate correct model file")
        void testIsValidModelFile() {
            TestModel model = new TestModel("valid", new int[]{1});
            ModelPersistence.save(model, testFilePath);
            
            assertTrue(ModelPersistence.isValidModelFile(testFilePath));
        }
        
        @Test
        @DisplayName("Should reject invalid file")
        void testIsInvalidModelFile() throws IOException {
            File invalidFile = tempDir.resolve("invalid.bin").toFile();
            try (FileOutputStream fos = new FileOutputStream(invalidFile)) {
                fos.write("invalid content".getBytes());
            }
            
            assertFalse(ModelPersistence.isValidModelFile(invalidFile.getAbsolutePath()));
        }
        
        @Test
        @DisplayName("Should return false for nonexistent file")
        void testValidationNonexistent() {
            assertFalse(ModelPersistence.isValidModelFile("/nonexistent/model.bin"));
        }
        
        @Test
        @DisplayName("Should return false for null path")
        void testValidationNullPath() {
            assertFalse(ModelPersistence.isValidModelFile(null));
        }
        
        @Test
        @DisplayName("Should return false for empty path")
        void testValidationEmptyPath() {
            assertFalse(ModelPersistence.isValidModelFile(""));
        }
    }
    
    @Nested
    @DisplayName("Byte Array Tests")
    class ByteArrayTests {
        
        @Test
        @DisplayName("Should serialize to bytes")
        void testToBytes() {
            TestModel model = new TestModel("bytes", new int[]{1, 2, 3});
            model.train();
            
            byte[] bytes = ModelPersistence.toBytes(model);
            
            assertNotNull(bytes);
            assertTrue(bytes.length > 0);
        }
        
        @Test
        @DisplayName("Should deserialize from bytes")
        void testFromBytes() {
            TestModel original = new TestModel("fromBytes", new int[]{5, 10});
            original.train();
            
            byte[] bytes = ModelPersistence.toBytes(original);
            TestModel loaded = ModelPersistence.fromBytes(bytes);
            
            assertNotNull(loaded);
            assertEquals(original.getName(), loaded.getName());
            assertArrayEquals(original.getParameters(), loaded.getParameters());
            assertTrue(loaded.isTrained());
        }
        
        @Test
        @DisplayName("Should reject null for toBytes")
        void testToBytesNull() {
            assertThrows(IllegalArgumentException.class, () -> 
                ModelPersistence.toBytes(null));
        }
        
        @Test
        @DisplayName("Should reject null for fromBytes")
        void testFromBytesNull() {
            assertThrows(IllegalArgumentException.class, () -> 
                ModelPersistence.fromBytes(null));
        }
        
        @Test
        @DisplayName("Should reject empty bytes")
        void testFromBytesEmpty() {
            assertThrows(IllegalArgumentException.class, () -> 
                ModelPersistence.fromBytes(new byte[0]));
        }
        
        @Test
        @DisplayName("Should throw on invalid bytes")
        void testFromBytesInvalid() {
            byte[] invalid = "not a model".getBytes();
            assertThrows(ModelPersistenceException.class, () -> 
                ModelPersistence.fromBytes(invalid));
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle large model")
        void testLargeModel() {
            // Create model with large arrays
            int[] bigParams = new int[1000];
            for (int i = 0; i < bigParams.length; i++) {
                bigParams[i] = i;
            }
            
            TestModel model = new TestModel("large", bigParams);
            model.train();
            
            ModelPersistence.save(model, testFilePath);
            TestModel loaded = ModelPersistence.load(testFilePath);
            
            assertArrayEquals(bigParams, loaded.getParameters());
        }
        
        @Test
        @DisplayName("Should handle special characters in model data")
        void testSpecialCharacters() {
            TestModel model = new TestModel("测试模型 Test & <Model>", new int[]{1});
            
            ModelPersistence.save(model, testFilePath);
            TestModel loaded = ModelPersistence.load(testFilePath);
            
            assertEquals("测试模型 Test & <Model>", loaded.getName());
        }
        
        @Test
        @DisplayName("Should overwrite existing file")
        void testOverwrite() {
            TestModel model1 = new TestModel("first", new int[]{1});
            TestModel model2 = new TestModel("second", new int[]{2});
            
            ModelPersistence.save(model1, testFilePath);
            ModelPersistence.save(model2, testFilePath);
            
            TestModel loaded = ModelPersistence.load(testFilePath);
            assertEquals("second", loaded.getName());
        }
        
        @Test
        @DisplayName("Should handle untrained model")
        void testUntrainedModel() {
            TestModel model = new TestModel("untrained", new int[]{1});
            // Don't call train()
            
            ModelPersistence.save(model, testFilePath);
            TestModel loaded = ModelPersistence.load(testFilePath);
            
            assertFalse(loaded.isTrained());
        }
    }
    
    @Nested
    @DisplayName("Round-Trip Tests")
    class RoundTripTests {
        
        @Test
        @DisplayName("Should maintain equality through file round-trip")
        void testFileRoundTrip() {
            TestModel original = new TestModel("roundTrip", new int[]{100, 200, 300});
            original.train();
            
            ModelPersistence.save(original, testFilePath);
            TestModel loaded = ModelPersistence.load(testFilePath);
            
            assertEquals(original, loaded);
        }
        
        @Test
        @DisplayName("Should maintain equality through bytes round-trip")
        void testBytesRoundTrip() {
            TestModel original = new TestModel("bytesRoundTrip", new int[]{1, 2, 3, 4, 5});
            original.train();
            
            byte[] bytes = ModelPersistence.toBytes(original);
            TestModel loaded = ModelPersistence.fromBytes(bytes);
            
            assertEquals(original, loaded);
        }
        
        @Test
        @DisplayName("Should handle multiple save/load cycles")
        void testMultipleCycles() {
            TestModel model = new TestModel("cycles", new int[]{42});
            model.train();
            
            for (int i = 0; i < 5; i++) {
                ModelPersistence.save(model, testFilePath);
                model = ModelPersistence.load(testFilePath);
            }
            
            assertEquals("cycles", model.getName());
            assertArrayEquals(new int[]{42}, model.getParameters());
            assertTrue(model.isTrained());
        }
    }
    
    @Nested
    @DisplayName("Exception Tests")
    class ExceptionTests {
        
        @Test
        @DisplayName("ModelPersistenceException should have message")
        void testExceptionMessage() {
            ModelPersistenceException ex = new ModelPersistenceException("test error");
            assertEquals("test error", ex.getMessage());
        }
        
        @Test
        @DisplayName("ModelPersistenceException should have cause")
        void testExceptionCause() {
            IOException cause = new IOException("io error");
            ModelPersistenceException ex = new ModelPersistenceException("wrapper", cause);
            
            assertEquals("wrapper", ex.getMessage());
            assertEquals(cause, ex.getCause());
        }
    }
}
