package com.mindforge.persistence;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;
import java.nio.file.*;

/**
 * Additional coverage tests for ModelPersistence.
 */
class ModelPersistenceCoverageTest {
    
    private Path tempDir;
    
    @BeforeEach
    void setUp() throws IOException {
        tempDir = Files.createTempDirectory("model_test");
    }
    
    @AfterEach
    void tearDown() throws IOException {
        // Clean up temp files
        Files.walk(tempDir)
            .sorted((a, b) -> -a.compareTo(b))
            .forEach(path -> {
                try {
                    Files.deleteIfExists(path);
                } catch (IOException e) {
                    // Ignore
                }
            });
    }
    
    @Nested
    @DisplayName("Save Tests")
    class SaveTests {
        
        @Test
        @DisplayName("Save null model throws exception")
        void testSaveNullModel() {
            assertThrows(IllegalArgumentException.class, 
                () -> ModelPersistence.save(null, "test.bin"));
        }
        
        @Test
        @DisplayName("Save with null path throws exception")
        void testSaveNullPath() {
            assertThrows(IllegalArgumentException.class, 
                () -> ModelPersistence.save("test", null));
        }
        
        @Test
        @DisplayName("Save with empty path throws exception")
        void testSaveEmptyPath() {
            assertThrows(IllegalArgumentException.class, 
                () -> ModelPersistence.save("test", "  "));
        }
        
        @Test
        @DisplayName("Save to invalid path throws ModelPersistenceException")
        void testSaveInvalidPath() {
            assertThrows(ModelPersistenceException.class, 
                () -> ModelPersistence.save("test", "/invalid/path/that/does/not/exist/model.bin"));
        }
    }
    
    @Nested
    @DisplayName("Load Tests")
    class LoadTests {
        
        @Test
        @DisplayName("Load with null path throws exception")
        void testLoadNullPath() {
            assertThrows(IllegalArgumentException.class, 
                () -> ModelPersistence.load(null));
        }
        
        @Test
        @DisplayName("Load with empty path throws exception")
        void testLoadEmptyPath() {
            assertThrows(IllegalArgumentException.class, 
                () -> ModelPersistence.load("   "));
        }
        
        @Test
        @DisplayName("Load non-existent file throws ModelPersistenceException")
        void testLoadNonExistentFile() {
            assertThrows(ModelPersistenceException.class, 
                () -> ModelPersistence.load("/non/existent/file.bin"));
        }
        
        @Test
        @DisplayName("Load invalid model file throws ModelPersistenceException")
        void testLoadInvalidModelFile() throws IOException {
            Path invalidFile = tempDir.resolve("invalid.bin");
            Files.write(invalidFile, "not a valid model file".getBytes());
            
            assertThrows(ModelPersistenceException.class, 
                () -> ModelPersistence.load(invalidFile.toString()));
        }
        
        @Test
        @DisplayName("Load with type checking - success")
        void testLoadWithTypeCheckingSuccess() throws IOException {
            Path modelPath = tempDir.resolve("string_model.bin");
            ModelPersistence.save("test string", modelPath.toString());
            
            String loaded = ModelPersistence.load(modelPath.toString(), String.class);
            assertEquals("test string", loaded);
        }
        
        @Test
        @DisplayName("Load with type checking - type mismatch")
        void testLoadWithTypeCheckingMismatch() throws IOException {
            Path modelPath = tempDir.resolve("string_model.bin");
            ModelPersistence.save("test string", modelPath.toString());
            
            assertThrows(ModelPersistenceException.class, 
                () -> ModelPersistence.load(modelPath.toString(), Integer.class));
        }
    }
    
    @Nested
    @DisplayName("Metadata Tests")
    class MetadataTests {
        
        @Test
        @DisplayName("Get metadata with null path throws exception")
        void testGetMetadataNullPath() {
            assertThrows(IllegalArgumentException.class, 
                () -> ModelPersistence.getMetadata(null));
        }
        
        @Test
        @DisplayName("Get metadata with empty path throws exception")
        void testGetMetadataEmptyPath() {
            assertThrows(IllegalArgumentException.class, 
                () -> ModelPersistence.getMetadata(""));
        }
        
        @Test
        @DisplayName("Get metadata for non-existent file throws exception")
        void testGetMetadataNonExistentFile() {
            assertThrows(ModelPersistenceException.class, 
                () -> ModelPersistence.getMetadata("/non/existent/file.bin"));
        }
        
        @Test
        @DisplayName("Get metadata for invalid file throws exception")
        void testGetMetadataInvalidFile() throws IOException {
            Path invalidFile = tempDir.resolve("invalid.bin");
            Files.write(invalidFile, "not valid".getBytes());
            
            assertThrows(ModelPersistenceException.class, 
                () -> ModelPersistence.getMetadata(invalidFile.toString()));
        }
        
        @Test
        @DisplayName("Get metadata success")
        void testGetMetadataSuccess() throws IOException {
            Path modelPath = tempDir.resolve("model.bin");
            ModelPersistence.save("test model", modelPath.toString());
            
            ModelPersistence.ModelMetadata metadata = ModelPersistence.getMetadata(modelPath.toString());
            
            assertEquals("java.lang.String", metadata.getClassName());
            assertEquals("String", metadata.getSimpleClassName());
            assertTrue(metadata.getSavedTimestamp() > 0);
            assertTrue(metadata.getFileSize() > 0);
            assertTrue(metadata.toString().contains("String"));
        }
        
        @Test
        @DisplayName("Metadata getSimpleClassName with no package")
        void testGetSimpleClassNameNoPackage() {
            ModelPersistence.ModelMetadata metadata = 
                new ModelPersistence.ModelMetadata("SimpleClass", 12345L, 100L);
            assertEquals("SimpleClass", metadata.getSimpleClassName());
        }
    }
    
    @Nested
    @DisplayName("isValidModelFile Tests")
    class ValidModelFileTests {
        
        @Test
        @DisplayName("Null path returns false")
        void testNullPathReturnsFalse() {
            assertFalse(ModelPersistence.isValidModelFile(null));
        }
        
        @Test
        @DisplayName("Empty path returns false")
        void testEmptyPathReturnsFalse() {
            assertFalse(ModelPersistence.isValidModelFile(""));
        }
        
        @Test
        @DisplayName("Non-existent file returns false")
        void testNonExistentFileReturnsFalse() {
            assertFalse(ModelPersistence.isValidModelFile("/non/existent/file.bin"));
        }
        
        @Test
        @DisplayName("Directory returns false")
        void testDirectoryReturnsFalse() {
            assertFalse(ModelPersistence.isValidModelFile(tempDir.toString()));
        }
        
        @Test
        @DisplayName("Invalid model file returns false")
        void testInvalidModelFileReturnsFalse() throws IOException {
            Path invalidFile = tempDir.resolve("invalid.bin");
            Files.write(invalidFile, "not valid".getBytes());
            
            assertFalse(ModelPersistence.isValidModelFile(invalidFile.toString()));
        }
        
        @Test
        @DisplayName("Valid model file returns true")
        void testValidModelFileReturnsTrue() throws IOException {
            Path modelPath = tempDir.resolve("valid_model.bin");
            ModelPersistence.save("test", modelPath.toString());
            
            assertTrue(ModelPersistence.isValidModelFile(modelPath.toString()));
        }
    }
    
    @Nested
    @DisplayName("Byte Array Serialization Tests")
    class ByteArrayTests {
        
        @Test
        @DisplayName("toBytes with null model throws exception")
        void testToBytesNullModel() {
            assertThrows(IllegalArgumentException.class, 
                () -> ModelPersistence.toBytes(null));
        }
        
        @Test
        @DisplayName("toBytes and fromBytes roundtrip")
        void testToBytesFromBytesRoundtrip() {
            String original = "test model data";
            byte[] bytes = ModelPersistence.toBytes(original);
            
            assertNotNull(bytes);
            assertTrue(bytes.length > 0);
            
            String restored = ModelPersistence.fromBytes(bytes);
            assertEquals(original, restored);
        }
        
        @Test
        @DisplayName("fromBytes with null throws exception")
        void testFromBytesNull() {
            assertThrows(IllegalArgumentException.class, 
                () -> ModelPersistence.fromBytes(null));
        }
        
        @Test
        @DisplayName("fromBytes with empty array throws exception")
        void testFromBytesEmpty() {
            assertThrows(IllegalArgumentException.class, 
                () -> ModelPersistence.fromBytes(new byte[0]));
        }
        
        @Test
        @DisplayName("fromBytes with invalid data throws exception")
        void testFromBytesInvalidData() {
            byte[] invalidBytes = "invalid model data".getBytes();
            assertThrows(ModelPersistenceException.class, 
                () -> ModelPersistence.fromBytes(invalidBytes));
        }
        
        @Test
        @DisplayName("Complex object serialization")
        void testComplexObjectSerialization() {
            int[] data = {1, 2, 3, 4, 5};
            byte[] bytes = ModelPersistence.toBytes(data);
            
            int[] restored = ModelPersistence.fromBytes(bytes);
            assertArrayEquals(data, restored);
        }
    }
    
    @Nested
    @DisplayName("Full Roundtrip Tests")
    class RoundtripTests {
        
        @Test
        @DisplayName("Save and load string")
        void testSaveLoadString() throws IOException {
            Path modelPath = tempDir.resolve("string.bin");
            String original = "Hello, MindForge!";
            
            ModelPersistence.save(original, modelPath.toString());
            String loaded = ModelPersistence.load(modelPath.toString());
            
            assertEquals(original, loaded);
        }
        
        @Test
        @DisplayName("Save and load array")
        void testSaveLoadArray() throws IOException {
            Path modelPath = tempDir.resolve("array.bin");
            double[][] original = {{1.0, 2.0}, {3.0, 4.0}};
            
            ModelPersistence.save(original, modelPath.toString());
            double[][] loaded = ModelPersistence.load(modelPath.toString());
            
            assertArrayEquals(original, loaded);
        }
    }
}
