package io.github.yasmramos.mindforge.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.io.FileWriter;
import java.nio.file.Path;

@DisplayName("Configuration Tests")
class ConfigurationTest {
    
    @TempDir
    Path tempDir;
    
    @Nested
    @DisplayName("Properties File Tests")
    class PropertiesFileTests {
        
        @Test
        @DisplayName("Should load properties file")
        void testLoadPropertiesFile() throws Exception {
            File propsFile = tempDir.resolve("test.properties").toFile();
            try (FileWriter writer = new FileWriter(propsFile)) {
                writer.write("key1=value1\n");
                writer.write("key2=value2\n");
                writer.write("number=42\n");
            }
            
            Configuration config = new Configuration();
            config.loadFromFile(propsFile.getAbsolutePath());
            
            assertEquals("value1", config.getString("key1"));
            assertEquals("value2", config.getString("key2"));
        }
        
        @Test
        @DisplayName("Should return default for missing key")
        void testDefaultValue() throws Exception {
            Configuration config = new Configuration();
            
            assertEquals("default", config.getString("missing", "default"));
        }
        
        @Test
        @DisplayName("Should get integer value")
        void testGetInteger() throws Exception {
            File propsFile = tempDir.resolve("test3.properties").toFile();
            try (FileWriter writer = new FileWriter(propsFile)) {
                writer.write("count=42\n");
            }
            
            Configuration config = new Configuration();
            config.loadFromFile(propsFile.getAbsolutePath());
            
            assertEquals(42, config.getInt("count"));
        }
        
        @Test
        @DisplayName("Should get double value")
        void testGetDouble() throws Exception {
            File propsFile = tempDir.resolve("test4.properties").toFile();
            try (FileWriter writer = new FileWriter(propsFile)) {
                writer.write("rate=0.01\n");
            }
            
            Configuration config = new Configuration();
            config.loadFromFile(propsFile.getAbsolutePath());
            
            assertEquals(0.01, config.getDouble("rate"), 0.001);
        }
        
        @Test
        @DisplayName("Should get boolean value")
        void testGetBoolean() throws Exception {
            File propsFile = tempDir.resolve("test5.properties").toFile();
            try (FileWriter writer = new FileWriter(propsFile)) {
                writer.write("enabled=true\n");
                writer.write("disabled=false\n");
            }
            
            Configuration config = new Configuration();
            config.loadFromFile(propsFile.getAbsolutePath());
            
            assertTrue(config.getBoolean("enabled"));
            assertFalse(config.getBoolean("disabled"));
        }
    }
    
    @Nested
    @DisplayName("Default Configuration Tests")
    class DefaultConfigTests {
        
        @Test
        @DisplayName("Should create empty configuration")
        void testEmptyConfiguration() {
            Configuration config = new Configuration();
            
            assertNotNull(config);
            assertEquals("default", config.getString("any.key", "default"));
        }
        
        @Test
        @DisplayName("Should set and get values programmatically")
        void testSetAndGet() {
            Configuration config = new Configuration();
            config.set("mykey", "myvalue");
            
            assertEquals("myvalue", config.getString("mykey"));
        }
        
        @Test
        @DisplayName("Should set integer values")
        void testSetInteger() {
            Configuration config = new Configuration();
            config.set("count", 100);
            
            assertEquals(100, config.getInt("count"));
        }
        
        @Test
        @DisplayName("Should set double values")
        void testSetDouble() {
            Configuration config = new Configuration();
            config.set("rate", 0.05);
            
            assertEquals(0.05, config.getDouble("rate"), 0.001);
        }
        
        @Test
        @DisplayName("Should get default configuration")
        void testGetDefault() {
            Configuration config = Configuration.getDefault();
            assertNotNull(config);
        }
        
        @Test
        @DisplayName("Should have default neural network settings")
        void testDefaultNNSettings() {
            Configuration config = new Configuration();
            
            // Check default values
            assertEquals(0.01, config.getDouble("nn.learning_rate"), 0.001);
            assertEquals(100, config.getInt("nn.epochs"));
            assertEquals(32, config.getInt("nn.batch_size"));
        }
    }
    
    @Nested
    @DisplayName("Save Configuration Tests")
    class SaveConfigTests {
        
        @Test
        @DisplayName("Should save to properties file")
        void testSaveToProperties() throws Exception {
            Configuration config = new Configuration();
            config.set("key1", "value1");
            config.set("key2", "value2");
            
            File outFile = tempDir.resolve("output.properties").toFile();
            config.saveToFile(outFile.getAbsolutePath());
            
            assertTrue(outFile.exists());
            assertTrue(outFile.length() > 0);
        }
        
        @Test
        @DisplayName("Should reload saved configuration")
        void testReloadSavedConfig() throws Exception {
            Configuration config1 = new Configuration();
            config1.set("testkey", "testvalue");
            
            File outFile = tempDir.resolve("reload.properties").toFile();
            config1.saveToFile(outFile.getAbsolutePath());
            
            Configuration config2 = new Configuration();
            config2.loadFromFile(outFile.getAbsolutePath());
            assertEquals("testvalue", config2.getString("testkey"));
        }
    }
    
    @Nested
    @DisplayName("JSON Configuration Tests")
    class JsonConfigTests {
        
        @Test
        @DisplayName("Should export to JSON")
        void testToJson() {
            Configuration config = new Configuration();
            config.set("key1", "value1");
            
            String json = config.toJson();
            assertNotNull(json);
            assertTrue(json.contains("key1"));
        }
        
        @Test
        @DisplayName("Should load from JSON")
        void testLoadFromJson() {
            Configuration config = new Configuration();
            config.loadFromJson("{\"mykey\": \"myvalue\", \"number\": 42}");
            
            assertEquals("myvalue", config.getString("mykey"));
        }
    }
    
    @Nested
    @DisplayName("Type Conversion Tests")
    class TypeConversionTests {
        
        @Test
        @DisplayName("Should convert string to integer")
        void testStringToInt() throws Exception {
            File propsFile = tempDir.resolve("conv1.properties").toFile();
            try (FileWriter writer = new FileWriter(propsFile)) {
                writer.write("num=123\n");
            }
            
            Configuration config = new Configuration();
            config.loadFromFile(propsFile.getAbsolutePath());
            assertEquals(123, config.getInt("num"));
        }
        
        @Test
        @DisplayName("Should return default for invalid integer")
        void testInvalidInteger() {
            Configuration config = new Configuration();
            config.set("num", "notanumber");
            
            assertEquals(0, config.getInt("num", 0));
        }
        
        @Test
        @DisplayName("Should handle boolean true")
        void testBooleanTrue() {
            Configuration config = new Configuration();
            config.set("bool1", true);
            
            assertTrue(config.getBoolean("bool1"));
        }
    }
    
    @Nested
    @DisplayName("Utility Methods Tests")
    class UtilityTests {
        
        @Test
        @DisplayName("Should check if key exists")
        void testHasKey() {
            Configuration config = new Configuration();
            config.set("existing", "value");
            
            assertTrue(config.hasKey("existing"));
            assertFalse(config.hasKey("nonexisting"));
        }
        
        @Test
        @DisplayName("Should remove key")
        void testRemove() {
            Configuration config = new Configuration();
            config.set("toremove", "value");
            
            assertTrue(config.hasKey("toremove"));
            config.remove("toremove");
            assertFalse(config.hasKey("toremove"));
        }
        
        @Test
        @DisplayName("Should clear all configuration")
        void testClear() {
            Configuration config = new Configuration();
            config.set("key1", "value1");
            config.set("key2", "value2");
            
            config.clear();
            
            assertFalse(config.hasKey("key1"));
            assertFalse(config.hasKey("key2"));
        }
        
        @Test
        @DisplayName("Should reset to defaults")
        void testReset() {
            Configuration config = new Configuration();
            config.set("custom", "value");
            
            config.reset();
            
            assertFalse(config.hasKey("custom"));
            // Default values should be back
            assertEquals(0.01, config.getDouble("nn.learning_rate"), 0.001);
        }
        
        @Test
        @DisplayName("Should get all keys")
        void testGetKeys() {
            Configuration config = new Configuration();
            config.set("mykey", "value");
            
            java.util.Set<String> keys = config.getKeys();
            
            assertNotNull(keys);
            assertTrue(keys.contains("mykey"));
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle missing file gracefully")
        void testMissingFile() {
            Configuration config = new Configuration();
            
            assertThrows(Exception.class, () -> {
                config.loadFromFile("/nonexistent/path/config.properties");
            });
        }
        
        @Test
        @DisplayName("Should handle empty properties file")
        void testEmptyPropertiesFile() throws Exception {
            File emptyFile = tempDir.resolve("empty.properties").toFile();
            emptyFile.createNewFile();
            
            Configuration config = new Configuration();
            config.loadFromFile(emptyFile.getAbsolutePath());
            
            assertEquals("default", config.getString("anykey", "default"));
        }
        
        @Test
        @DisplayName("Should handle special characters in values")
        void testSpecialCharacters() throws Exception {
            File propsFile = tempDir.resolve("special.properties").toFile();
            try (FileWriter writer = new FileWriter(propsFile)) {
                writer.write("path=/home/user/data\n");
                writer.write("url=http://example.com\n");
            }
            
            Configuration config = new Configuration();
            config.loadFromFile(propsFile.getAbsolutePath());
            assertEquals("/home/user/data", config.getString("path"));
        }
    }
}
