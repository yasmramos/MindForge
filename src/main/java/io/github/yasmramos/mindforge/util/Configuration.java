package io.github.yasmramos.mindforge.util;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

/**
 * Configuration management for MindForge.
 * Supports loading from properties files and programmatic configuration.
 */
public class Configuration {
    
    private Map<String, Object> config;
    private static Configuration defaultInstance;
    
    /**
     * Create an empty configuration.
     */
    public Configuration() {
        this.config = new HashMap<>();
        setDefaults();
    }
    
    /**
     * Get the default configuration instance.
     * 
     * @return default configuration
     */
    public static synchronized Configuration getDefault() {
        if (defaultInstance == null) {
            defaultInstance = new Configuration();
        }
        return defaultInstance;
    }
    
    /**
     * Set default configuration values.
     */
    private void setDefaults() {
        // Neural network defaults
        config.put("nn.learning_rate", 0.01);
        config.put("nn.epochs", 100);
        config.put("nn.batch_size", 32);
        config.put("nn.momentum", 0.9);
        config.put("nn.weight_decay", 0.0001);
        
        // Training defaults
        config.put("train.validation_split", 0.2);
        config.put("train.shuffle", true);
        config.put("train.random_seed", 42);
        config.put("train.early_stopping", true);
        config.put("train.patience", 10);
        
        // Logging defaults
        config.put("log.level", "INFO");
        config.put("log.console", true);
        config.put("log.file", null);
        
        // Model defaults
        config.put("model.save_best", true);
        config.put("model.checkpoint_interval", 10);
    }
    
    /**
     * Load configuration from a properties file.
     * 
     * @param filePath path to properties file
     * @throws IOException if file cannot be read
     */
    public void loadFromFile(String filePath) throws IOException {
        Properties props = new Properties();
        try (InputStream is = new FileInputStream(filePath)) {
            props.load(is);
        }
        
        for (String key : props.stringPropertyNames()) {
            String value = props.getProperty(key);
            config.put(key, parseValue(value));
        }
    }
    
    /**
     * Save configuration to a properties file.
     * 
     * @param filePath path to properties file
     * @throws IOException if file cannot be written
     */
    public void saveToFile(String filePath) throws IOException {
        Properties props = new Properties();
        for (Map.Entry<String, Object> entry : config.entrySet()) {
            if (entry.getValue() != null) {
                props.setProperty(entry.getKey(), entry.getValue().toString());
            }
        }
        
        try (OutputStream os = new FileOutputStream(filePath)) {
            props.store(os, "MindForge Configuration");
        }
    }
    
    /**
     * Load configuration from JSON string.
     * 
     * @param json JSON configuration string
     */
    public void loadFromJson(String json) {
        // Simple JSON parser for flat key-value pairs
        json = json.trim();
        if (json.startsWith("{") && json.endsWith("}")) {
            json = json.substring(1, json.length() - 1);
        }
        
        String[] pairs = json.split(",");
        for (String pair : pairs) {
            String[] keyValue = pair.split(":", 2);
            if (keyValue.length == 2) {
                String key = keyValue[0].trim().replaceAll("\"", "");
                String value = keyValue[1].trim().replaceAll("\"", "");
                config.put(key, parseValue(value));
            }
        }
    }
    
    /**
     * Export configuration to JSON string.
     * 
     * @return JSON string
     */
    public String toJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        
        int count = 0;
        for (Map.Entry<String, Object> entry : config.entrySet()) {
            sb.append("  \"").append(entry.getKey()).append("\": ");
            Object value = entry.getValue();
            
            if (value == null) {
                sb.append("null");
            } else if (value instanceof String) {
                sb.append("\"").append(value).append("\"");
            } else if (value instanceof Boolean) {
                sb.append(value.toString().toLowerCase());
            } else {
                sb.append(value);
            }
            
            if (++count < config.size()) {
                sb.append(",");
            }
            sb.append("\n");
        }
        
        sb.append("}");
        return sb.toString();
    }
    
    private Object parseValue(String value) {
        if (value == null || value.equalsIgnoreCase("null")) {
            return null;
        }
        if (value.equalsIgnoreCase("true")) {
            return true;
        }
        if (value.equalsIgnoreCase("false")) {
            return false;
        }
        
        try {
            if (value.contains(".")) {
                return Double.parseDouble(value);
            } else {
                return Integer.parseInt(value);
            }
        } catch (NumberFormatException e) {
            return value;
        }
    }
    
    /**
     * Get a string value.
     * 
     * @param key configuration key
     * @return string value or null
     */
    public String getString(String key) {
        Object value = config.get(key);
        return value != null ? value.toString() : null;
    }
    
    /**
     * Get a string value with default.
     * 
     * @param key configuration key
     * @param defaultValue default value
     * @return string value or default
     */
    public String getString(String key, String defaultValue) {
        String value = getString(key);
        return value != null ? value : defaultValue;
    }
    
    /**
     * Get an integer value.
     * 
     * @param key configuration key
     * @return integer value or 0
     */
    public int getInt(String key) {
        return getInt(key, 0);
    }
    
    /**
     * Get an integer value with default.
     * 
     * @param key configuration key
     * @param defaultValue default value
     * @return integer value or default
     */
    public int getInt(String key, int defaultValue) {
        Object value = config.get(key);
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        if (value instanceof String) {
            try {
                return Integer.parseInt((String) value);
            } catch (NumberFormatException e) {
                return defaultValue;
            }
        }
        return defaultValue;
    }
    
    /**
     * Get a double value.
     * 
     * @param key configuration key
     * @return double value or 0.0
     */
    public double getDouble(String key) {
        return getDouble(key, 0.0);
    }
    
    /**
     * Get a double value with default.
     * 
     * @param key configuration key
     * @param defaultValue default value
     * @return double value or default
     */
    public double getDouble(String key, double defaultValue) {
        Object value = config.get(key);
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        if (value instanceof String) {
            try {
                return Double.parseDouble((String) value);
            } catch (NumberFormatException e) {
                return defaultValue;
            }
        }
        return defaultValue;
    }
    
    /**
     * Get a boolean value.
     * 
     * @param key configuration key
     * @return boolean value or false
     */
    public boolean getBoolean(String key) {
        return getBoolean(key, false);
    }
    
    /**
     * Get a boolean value with default.
     * 
     * @param key configuration key
     * @param defaultValue default value
     * @return boolean value or default
     */
    public boolean getBoolean(String key, boolean defaultValue) {
        Object value = config.get(key);
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        if (value instanceof String) {
            return Boolean.parseBoolean((String) value);
        }
        return defaultValue;
    }
    
    /**
     * Set a configuration value.
     * 
     * @param key configuration key
     * @param value value to set
     */
    public void set(String key, Object value) {
        config.put(key, value);
    }
    
    /**
     * Check if a key exists.
     * 
     * @param key configuration key
     * @return true if key exists
     */
    public boolean hasKey(String key) {
        return config.containsKey(key);
    }
    
    /**
     * Remove a configuration key.
     * 
     * @param key configuration key
     */
    public void remove(String key) {
        config.remove(key);
    }
    
    /**
     * Clear all configuration.
     */
    public void clear() {
        config.clear();
    }
    
    /**
     * Reset to default values.
     */
    public void reset() {
        config.clear();
        setDefaults();
    }
    
    /**
     * Get all configuration keys.
     * 
     * @return set of keys
     */
    public java.util.Set<String> getKeys() {
        return config.keySet();
    }
    
    @Override
    public String toString() {
        return toJson();
    }
}
