package com.mindforge.api;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/**
 * Simple HTTP client for interacting with ModelServer.
 */
public class ModelClient {
    
    private String baseUrl;
    private int timeout;
    
    /**
     * Create a client for the specified server.
     * 
     * @param host server host
     * @param port server port
     */
    public ModelClient(String host, int port) {
        this.baseUrl = "http://" + host + ":" + port;
        this.timeout = 30000; // 30 seconds
    }
    
    /**
     * Create a client for localhost.
     * 
     * @param port server port
     */
    public ModelClient(int port) {
        this("localhost", port);
    }
    
    /**
     * Set the request timeout.
     * 
     * @param timeout timeout in milliseconds
     */
    public void setTimeout(int timeout) {
        this.timeout = timeout;
    }
    
    /**
     * Check server health.
     * 
     * @return true if server is healthy
     */
    public boolean isHealthy() {
        try {
            String response = get("/health");
            return response.contains("healthy");
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Get list of available models.
     * 
     * @return JSON string with models
     * @throws IOException if request fails
     */
    public String getModels() throws IOException {
        return get("/models");
    }
    
    /**
     * Make predictions using a classifier model.
     * 
     * @param modelName model name
     * @param features feature matrix
     * @return predicted class labels
     * @throws IOException if request fails
     */
    public int[] predictClassification(String modelName, double[][] features) throws IOException {
        String json = featuresToJson(features);
        String response = post("/predict/" + modelName, json);
        return parseIntPredictions(response);
    }
    
    /**
     * Make predictions using a regressor model.
     * 
     * @param modelName model name
     * @param features feature matrix
     * @return predicted values
     * @throws IOException if request fails
     */
    public double[] predictRegression(String modelName, double[][] features) throws IOException {
        String json = featuresToJson(features);
        String response = post("/predict/" + modelName, json);
        return parseDoublePredictions(response);
    }
    
    /**
     * Perform a GET request.
     * 
     * @param path request path
     * @return response body
     * @throws IOException if request fails
     */
    public String get(String path) throws IOException {
        URL url = new URL(baseUrl + path);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        
        try {
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(timeout);
            conn.setReadTimeout(timeout);
            
            return readResponse(conn);
        } finally {
            conn.disconnect();
        }
    }
    
    /**
     * Perform a POST request.
     * 
     * @param path request path
     * @param body request body
     * @return response body
     * @throws IOException if request fails
     */
    public String post(String path, String body) throws IOException {
        URL url = new URL(baseUrl + path);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        
        try {
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setConnectTimeout(timeout);
            conn.setReadTimeout(timeout);
            conn.setDoOutput(true);
            
            try (OutputStream os = conn.getOutputStream()) {
                byte[] input = body.getBytes(StandardCharsets.UTF_8);
                os.write(input, 0, input.length);
            }
            
            return readResponse(conn);
        } finally {
            conn.disconnect();
        }
    }
    
    private String readResponse(HttpURLConnection conn) throws IOException {
        int responseCode = conn.getResponseCode();
        
        InputStream is = responseCode < 400 ? conn.getInputStream() : conn.getErrorStream();
        
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            StringBuilder response = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                response.append(line);
            }
            
            if (responseCode >= 400) {
                throw new IOException("HTTP " + responseCode + ": " + response.toString());
            }
            
            return response.toString();
        }
    }
    
    private String featuresToJson(double[][] features) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"features\": [");
        
        for (int i = 0; i < features.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append("[");
            for (int j = 0; j < features[i].length; j++) {
                if (j > 0) sb.append(", ");
                sb.append(features[i][j]);
            }
            sb.append("]");
        }
        
        sb.append("]}");
        return sb.toString();
    }
    
    private int[] parseIntPredictions(String json) {
        // Extract predictions array
        int start = json.indexOf("[");
        int end = json.lastIndexOf("]");
        
        if (start == -1 || end == -1) {
            throw new IllegalArgumentException("Invalid response format");
        }
        
        String arrayStr = json.substring(start + 1, end).trim();
        if (arrayStr.isEmpty()) {
            return new int[0];
        }
        
        String[] values = arrayStr.split(",");
        int[] predictions = new int[values.length];
        
        for (int i = 0; i < values.length; i++) {
            predictions[i] = Integer.parseInt(values[i].trim());
        }
        
        return predictions;
    }
    
    private double[] parseDoublePredictions(String json) {
        // Extract predictions array
        int start = json.indexOf("[");
        int end = json.lastIndexOf("]");
        
        if (start == -1 || end == -1) {
            throw new IllegalArgumentException("Invalid response format");
        }
        
        String arrayStr = json.substring(start + 1, end).trim();
        if (arrayStr.isEmpty()) {
            return new double[0];
        }
        
        String[] values = arrayStr.split(",");
        double[] predictions = new double[values.length];
        
        for (int i = 0; i < values.length; i++) {
            predictions[i] = Double.parseDouble(values[i].trim());
        }
        
        return predictions;
    }
    
    /**
     * Get the base URL.
     * 
     * @return base URL
     */
    public String getBaseUrl() {
        return baseUrl;
    }
}
