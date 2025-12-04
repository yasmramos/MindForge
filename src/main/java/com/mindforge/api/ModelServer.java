package com.mindforge.api;

import com.mindforge.classification.Classifier;
import com.mindforge.regression.Regressor;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.*;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;

/**
 * Simple HTTP server for serving ML models as REST API.
 */
public class ModelServer {
    
    private HttpServer server;
    private int port;
    private Map<String, Object> models;
    private boolean running;
    
    /**
     * Create a model server on the specified port.
     * 
     * @param port server port
     */
    public ModelServer(int port) {
        this.port = port;
        this.models = new HashMap<>();
        this.running = false;
    }
    
    /**
     * Register a classifier model.
     * 
     * @param name model name/endpoint
     * @param classifier classifier model
     */
    public void registerClassifier(String name, Classifier classifier) {
        models.put(name, classifier);
    }
    
    /**
     * Register a regressor model.
     * 
     * @param name model name/endpoint
     * @param regressor regressor model
     */
    public void registerRegressor(String name, Regressor regressor) {
        models.put(name, regressor);
    }
    
    /**
     * Start the server.
     * 
     * @throws IOException if server cannot be started
     */
    public void start() throws IOException {
        server = HttpServer.create(new InetSocketAddress(port), 0);
        
        // Health check endpoint
        server.createContext("/health", new HealthHandler());
        
        // List models endpoint
        server.createContext("/models", new ModelsHandler());
        
        // Prediction endpoints for each model
        for (String modelName : models.keySet()) {
            server.createContext("/predict/" + modelName, new PredictHandler(modelName));
        }
        
        server.setExecutor(Executors.newFixedThreadPool(10));
        server.start();
        running = true;
        
        System.out.println("MindForge Model Server started on port " + port);
        System.out.println("Endpoints:");
        System.out.println("  GET  /health - Health check");
        System.out.println("  GET  /models - List registered models");
        for (String modelName : models.keySet()) {
            System.out.println("  POST /predict/" + modelName + " - Predict with " + modelName);
        }
    }
    
    /**
     * Stop the server.
     */
    public void stop() {
        if (server != null) {
            server.stop(0);
            running = false;
            System.out.println("Server stopped");
        }
    }
    
    /**
     * Check if server is running.
     * 
     * @return true if running
     */
    public boolean isRunning() {
        return running;
    }
    
    /**
     * Get the server port.
     * 
     * @return port number
     */
    public int getPort() {
        return port;
    }
    
    /**
     * Health check handler.
     */
    private class HealthHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            String response = "{\"status\": \"healthy\", \"models\": " + models.size() + "}";
            sendResponse(exchange, 200, response);
        }
    }
    
    /**
     * Models list handler.
     */
    private class ModelsHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            StringBuilder sb = new StringBuilder();
            sb.append("{\"models\": [");
            
            int count = 0;
            for (Map.Entry<String, Object> entry : models.entrySet()) {
                if (count > 0) sb.append(", ");
                sb.append("{\"name\": \"").append(entry.getKey()).append("\", ");
                sb.append("\"type\": \"");
                if (entry.getValue() instanceof Classifier) {
                    sb.append("classifier");
                } else if (entry.getValue() instanceof Regressor) {
                    sb.append("regressor");
                } else {
                    sb.append("unknown");
                }
                sb.append("\"}");
                count++;
            }
            
            sb.append("]}");
            sendResponse(exchange, 200, sb.toString());
        }
    }
    
    /**
     * Prediction handler.
     */
    private class PredictHandler implements HttpHandler {
        private String modelName;
        
        public PredictHandler(String modelName) {
            this.modelName = modelName;
        }
        
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod())) {
                sendResponse(exchange, 405, "{\"error\": \"Method not allowed. Use POST.\"}");
                return;
            }
            
            try {
                // Read request body
                String body = readRequestBody(exchange);
                double[][] features = parseFeatures(body);
                
                Object model = models.get(modelName);
                String response;
                
                if (model instanceof Classifier) {
                    Classifier classifier = (Classifier) model;
                    int[] predictions = classifier.predict(features);
                    response = formatPredictions(predictions);
                } else if (model instanceof Regressor) {
                    Regressor regressor = (Regressor) model;
                    double[] predictions = regressor.predict(features);
                    response = formatPredictions(predictions);
                } else {
                    sendResponse(exchange, 500, "{\"error\": \"Unknown model type\"}");
                    return;
                }
                
                sendResponse(exchange, 200, response);
                
            } catch (Exception e) {
                sendResponse(exchange, 400, "{\"error\": \"" + e.getMessage() + "\"}");
            }
        }
    }
    
    /**
     * Parse features from JSON request.
     * Expected format: {"features": [[1.0, 2.0], [3.0, 4.0]]}
     */
    private double[][] parseFeatures(String json) {
        // Simple JSON parser for features array
        json = json.trim();
        
        int start = json.indexOf("[[");
        int end = json.lastIndexOf("]]");
        
        if (start == -1 || end == -1) {
            throw new IllegalArgumentException("Invalid JSON format. Expected: {\"features\": [[...]]}");
        }
        
        String arrayStr = json.substring(start + 1, end + 1);
        
        // Split into rows
        String[] rows = arrayStr.split("\\],\\s*\\[");
        double[][] features = new double[rows.length][];
        
        for (int i = 0; i < rows.length; i++) {
            String row = rows[i].replaceAll("[\\[\\]]", "").trim();
            String[] values = row.split(",");
            features[i] = new double[values.length];
            
            for (int j = 0; j < values.length; j++) {
                features[i][j] = Double.parseDouble(values[j].trim());
            }
        }
        
        return features;
    }
    
    /**
     * Format int predictions as JSON.
     */
    private String formatPredictions(int[] predictions) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"predictions\": [");
        for (int i = 0; i < predictions.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(predictions[i]);
        }
        sb.append("]}");
        return sb.toString();
    }
    
    /**
     * Format double predictions as JSON.
     */
    private String formatPredictions(double[] predictions) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"predictions\": [");
        for (int i = 0; i < predictions.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.6f", predictions[i]));
        }
        sb.append("]}");
        return sb.toString();
    }
    
    /**
     * Read request body.
     */
    private String readRequestBody(HttpExchange exchange) throws IOException {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(exchange.getRequestBody(), StandardCharsets.UTF_8))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            return sb.toString();
        }
    }
    
    /**
     * Send HTTP response.
     */
    private void sendResponse(HttpExchange exchange, int statusCode, String response) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        
        byte[] bytes = response.getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(statusCode, bytes.length);
        
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }
    
    /**
     * Main method to start a server with example models.
     */
    public static void main(String[] args) {
        int port = 8080;
        if (args.length > 0) {
            port = Integer.parseInt(args[0]);
        }
        
        ModelServer server = new ModelServer(port);
        
        try {
            server.start();
            
            // Keep running until interrupted
            System.out.println("\nPress Ctrl+C to stop the server");
            Thread.currentThread().join();
            
        } catch (IOException e) {
            System.err.println("Failed to start server: " + e.getMessage());
        } catch (InterruptedException e) {
            server.stop();
        }
    }
}
