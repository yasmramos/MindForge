package io.github.yasmramos.mindforge.visualization;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

/**
 * Chart generator for visualization of ML results.
 * Generates HTML/SVG charts that can be viewed in a browser.
 */
public class ChartGenerator {
    
    private static final int DEFAULT_WIDTH = 800;
    private static final int DEFAULT_HEIGHT = 500;
    private static final String[] DEFAULT_COLORS = {
        "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b"
    };
    
    /**
     * Generate a line chart from data.
     * 
     * @param title chart title
     * @param xLabel x-axis label
     * @param yLabel y-axis label
     * @param data y values
     * @param outputPath output HTML file path
     * @throws IOException if file cannot be written
     */
    public static void lineChart(String title, String xLabel, String yLabel,
                                  double[] data, String outputPath) throws IOException {
        lineChart(title, xLabel, yLabel, new double[][]{data}, new String[]{"Series 1"}, outputPath);
    }
    
    /**
     * Generate a multi-series line chart.
     * 
     * @param title chart title
     * @param xLabel x-axis label
     * @param yLabel y-axis label
     * @param series array of data series
     * @param seriesNames names for each series
     * @param outputPath output HTML file path
     * @throws IOException if file cannot be written
     */
    public static void lineChart(String title, String xLabel, String yLabel,
                                  double[][] series, String[] seriesNames,
                                  String outputPath) throws IOException {
        StringBuilder sb = new StringBuilder();
        generateHtmlHeader(sb, title, DEFAULT_WIDTH, DEFAULT_HEIGHT);
        
        // Find data range
        double minVal = Double.MAX_VALUE, maxVal = Double.MIN_VALUE;
        int maxLen = 0;
        for (double[] data : series) {
            maxLen = Math.max(maxLen, data.length);
            for (double v : data) {
                minVal = Math.min(minVal, v);
                maxVal = Math.max(maxVal, v);
            }
        }
        
        double range = maxVal - minVal;
        if (range == 0) range = 1;
        double padding = range * 0.1;
        minVal -= padding;
        maxVal += padding;
        
        int chartWidth = DEFAULT_WIDTH - 100;
        int chartHeight = DEFAULT_HEIGHT - 100;
        int offsetX = 60;
        int offsetY = 40;
        
        // Draw axes
        sb.append(String.format("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#333' stroke-width='2'/>\n",
                offsetX, offsetY, offsetX, offsetY + chartHeight));
        sb.append(String.format("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#333' stroke-width='2'/>\n",
                offsetX, offsetY + chartHeight, offsetX + chartWidth, offsetY + chartHeight));
        
        // Draw grid lines and labels
        for (int i = 0; i <= 5; i++) {
            double yVal = minVal + (maxVal - minVal) * i / 5;
            int y = offsetY + chartHeight - (int) ((yVal - minVal) / (maxVal - minVal) * chartHeight);
            sb.append(String.format("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#ddd' stroke-width='1'/>\n",
                    offsetX, y, offsetX + chartWidth, y));
            sb.append(String.format("<text x='%d' y='%d' text-anchor='end' font-size='12'>%.2f</text>\n",
                    offsetX - 5, y + 4, yVal));
        }
        
        // Draw data lines
        for (int s = 0; s < series.length; s++) {
            double[] data = series[s];
            String color = DEFAULT_COLORS[s % DEFAULT_COLORS.length];
            
            StringBuilder pathData = new StringBuilder();
            for (int i = 0; i < data.length; i++) {
                int x = offsetX + (int) ((double) i / (maxLen - 1) * chartWidth);
                int y = offsetY + chartHeight - (int) ((data[i] - minVal) / (maxVal - minVal) * chartHeight);
                
                if (i == 0) {
                    pathData.append(String.format("M %d %d", x, y));
                } else {
                    pathData.append(String.format(" L %d %d", x, y));
                }
            }
            
            sb.append(String.format("<path d='%s' stroke='%s' stroke-width='2' fill='none'/>\n",
                    pathData, color));
        }
        
        // Draw legend
        int legendY = offsetY + 20;
        for (int s = 0; s < seriesNames.length; s++) {
            String color = DEFAULT_COLORS[s % DEFAULT_COLORS.length];
            sb.append(String.format("<rect x='%d' y='%d' width='15' height='15' fill='%s'/>\n",
                    offsetX + chartWidth - 100, legendY + s * 20, color));
            sb.append(String.format("<text x='%d' y='%d' font-size='12'>%s</text>\n",
                    offsetX + chartWidth - 80, legendY + s * 20 + 12, seriesNames[s]));
        }
        
        // Labels
        sb.append(String.format("<text x='%d' y='%d' text-anchor='middle' font-size='16' font-weight='bold'>%s</text>\n",
                offsetX + chartWidth / 2, 25, title));
        sb.append(String.format("<text x='%d' y='%d' text-anchor='middle' font-size='14'>%s</text>\n",
                offsetX + chartWidth / 2, offsetY + chartHeight + 40, xLabel));
        sb.append(String.format("<text x='20' y='%d' text-anchor='middle' font-size='14' transform='rotate(-90, 20, %d)'>%s</text>\n",
                offsetY + chartHeight / 2, offsetY + chartHeight / 2, yLabel));
        
        generateHtmlFooter(sb);
        writeToFile(sb.toString(), outputPath);
    }
    
    /**
     * Generate a training history chart.
     * 
     * @param trainLoss training loss values
     * @param valLoss validation loss values (can be null)
     * @param outputPath output HTML file path
     * @throws IOException if file cannot be written
     */
    public static void trainingHistory(List<Double> trainLoss, List<Double> valLoss,
                                        String outputPath) throws IOException {
        double[] train = trainLoss.stream().mapToDouble(d -> d).toArray();
        
        if (valLoss != null && !valLoss.isEmpty()) {
            double[] val = valLoss.stream().mapToDouble(d -> d).toArray();
            lineChart("Training History", "Epoch", "Loss",
                    new double[][]{train, val},
                    new String[]{"Training Loss", "Validation Loss"},
                    outputPath);
        } else {
            lineChart("Training History", "Epoch", "Loss", train, outputPath);
        }
    }
    
    /**
     * Generate a bar chart.
     * 
     * @param title chart title
     * @param labels bar labels
     * @param values bar values
     * @param outputPath output HTML file path
     * @throws IOException if file cannot be written
     */
    public static void barChart(String title, String[] labels, double[] values,
                                 String outputPath) throws IOException {
        StringBuilder sb = new StringBuilder();
        generateHtmlHeader(sb, title, DEFAULT_WIDTH, DEFAULT_HEIGHT);
        
        double maxVal = 0;
        for (double v : values) {
            maxVal = Math.max(maxVal, v);
        }
        if (maxVal == 0) maxVal = 1;
        
        int chartWidth = DEFAULT_WIDTH - 100;
        int chartHeight = DEFAULT_HEIGHT - 120;
        int offsetX = 60;
        int offsetY = 50;
        int barWidth = (chartWidth - 20) / values.length - 10;
        
        // Draw axes
        sb.append(String.format("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#333' stroke-width='2'/>\n",
                offsetX, offsetY, offsetX, offsetY + chartHeight));
        sb.append(String.format("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#333' stroke-width='2'/>\n",
                offsetX, offsetY + chartHeight, offsetX + chartWidth, offsetY + chartHeight));
        
        // Draw bars
        for (int i = 0; i < values.length; i++) {
            int x = offsetX + 20 + i * (barWidth + 10);
            int barHeight = (int) (values[i] / maxVal * chartHeight);
            int y = offsetY + chartHeight - barHeight;
            String color = DEFAULT_COLORS[i % DEFAULT_COLORS.length];
            
            sb.append(String.format("<rect x='%d' y='%d' width='%d' height='%d' fill='%s'/>\n",
                    x, y, barWidth, barHeight, color));
            
            // Value label
            sb.append(String.format("<text x='%d' y='%d' text-anchor='middle' font-size='12'>%.2f</text>\n",
                    x + barWidth / 2, y - 5, values[i]));
            
            // Bar label
            sb.append(String.format("<text x='%d' y='%d' text-anchor='middle' font-size='11' transform='rotate(-45, %d, %d)'>%s</text>\n",
                    x + barWidth / 2, offsetY + chartHeight + 20,
                    x + barWidth / 2, offsetY + chartHeight + 20, labels[i]));
        }
        
        // Title
        sb.append(String.format("<text x='%d' y='25' text-anchor='middle' font-size='16' font-weight='bold'>%s</text>\n",
                offsetX + chartWidth / 2, title));
        
        generateHtmlFooter(sb);
        writeToFile(sb.toString(), outputPath);
    }
    
    /**
     * Generate a confusion matrix heatmap.
     * 
     * @param matrix confusion matrix
     * @param labels class labels
     * @param outputPath output HTML file path
     * @throws IOException if file cannot be written
     */
    public static void confusionMatrixHeatmap(int[][] matrix, String[] labels,
                                               String outputPath) throws IOException {
        StringBuilder sb = new StringBuilder();
        int size = matrix.length;
        int cellSize = 60;
        int width = size * cellSize + 150;
        int height = size * cellSize + 150;
        
        generateHtmlHeader(sb, "Confusion Matrix", width, height);
        
        int offsetX = 80;
        int offsetY = 60;
        
        // Find max value for color scaling
        int maxVal = 0;
        for (int[] row : matrix) {
            for (int v : row) {
                maxVal = Math.max(maxVal, v);
            }
        }
        if (maxVal == 0) maxVal = 1;
        
        // Draw cells
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int x = offsetX + j * cellSize;
                int y = offsetY + i * cellSize;
                
                // Color based on value (blue gradient)
                double intensity = (double) matrix[i][j] / maxVal;
                int r = (int) (255 * (1 - intensity));
                int g = (int) (255 * (1 - intensity));
                int b = 255;
                
                sb.append(String.format("<rect x='%d' y='%d' width='%d' height='%d' fill='rgb(%d,%d,%d)' stroke='white'/>\n",
                        x, y, cellSize, cellSize, r, g, b));
                
                // Value text
                String textColor = intensity > 0.5 ? "white" : "black";
                sb.append(String.format("<text x='%d' y='%d' text-anchor='middle' font-size='14' fill='%s'>%d</text>\n",
                        x + cellSize / 2, y + cellSize / 2 + 5, textColor, matrix[i][j]));
            }
        }
        
        // Labels
        for (int i = 0; i < size; i++) {
            String label = labels != null && i < labels.length ? labels[i] : String.valueOf(i);
            
            // Row labels (True)
            sb.append(String.format("<text x='%d' y='%d' text-anchor='end' font-size='12'>%s</text>\n",
                    offsetX - 10, offsetY + i * cellSize + cellSize / 2 + 5, label));
            
            // Column labels (Predicted)
            sb.append(String.format("<text x='%d' y='%d' text-anchor='middle' font-size='12'>%s</text>\n",
                    offsetX + i * cellSize + cellSize / 2, offsetY - 10, label));
        }
        
        // Axis titles
        sb.append(String.format("<text x='%d' y='25' text-anchor='middle' font-size='14' font-weight='bold'>Predicted</text>\n",
                offsetX + size * cellSize / 2));
        sb.append(String.format("<text x='20' y='%d' text-anchor='middle' font-size='14' font-weight='bold' transform='rotate(-90, 20, %d)'>True</text>\n",
                offsetY + size * cellSize / 2, offsetY + size * cellSize / 2));
        
        generateHtmlFooter(sb);
        writeToFile(sb.toString(), outputPath);
    }
    
    /**
     * Generate a scatter plot.
     * 
     * @param title chart title
     * @param xLabel x-axis label
     * @param yLabel y-axis label
     * @param x x coordinates
     * @param y y coordinates
     * @param labels point labels (for coloring)
     * @param outputPath output HTML file path
     * @throws IOException if file cannot be written
     */
    public static void scatterPlot(String title, String xLabel, String yLabel,
                                    double[] x, double[] y, int[] labels,
                                    String outputPath) throws IOException {
        StringBuilder sb = new StringBuilder();
        generateHtmlHeader(sb, title, DEFAULT_WIDTH, DEFAULT_HEIGHT);
        
        // Find data range
        double minX = Double.MAX_VALUE, maxX = Double.MIN_VALUE;
        double minY = Double.MAX_VALUE, maxY = Double.MIN_VALUE;
        
        for (int i = 0; i < x.length; i++) {
            minX = Math.min(minX, x[i]);
            maxX = Math.max(maxX, x[i]);
            minY = Math.min(minY, y[i]);
            maxY = Math.max(maxY, y[i]);
        }
        
        double rangeX = maxX - minX;
        double rangeY = maxY - minY;
        if (rangeX == 0) rangeX = 1;
        if (rangeY == 0) rangeY = 1;
        
        int chartWidth = DEFAULT_WIDTH - 100;
        int chartHeight = DEFAULT_HEIGHT - 100;
        int offsetX = 60;
        int offsetY = 40;
        
        // Draw axes
        sb.append(String.format("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#333' stroke-width='2'/>\n",
                offsetX, offsetY, offsetX, offsetY + chartHeight));
        sb.append(String.format("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#333' stroke-width='2'/>\n",
                offsetX, offsetY + chartHeight, offsetX + chartWidth, offsetY + chartHeight));
        
        // Draw points
        for (int i = 0; i < x.length; i++) {
            int px = offsetX + (int) ((x[i] - minX) / rangeX * chartWidth);
            int py = offsetY + chartHeight - (int) ((y[i] - minY) / rangeY * chartHeight);
            
            String color = DEFAULT_COLORS[0];
            if (labels != null && i < labels.length) {
                color = DEFAULT_COLORS[labels[i] % DEFAULT_COLORS.length];
            }
            
            sb.append(String.format("<circle cx='%d' cy='%d' r='5' fill='%s' opacity='0.7'/>\n",
                    px, py, color));
        }
        
        // Labels
        sb.append(String.format("<text x='%d' y='%d' text-anchor='middle' font-size='16' font-weight='bold'>%s</text>\n",
                offsetX + chartWidth / 2, 25, title));
        sb.append(String.format("<text x='%d' y='%d' text-anchor='middle' font-size='14'>%s</text>\n",
                offsetX + chartWidth / 2, offsetY + chartHeight + 40, xLabel));
        sb.append(String.format("<text x='20' y='%d' text-anchor='middle' font-size='14' transform='rotate(-90, 20, %d)'>%s</text>\n",
                offsetY + chartHeight / 2, offsetY + chartHeight / 2, yLabel));
        
        generateHtmlFooter(sb);
        writeToFile(sb.toString(), outputPath);
    }
    
    /**
     * Generate ROC curve chart.
     * 
     * @param fpr false positive rates
     * @param tpr true positive rates
     * @param auc area under curve
     * @param outputPath output HTML file path
     * @throws IOException if file cannot be written
     */
    public static void rocCurve(double[] fpr, double[] tpr, double auc,
                                 String outputPath) throws IOException {
        StringBuilder sb = new StringBuilder();
        generateHtmlHeader(sb, String.format("ROC Curve (AUC = %.4f)", auc), DEFAULT_WIDTH, DEFAULT_HEIGHT);
        
        int chartWidth = DEFAULT_WIDTH - 100;
        int chartHeight = DEFAULT_HEIGHT - 100;
        int offsetX = 60;
        int offsetY = 40;
        
        // Draw axes
        sb.append(String.format("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#333' stroke-width='2'/>\n",
                offsetX, offsetY, offsetX, offsetY + chartHeight));
        sb.append(String.format("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#333' stroke-width='2'/>\n",
                offsetX, offsetY + chartHeight, offsetX + chartWidth, offsetY + chartHeight));
        
        // Draw diagonal (random classifier line)
        sb.append(String.format("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke='#ccc' stroke-width='1' stroke-dasharray='5,5'/>\n",
                offsetX, offsetY + chartHeight, offsetX + chartWidth, offsetY));
        
        // Draw ROC curve
        StringBuilder pathData = new StringBuilder();
        for (int i = 0; i < fpr.length; i++) {
            int x = offsetX + (int) (fpr[i] * chartWidth);
            int y = offsetY + chartHeight - (int) (tpr[i] * chartHeight);
            
            if (i == 0) {
                pathData.append(String.format("M %d %d", x, y));
            } else {
                pathData.append(String.format(" L %d %d", x, y));
            }
        }
        
        sb.append(String.format("<path d='%s' stroke='%s' stroke-width='2' fill='none'/>\n",
                pathData, DEFAULT_COLORS[0]));
        
        // Labels
        sb.append(String.format("<text x='%d' y='%d' text-anchor='middle' font-size='14'>False Positive Rate</text>\n",
                offsetX + chartWidth / 2, offsetY + chartHeight + 40));
        sb.append(String.format("<text x='20' y='%d' text-anchor='middle' font-size='14' transform='rotate(-90, 20, %d)'>True Positive Rate</text>\n",
                offsetY + chartHeight / 2, offsetY + chartHeight / 2));
        
        generateHtmlFooter(sb);
        writeToFile(sb.toString(), outputPath);
    }
    
    private static void generateHtmlHeader(StringBuilder sb, String title, int width, int height) {
        sb.append("<!DOCTYPE html>\n<html>\n<head>\n");
        sb.append("<title>").append(title).append("</title>\n");
        sb.append("<style>body { font-family: Arial, sans-serif; margin: 20px; }</style>\n");
        sb.append("</head>\n<body>\n");
        sb.append(String.format("<svg width='%d' height='%d' xmlns='http://www.w3.org/2000/svg'>\n", width, height));
        sb.append("<rect width='100%' height='100%' fill='white'/>\n");
    }
    
    private static void generateHtmlFooter(StringBuilder sb) {
        sb.append("</svg>\n</body>\n</html>");
    }
    
    private static void writeToFile(String content, String path) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(path))) {
            writer.print(content);
        }
    }
}
