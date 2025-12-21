package io.github.yasmramos.mindforge.timeseries;

import java.io.Serializable;

/**
 * Simple Moving Average for time series smoothing and forecasting.
 */
public class SimpleMovingAverage implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int window;
    private double[] originalSeries;
    private double[] smoothedSeries;
    private double lastAverage;
    
    public SimpleMovingAverage(int window) {
        if (window < 1) {
            throw new IllegalArgumentException("Window must be at least 1");
        }
        this.window = window;
    }
    
    /**
     * Fit the moving average to the time series data.
     * @param series The time series data
     */
    public void fit(double[] series) {
        if (series.length < window) {
            throw new IllegalArgumentException("Series length must be at least equal to window size");
        }
        
        this.originalSeries = series.clone();
        int n = series.length;
        smoothedSeries = new double[n - window + 1];
        
        // Calculate first moving average
        double sum = 0;
        for (int i = 0; i < window; i++) {
            sum += series[i];
        }
        smoothedSeries[0] = sum / window;
        
        // Calculate remaining using sliding window
        for (int i = 1; i < smoothedSeries.length; i++) {
            sum = sum - series[i - 1] + series[i + window - 1];
            smoothedSeries[i] = sum / window;
        }
        
        lastAverage = smoothedSeries[smoothedSeries.length - 1];
    }
    
    /**
     * Forecast future values.
     * Simple MA forecasts the last average for all future steps.
     * @param steps Number of steps to forecast
     * @return Array of forecasted values
     */
    public double[] forecast(int steps) {
        if (smoothedSeries == null) {
            throw new IllegalStateException("Model must be fitted before forecasting");
        }
        
        double[] forecasts = new double[steps];
        for (int i = 0; i < steps; i++) {
            forecasts[i] = lastAverage;
        }
        return forecasts;
    }
    
    /**
     * Get smoothed values.
     * @return Array of smoothed values
     */
    public double[] getSmoothedSeries() {
        return smoothedSeries != null ? smoothedSeries.clone() : null;
    }
    
    /**
     * Transform a series using the moving average.
     * @param series The series to transform
     * @return Smoothed series
     */
    public double[] transform(double[] series) {
        if (series.length < window) {
            throw new IllegalArgumentException("Series length must be at least equal to window size");
        }
        
        int n = series.length;
        double[] result = new double[n - window + 1];
        
        double sum = 0;
        for (int i = 0; i < window; i++) {
            sum += series[i];
        }
        result[0] = sum / window;
        
        for (int i = 1; i < result.length; i++) {
            sum = sum - series[i - 1] + series[i + window - 1];
            result[i] = sum / window;
        }
        
        return result;
    }
    
    public int getWindow() { return window; }
    public double getLastAverage() { return lastAverage; }
}
