package io.github.yasmramos.mindforge.timeseries;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Exponential Smoothing methods for time series forecasting.
 * Supports Simple, Double (Holt), and Triple (Holt-Winters) exponential smoothing.
 */
public class ExponentialSmoothing implements Serializable {
    private static final long serialVersionUID = 1L;
    
    public enum Method {
        SIMPLE,      // Single exponential smoothing
        DOUBLE,      // Holt's linear trend
        TRIPLE       // Holt-Winters with seasonality
    }
    
    public enum SeasonalType {
        ADDITIVE,
        MULTIPLICATIVE
    }
    
    private final Method method;
    private final SeasonalType seasonalType;
    private final double alpha; // Level smoothing
    private final double beta;  // Trend smoothing
    private final double gamma; // Seasonal smoothing
    private final int seasonalPeriod;
    
    private double level;
    private double trend;
    private double[] seasonal;
    private double[] fittedValues;
    private double[] originalSeries;
    
    private ExponentialSmoothing(Builder builder) {
        this.method = builder.method;
        this.seasonalType = builder.seasonalType;
        this.alpha = builder.alpha;
        this.beta = builder.beta;
        this.gamma = builder.gamma;
        this.seasonalPeriod = builder.seasonalPeriod;
    }
    
    /**
     * Fit the exponential smoothing model to the time series data.
     * @param series The time series data
     */
    public void fit(double[] series) {
        this.originalSeries = series.clone();
        
        switch (method) {
            case SIMPLE:
                fitSimple(series);
                break;
            case DOUBLE:
                fitDouble(series);
                break;
            case TRIPLE:
                fitTriple(series);
                break;
        }
    }
    
    private void fitSimple(double[] series) {
        int n = series.length;
        fittedValues = new double[n];
        
        // Initialize level with first observation
        level = series[0];
        fittedValues[0] = level;
        
        for (int t = 1; t < n; t++) {
            double prevLevel = level;
            level = alpha * series[t] + (1 - alpha) * prevLevel;
            fittedValues[t] = prevLevel;
        }
    }
    
    private void fitDouble(double[] series) {
        int n = series.length;
        fittedValues = new double[n];
        
        // Initialize level and trend
        level = series[0];
        trend = series[1] - series[0];
        fittedValues[0] = level;
        fittedValues[1] = level + trend;
        
        for (int t = 2; t < n; t++) {
            double prevLevel = level;
            double prevTrend = trend;
            
            level = alpha * series[t] + (1 - alpha) * (prevLevel + prevTrend);
            trend = beta * (level - prevLevel) + (1 - beta) * prevTrend;
            
            fittedValues[t] = prevLevel + prevTrend;
        }
    }
    
    private void fitTriple(double[] series) {
        int n = series.length;
        int m = seasonalPeriod;
        
        if (n < 2 * m) {
            throw new IllegalArgumentException("Series too short for seasonal period. Need at least " + (2 * m) + " observations.");
        }
        
        fittedValues = new double[n];
        seasonal = new double[m];
        
        // Initialize level as average of first season
        level = 0;
        for (int i = 0; i < m; i++) {
            level += series[i];
        }
        level /= m;
        
        // Initialize trend
        double sum1 = 0, sum2 = 0;
        for (int i = 0; i < m; i++) {
            sum1 += series[i];
            sum2 += series[m + i];
        }
        trend = (sum2 - sum1) / (m * m);
        
        // Initialize seasonal components
        if (seasonalType == SeasonalType.ADDITIVE) {
            for (int i = 0; i < m; i++) {
                seasonal[i] = series[i] - level;
            }
        } else {
            for (int i = 0; i < m; i++) {
                seasonal[i] = series[i] / level;
            }
        }
        
        // Fit the model
        for (int t = 0; t < n; t++) {
            int seasonIndex = t % m;
            double prevLevel = level;
            double prevTrend = trend;
            double prevSeasonal = seasonal[seasonIndex];
            
            if (seasonalType == SeasonalType.ADDITIVE) {
                level = alpha * (series[t] - prevSeasonal) + (1 - alpha) * (prevLevel + prevTrend);
                trend = beta * (level - prevLevel) + (1 - beta) * prevTrend;
                seasonal[seasonIndex] = gamma * (series[t] - level) + (1 - gamma) * prevSeasonal;
                fittedValues[t] = prevLevel + prevTrend + prevSeasonal;
            } else {
                level = alpha * (series[t] / prevSeasonal) + (1 - alpha) * (prevLevel + prevTrend);
                trend = beta * (level - prevLevel) + (1 - beta) * prevTrend;
                seasonal[seasonIndex] = gamma * (series[t] / level) + (1 - gamma) * prevSeasonal;
                fittedValues[t] = (prevLevel + prevTrend) * prevSeasonal;
            }
        }
    }
    
    /**
     * Forecast future values.
     * @param steps Number of steps to forecast
     * @return Array of forecasted values
     */
    public double[] forecast(int steps) {
        if (fittedValues == null) {
            throw new IllegalStateException("Model must be fitted before forecasting");
        }
        
        double[] forecasts = new double[steps];
        
        switch (method) {
            case SIMPLE:
                Arrays.fill(forecasts, level);
                break;
            case DOUBLE:
                for (int h = 1; h <= steps; h++) {
                    forecasts[h - 1] = level + h * trend;
                }
                break;
            case TRIPLE:
                for (int h = 1; h <= steps; h++) {
                    int seasonIndex = (originalSeries.length + h - 1) % seasonalPeriod;
                    if (seasonalType == SeasonalType.ADDITIVE) {
                        forecasts[h - 1] = level + h * trend + seasonal[seasonIndex];
                    } else {
                        forecasts[h - 1] = (level + h * trend) * seasonal[seasonIndex];
                    }
                }
                break;
        }
        
        return forecasts;
    }
    
    /**
     * Get fitted values for the training data.
     * @return Array of fitted values
     */
    public double[] fittedValues() {
        return fittedValues != null ? fittedValues.clone() : null;
    }
    
    // Getters
    public double getLevel() { return level; }
    public double getTrend() { return trend; }
    public double[] getSeasonal() { return seasonal != null ? seasonal.clone() : null; }
    public double getAlpha() { return alpha; }
    public double getBeta() { return beta; }
    public double getGamma() { return gamma; }
    
    public static class Builder {
        private Method method = Method.SIMPLE;
        private SeasonalType seasonalType = SeasonalType.ADDITIVE;
        private double alpha = 0.3;
        private double beta = 0.1;
        private double gamma = 0.1;
        private int seasonalPeriod = 12;
        
        public Builder method(Method method) { this.method = method; return this; }
        public Builder seasonalType(SeasonalType type) { this.seasonalType = type; return this; }
        public Builder alpha(double alpha) { this.alpha = alpha; return this; }
        public Builder beta(double beta) { this.beta = beta; return this; }
        public Builder gamma(double gamma) { this.gamma = gamma; return this; }
        public Builder seasonalPeriod(int period) { this.seasonalPeriod = period; return this; }
        
        public ExponentialSmoothing build() { return new ExponentialSmoothing(this); }
    }
}
