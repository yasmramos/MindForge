package io.github.yasmramos.mindforge.timeseries;

import java.io.Serializable;
import java.util.Arrays;

/**
 * AutoRegressive Integrated Moving Average (ARIMA) model for time series forecasting.
 * ARIMA(p, d, q) where:
 * - p: order of autoregressive part
 * - d: degree of differencing
 * - q: order of moving average part
 */
public class ARIMA implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int p; // AR order
    private final int d; // Differencing order
    private final int q; // MA order
    private final int maxIterations;
    private final double tol;
    
    private double[] arCoefficients;
    private double[] maCoefficients;
    private double intercept;
    private double[] residuals;
    private double[] differencedSeries;
    private double[] originalSeries;
    
    private ARIMA(Builder builder) {
        this.p = builder.p;
        this.d = builder.d;
        this.q = builder.q;
        this.maxIterations = builder.maxIterations;
        this.tol = builder.tol;
    }
    
    /**
     * Fit the ARIMA model to the time series data.
     * @param series The time series data
     */
    public void fit(double[] series) {
        this.originalSeries = series.clone();
        
        // Apply differencing
        this.differencedSeries = difference(series, d);
        
        // Estimate AR coefficients using Yule-Walker equations
        if (p > 0) {
            arCoefficients = estimateARCoefficients(differencedSeries, p);
        } else {
            arCoefficients = new double[0];
        }
        
        // Calculate residuals from AR model
        residuals = calculateResiduals(differencedSeries, arCoefficients);
        
        // Estimate MA coefficients
        if (q > 0) {
            maCoefficients = estimateMACoefficients(residuals, q);
        } else {
            maCoefficients = new double[0];
        }
        
        // Calculate intercept
        intercept = calculateMean(differencedSeries);
    }
    
    /**
     * Forecast future values.
     * @param steps Number of steps to forecast
     * @return Array of forecasted values
     */
    public double[] forecast(int steps) {
        if (differencedSeries == null) {
            throw new IllegalStateException("Model must be fitted before forecasting");
        }
        
        double[] forecasts = new double[steps];
        double[] extendedSeries = Arrays.copyOf(differencedSeries, differencedSeries.length + steps);
        double[] extendedResiduals = Arrays.copyOf(residuals, residuals.length + steps);
        
        for (int t = 0; t < steps; t++) {
            int currentIndex = differencedSeries.length + t;
            double forecast = intercept;
            
            // AR component
            for (int i = 0; i < p && i < currentIndex; i++) {
                forecast += arCoefficients[i] * extendedSeries[currentIndex - 1 - i];
            }
            
            // MA component
            for (int i = 0; i < q && i < currentIndex; i++) {
                forecast += maCoefficients[i] * extendedResiduals[currentIndex - 1 - i];
            }
            
            extendedSeries[currentIndex] = forecast;
            extendedResiduals[currentIndex] = 0; // Assume zero residual for forecasts
            forecasts[t] = forecast;
        }
        
        // Reverse differencing
        return inverseDifference(forecasts, originalSeries, d);
    }
    
    /**
     * Get fitted values for the training data.
     * @return Array of fitted values
     */
    public double[] fittedValues() {
        if (differencedSeries == null) {
            throw new IllegalStateException("Model must be fitted first");
        }
        
        double[] fitted = new double[differencedSeries.length];
        
        for (int t = 0; t < differencedSeries.length; t++) {
            double value = intercept;
            
            // AR component
            for (int i = 0; i < p && i < t; i++) {
                value += arCoefficients[i] * differencedSeries[t - 1 - i];
            }
            
            // MA component
            for (int i = 0; i < q && i < t; i++) {
                value += maCoefficients[i] * residuals[t - 1 - i];
            }
            
            fitted[t] = value;
        }
        
        return inverseDifferenceInPlace(fitted, originalSeries, d);
    }
    
    private double[] difference(double[] series, int order) {
        double[] result = series.clone();
        for (int i = 0; i < order; i++) {
            double[] temp = new double[result.length - 1];
            for (int j = 1; j < result.length; j++) {
                temp[j - 1] = result[j] - result[j - 1];
            }
            result = temp;
        }
        return result;
    }
    
    private double[] inverseDifference(double[] forecasts, double[] original, int order) {
        if (order == 0) {
            return forecasts;
        }
        
        double[] result = forecasts.clone();
        double lastValue = original[original.length - 1];
        
        for (int d = 0; d < order; d++) {
            for (int i = 0; i < result.length; i++) {
                result[i] = result[i] + lastValue;
                lastValue = result[i];
            }
            lastValue = original[original.length - 1];
        }
        
        return result;
    }
    
    private double[] inverseDifferenceInPlace(double[] fitted, double[] original, int order) {
        if (order == 0) {
            return fitted;
        }
        
        double[] result = new double[original.length];
        int offset = original.length - fitted.length;
        
        for (int i = 0; i < offset; i++) {
            result[i] = original[i];
        }
        
        for (int i = 0; i < fitted.length; i++) {
            result[offset + i] = fitted[i] + original[offset + i - 1];
        }
        
        return result;
    }
    
    private double[] estimateARCoefficients(double[] series, int order) {
        // Yule-Walker equations
        double[] r = new double[order + 1];
        int n = series.length;
        double mean = calculateMean(series);
        
        // Calculate autocorrelations
        for (int k = 0; k <= order; k++) {
            double sum = 0;
            for (int t = k; t < n; t++) {
                sum += (series[t] - mean) * (series[t - k] - mean);
            }
            r[k] = sum / n;
        }
        
        // Solve Yule-Walker equations using Levinson-Durbin recursion
        double[] phi = new double[order];
        double[] phiTemp = new double[order];
        
        phi[0] = r[1] / r[0];
        
        for (int k = 1; k < order; k++) {
            double num = r[k + 1];
            for (int j = 0; j < k; j++) {
                num -= phi[j] * r[k - j];
            }
            
            double den = r[0];
            for (int j = 0; j < k; j++) {
                den -= phi[j] * r[j + 1];
            }
            
            phiTemp[k] = num / den;
            
            for (int j = 0; j < k; j++) {
                phiTemp[j] = phi[j] - phiTemp[k] * phi[k - 1 - j];
            }
            
            System.arraycopy(phiTemp, 0, phi, 0, k + 1);
        }
        
        return phi;
    }
    
    private double[] estimateMACoefficients(double[] residuals, int order) {
        // Simple estimation using autocorrelation of residuals
        double[] theta = new double[order];
        double mean = calculateMean(residuals);
        int n = residuals.length;
        
        double var = 0;
        for (double r : residuals) {
            var += (r - mean) * (r - mean);
        }
        var /= n;
        
        for (int k = 0; k < order; k++) {
            double cov = 0;
            for (int t = k + 1; t < n; t++) {
                cov += (residuals[t] - mean) * (residuals[t - k - 1] - mean);
            }
            theta[k] = (cov / n) / var;
        }
        
        return theta;
    }
    
    private double[] calculateResiduals(double[] series, double[] arCoef) {
        double[] residuals = new double[series.length];
        double mean = calculateMean(series);
        
        for (int t = 0; t < series.length; t++) {
            double predicted = mean;
            for (int i = 0; i < arCoef.length && i < t; i++) {
                predicted += arCoef[i] * (series[t - 1 - i] - mean);
            }
            residuals[t] = series[t] - predicted;
        }
        
        return residuals;
    }
    
    private double calculateMean(double[] data) {
        double sum = 0;
        for (double v : data) {
            sum += v;
        }
        return sum / data.length;
    }
    
    // Getters
    public double[] getARCoefficients() { return arCoefficients != null ? arCoefficients.clone() : null; }
    public double[] getMACoefficients() { return maCoefficients != null ? maCoefficients.clone() : null; }
    public double getIntercept() { return intercept; }
    public int getP() { return p; }
    public int getD() { return d; }
    public int getQ() { return q; }
    
    public static class Builder {
        private int p = 1;
        private int d = 0;
        private int q = 1;
        private int maxIterations = 100;
        private double tol = 1e-6;
        
        public Builder p(int p) { this.p = p; return this; }
        public Builder d(int d) { this.d = d; return this; }
        public Builder q(int q) { this.q = q; return this; }
        public Builder maxIterations(int maxIterations) { this.maxIterations = maxIterations; return this; }
        public Builder tol(double tol) { this.tol = tol; return this; }
        
        public ARIMA build() { return new ARIMA(this); }
    }
}
