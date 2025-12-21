package io.github.yasmramos.mindforge.timeseries;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Time Series forecasting classes.
 */
public class TimeSeriesTest {
    
    @Test
    public void testARIMAFitPredict() {
        // Simple time series with trend
        double[] series = new double[50];
        for (int i = 0; i < 50; i++) {
            series[i] = 10 + 0.5 * i + Math.sin(i * 0.5) * 2;
        }
        
        ARIMA arima = new ARIMA.Builder()
            .p(2)
            .d(1)
            .q(1)
            .build();
        
        arima.fit(series);
        
        double[] forecast = arima.forecast(5);
        assertNotNull(forecast);
        assertEquals(5, forecast.length);
        
        // Forecasts should continue the trend
        assertTrue(forecast[0] > series[series.length - 1] - 10);
    }
    
    @Test
    public void testARIMAFittedValues() {
        double[] series = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        
        ARIMA arima = new ARIMA.Builder()
            .p(1)
            .d(0)
            .q(0)
            .build();
        
        arima.fit(series);
        
        double[] fitted = arima.fittedValues();
        assertNotNull(fitted);
        assertEquals(series.length, fitted.length);
    }
    
    @Test
    public void testARIMAGetters() {
        ARIMA arima = new ARIMA.Builder()
            .p(2)
            .d(1)
            .q(3)
            .build();
        
        assertEquals(2, arima.getP());
        assertEquals(1, arima.getD());
        assertEquals(3, arima.getQ());
    }
    
    @Test
    public void testExponentialSmoothingSimple() {
        double[] series = {10, 12, 14, 13, 15, 17, 16, 18, 20, 19};
        
        ExponentialSmoothing es = new ExponentialSmoothing.Builder()
            .method(ExponentialSmoothing.Method.SIMPLE)
            .alpha(0.3)
            .build();
        
        es.fit(series);
        
        double[] forecast = es.forecast(3);
        assertNotNull(forecast);
        assertEquals(3, forecast.length);
        
        // All forecasts should be equal for simple ES
        assertEquals(forecast[0], forecast[1], 0.001);
        assertEquals(forecast[1], forecast[2], 0.001);
    }
    
    @Test
    public void testExponentialSmoothingDouble() {
        double[] series = new double[20];
        for (int i = 0; i < 20; i++) {
            series[i] = 10 + i * 2; // Linear trend
        }
        
        ExponentialSmoothing es = new ExponentialSmoothing.Builder()
            .method(ExponentialSmoothing.Method.DOUBLE)
            .alpha(0.8)
            .beta(0.2)
            .build();
        
        es.fit(series);
        
        double[] forecast = es.forecast(5);
        assertNotNull(forecast);
        assertEquals(5, forecast.length);
        
        // Forecasts should continue the trend
        assertTrue(forecast[4] > forecast[0]);
    }
    
    @Test
    public void testExponentialSmoothingTriple() {
        // Series with seasonality (period = 4)
        double[] series = new double[24];
        for (int i = 0; i < 24; i++) {
            series[i] = 100 + i * 2 + (i % 4) * 10;
        }
        
        ExponentialSmoothing es = new ExponentialSmoothing.Builder()
            .method(ExponentialSmoothing.Method.TRIPLE)
            .alpha(0.5)
            .beta(0.1)
            .gamma(0.3)
            .seasonalPeriod(4)
            .seasonalType(ExponentialSmoothing.SeasonalType.ADDITIVE)
            .build();
        
        es.fit(series);
        
        double[] forecast = es.forecast(4);
        assertNotNull(forecast);
        assertEquals(4, forecast.length);
    }
    
    @Test
    public void testExponentialSmoothingFittedValues() {
        double[] series = {10, 12, 14, 16, 18, 20};
        
        ExponentialSmoothing es = new ExponentialSmoothing.Builder()
            .method(ExponentialSmoothing.Method.SIMPLE)
            .alpha(0.5)
            .build();
        
        es.fit(series);
        
        double[] fitted = es.fittedValues();
        assertNotNull(fitted);
        assertEquals(series.length, fitted.length);
    }
    
    @Test
    public void testSimpleMovingAverageFit() {
        double[] series = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        
        SimpleMovingAverage sma = new SimpleMovingAverage(3);
        sma.fit(series);
        
        double[] smoothed = sma.getSmoothedSeries();
        assertNotNull(smoothed);
        assertEquals(8, smoothed.length); // n - window + 1
        
        // First average should be (1+2+3)/3 = 2
        assertEquals(2.0, smoothed[0], 0.001);
    }
    
    @Test
    public void testSimpleMovingAverageForecast() {
        double[] series = {10, 20, 30, 40, 50};
        
        SimpleMovingAverage sma = new SimpleMovingAverage(3);
        sma.fit(series);
        
        double[] forecast = sma.forecast(3);
        assertNotNull(forecast);
        assertEquals(3, forecast.length);
        
        // All forecasts should equal the last average
        double lastAvg = sma.getLastAverage();
        for (double f : forecast) {
            assertEquals(lastAvg, f, 0.001);
        }
    }
    
    @Test
    public void testSimpleMovingAverageTransform() {
        double[] series = {2, 4, 6, 8, 10};
        
        SimpleMovingAverage sma = new SimpleMovingAverage(2);
        double[] result = sma.transform(series);
        
        assertEquals(4, result.length);
        assertEquals(3.0, result[0], 0.001); // (2+4)/2
        assertEquals(5.0, result[1], 0.001); // (4+6)/2
    }
    
    @Test
    public void testSimpleMovingAverageInvalidWindow() {
        assertThrows(IllegalArgumentException.class, () -> new SimpleMovingAverage(0));
    }
    
    @Test
    public void testSimpleMovingAverageTooShortSeries() {
        SimpleMovingAverage sma = new SimpleMovingAverage(5);
        double[] series = {1, 2, 3};
        
        assertThrows(IllegalArgumentException.class, () -> sma.fit(series));
    }
}
