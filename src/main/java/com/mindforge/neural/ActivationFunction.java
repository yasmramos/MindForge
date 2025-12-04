package com.mindforge.neural;

/**
 * Activation functions for neural networks.
 */
public enum ActivationFunction {
    
    SIGMOID {
        @Override
        public double apply(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
        
        @Override
        public double derivative(double x) {
            double sigmoid = apply(x);
            return sigmoid * (1.0 - sigmoid);
        }
    },
    
    TANH {
        @Override
        public double apply(double x) {
            return Math.tanh(x);
        }
        
        @Override
        public double derivative(double x) {
            double tanh = Math.tanh(x);
            return 1.0 - tanh * tanh;
        }
    },
    
    RELU {
        @Override
        public double apply(double x) {
            return Math.max(0, x);
        }
        
        @Override
        public double derivative(double x) {
            return x > 0 ? 1.0 : 0.0;
        }
    },
    
    LEAKY_RELU {
        private static final double ALPHA = 0.01;
        
        @Override
        public double apply(double x) {
            return x > 0 ? x : ALPHA * x;
        }
        
        @Override
        public double derivative(double x) {
            return x > 0 ? 1.0 : ALPHA;
        }
    },
    
    ELU {
        private static final double ALPHA = 1.0;
        
        @Override
        public double apply(double x) {
            return x > 0 ? x : ALPHA * (Math.exp(x) - 1);
        }
        
        @Override
        public double derivative(double x) {
            return x > 0 ? 1.0 : apply(x) + ALPHA;
        }
    },
    
    SOFTPLUS {
        @Override
        public double apply(double x) {
            return Math.log(1 + Math.exp(x));
        }
        
        @Override
        public double derivative(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
    },
    
    SWISH {
        @Override
        public double apply(double x) {
            return x * SIGMOID.apply(x);
        }
        
        @Override
        public double derivative(double x) {
            double sigmoid = SIGMOID.apply(x);
            return sigmoid + x * sigmoid * (1 - sigmoid);
        }
    },
    
    LINEAR {
        @Override
        public double apply(double x) {
            return x;
        }
        
        @Override
        public double derivative(double x) {
            return 1.0;
        }
    },
    
    SOFTMAX {
        @Override
        public double apply(double x) {
            // Softmax is applied to vectors, not scalars
            // This is a placeholder for compatibility
            return Math.exp(x);
        }
        
        @Override
        public double derivative(double x) {
            double softmax = apply(x);
            return softmax * (1 - softmax);
        }
    };
    
    /**
     * Apply the activation function.
     * 
     * @param x input value
     * @return activated value
     */
    public abstract double apply(double x);
    
    /**
     * Calculate the derivative of the activation function.
     * 
     * @param x input value
     * @return derivative at x
     */
    public abstract double derivative(double x);
    
    /**
     * Apply activation to an array.
     * 
     * @param x input array
     * @return activated array
     */
    public double[] apply(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = apply(x[i]);
        }
        return result;
    }
    
    /**
     * Apply softmax to an array (for SOFTMAX activation).
     * 
     * @param x input array
     * @return softmax probabilities
     */
    public static double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            if (v > max) max = v;
        }
        
        double sum = 0.0;
        double[] exp = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            exp[i] = Math.exp(x[i] - max);
            sum += exp[i];
        }
        
        for (int i = 0; i < x.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }
}
