package io.github.yasmramos.mindforge.neural;

import java.io.Serializable;
import java.util.Random;

/**
 * 2D Convolutional layer for Convolutional Neural Networks.
 * 
 * Applies learnable filters to input feature maps to detect spatial patterns.
 * Supports configurable kernel size, stride, and padding.
 * 
 * Input shape: [channels, height, width] flattened to 1D array
 * Output shape: [numFilters, outHeight, outWidth] flattened to 1D array
 * 
 * @author MindForge Team
 * @version 1.2.0
 */
public class Conv2DLayer extends Layer implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // Layer configuration
    private final int inputChannels;
    private final int inputHeight;
    private final int inputWidth;
    private final int numFilters;
    private final int kernelSize;
    private final int stride;
    private final int padding;
    private final ActivationFunction activation;
    
    // Output dimensions
    private final int outputHeight;
    private final int outputWidth;
    
    // Learnable parameters
    private double[][][][] filters;  // [numFilters][inputChannels][kernelSize][kernelSize]
    private double[] biases;         // [numFilters]
    
    // Momentum for optimization
    private double[][][][] filterMomentum;
    private double[] biasMomentum;
    private double momentum = 0.9;
    
    // Cache for backpropagation
    private double[][][] lastInput3D;   // [channels][height][width]
    private double[][][] preActivation; // [numFilters][outHeight][outWidth]
    private double[][][] lastOutput3D;  // [numFilters][outHeight][outWidth]
    
    /**
     * Creates a 2D convolutional layer.
     *
     * @param inputChannels number of input channels
     * @param inputHeight height of input feature maps
     * @param inputWidth width of input feature maps
     * @param numFilters number of convolutional filters
     * @param kernelSize size of the square kernel
     * @param stride stride of the convolution
     * @param padding zero-padding added to both sides
     * @param activation activation function
     */
    public Conv2DLayer(int inputChannels, int inputHeight, int inputWidth,
                       int numFilters, int kernelSize, int stride, int padding,
                       ActivationFunction activation) {
        this(inputChannels, inputHeight, inputWidth, numFilters, kernelSize,
             stride, padding, activation, System.currentTimeMillis());
    }
    
    /**
     * Creates a 2D convolutional layer with seeded random initialization.
     *
     * @param inputChannels number of input channels
     * @param inputHeight height of input feature maps
     * @param inputWidth width of input feature maps
     * @param numFilters number of convolutional filters
     * @param kernelSize size of the square kernel
     * @param stride stride of the convolution
     * @param padding zero-padding added to both sides
     * @param activation activation function
     * @param seed random seed for reproducibility
     */
    public Conv2DLayer(int inputChannels, int inputHeight, int inputWidth,
                       int numFilters, int kernelSize, int stride, int padding,
                       ActivationFunction activation, long seed) {
        validateParameters(inputChannels, inputHeight, inputWidth, numFilters, 
                          kernelSize, stride, padding);
        
        this.inputChannels = inputChannels;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.numFilters = numFilters;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.activation = activation;
        
        // Calculate output dimensions
        this.outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
        this.outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;
        
        // Set input/output sizes for Layer interface
        this.inputSize = inputChannels * inputHeight * inputWidth;
        this.outputSize = numFilters * outputHeight * outputWidth;
        
        // Initialize filters using He initialization (good for ReLU)
        initializeWeights(seed);
    }
    
    private void validateParameters(int inputChannels, int inputHeight, int inputWidth,
                                   int numFilters, int kernelSize, int stride, int padding) {
        if (inputChannels <= 0) {
            throw new IllegalArgumentException("inputChannels must be positive");
        }
        if (inputHeight <= 0 || inputWidth <= 0) {
            throw new IllegalArgumentException("Input dimensions must be positive");
        }
        if (numFilters <= 0) {
            throw new IllegalArgumentException("numFilters must be positive");
        }
        if (kernelSize <= 0) {
            throw new IllegalArgumentException("kernelSize must be positive");
        }
        if (stride <= 0) {
            throw new IllegalArgumentException("stride must be positive");
        }
        if (padding < 0) {
            throw new IllegalArgumentException("padding cannot be negative");
        }
        
        int outH = (inputHeight + 2 * padding - kernelSize) / stride + 1;
        int outW = (inputWidth + 2 * padding - kernelSize) / stride + 1;
        if (outH <= 0 || outW <= 0) {
            throw new IllegalArgumentException(
                "Invalid configuration: output dimensions would be non-positive");
        }
    }
    
    private void initializeWeights(long seed) {
        Random random = new Random(seed);
        
        // He initialization: scale = sqrt(2 / fan_in)
        int fanIn = inputChannels * kernelSize * kernelSize;
        double scale = Math.sqrt(2.0 / fanIn);
        
        filters = new double[numFilters][inputChannels][kernelSize][kernelSize];
        biases = new double[numFilters];
        filterMomentum = new double[numFilters][inputChannels][kernelSize][kernelSize];
        biasMomentum = new double[numFilters];
        
        for (int f = 0; f < numFilters; f++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int i = 0; i < kernelSize; i++) {
                    for (int j = 0; j < kernelSize; j++) {
                        filters[f][c][i][j] = random.nextGaussian() * scale;
                    }
                }
            }
            biases[f] = 0.0;
        }
    }
    
    @Override
    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException(
                "Input size mismatch: expected " + inputSize + ", got " + input.length);
        }
        
        // Reshape input to 3D
        lastInput3D = reshape1Dto3D(input, inputChannels, inputHeight, inputWidth);
        
        // Apply padding
        double[][][] paddedInput = applyPadding(lastInput3D);
        
        // Perform convolution
        preActivation = new double[numFilters][outputHeight][outputWidth];
        lastOutput3D = new double[numFilters][outputHeight][outputWidth];
        
        for (int f = 0; f < numFilters; f++) {
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    double sum = biases[f];
                    
                    for (int c = 0; c < inputChannels; c++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                sum += paddedInput[c][ih][iw] * filters[f][c][kh][kw];
                            }
                        }
                    }
                    
                    preActivation[f][oh][ow] = sum;
                    lastOutput3D[f][oh][ow] = activation.apply(sum);
                }
            }
        }
        
        // Flatten output to 1D
        lastOutput = reshape3Dto1D(lastOutput3D);
        return lastOutput;
    }
    
    @Override
    public double[] backward(double[] gradOutput, double learningRate) {
        // Reshape gradient to 3D
        double[][][] gradOutput3D = reshape1Dto3D(gradOutput, numFilters, outputHeight, outputWidth);
        
        // Apply activation derivative
        double[][][] gradActivation = new double[numFilters][outputHeight][outputWidth];
        for (int f = 0; f < numFilters; f++) {
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    gradActivation[f][oh][ow] = gradOutput3D[f][oh][ow] * 
                        activation.derivative(preActivation[f][oh][ow]);
                }
            }
        }
        
        // Compute gradients for filters and biases
        double[][][][] gradFilters = new double[numFilters][inputChannels][kernelSize][kernelSize];
        double[] gradBiases = new double[numFilters];
        
        // Apply padding to input for gradient computation
        double[][][] paddedInput = applyPadding(lastInput3D);
        
        for (int f = 0; f < numFilters; f++) {
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    double grad = gradActivation[f][oh][ow];
                    gradBiases[f] += grad;
                    
                    for (int c = 0; c < inputChannels; c++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                gradFilters[f][c][kh][kw] += grad * paddedInput[c][ih][iw];
                            }
                        }
                    }
                }
            }
        }
        
        // Compute gradient with respect to input
        int paddedHeight = inputHeight + 2 * padding;
        int paddedWidth = inputWidth + 2 * padding;
        double[][][] gradPaddedInput = new double[inputChannels][paddedHeight][paddedWidth];
        
        for (int f = 0; f < numFilters; f++) {
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    double grad = gradActivation[f][oh][ow];
                    
                    for (int c = 0; c < inputChannels; c++) {
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                gradPaddedInput[c][ih][iw] += grad * filters[f][c][kh][kw];
                            }
                        }
                    }
                }
            }
        }
        
        // Remove padding from gradient
        double[][][] gradInput3D = removePadding(gradPaddedInput);
        
        // Update weights with momentum
        for (int f = 0; f < numFilters; f++) {
            for (int c = 0; c < inputChannels; c++) {
                for (int kh = 0; kh < kernelSize; kh++) {
                    for (int kw = 0; kw < kernelSize; kw++) {
                        filterMomentum[f][c][kh][kw] = momentum * filterMomentum[f][c][kh][kw] +
                            learningRate * gradFilters[f][c][kh][kw];
                        filters[f][c][kh][kw] -= filterMomentum[f][c][kh][kw];
                    }
                }
            }
            biasMomentum[f] = momentum * biasMomentum[f] + learningRate * gradBiases[f];
            biases[f] -= biasMomentum[f];
        }
        
        return reshape3Dto1D(gradInput3D);
    }
    
    // Helper methods for tensor operations
    
    private double[][][] reshape1Dto3D(double[] array, int channels, int height, int width) {
        double[][][] result = new double[channels][height][width];
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result[c][h][w] = array[idx++];
                }
            }
        }
        return result;
    }
    
    private double[] reshape3Dto1D(double[][][] tensor) {
        int channels = tensor.length;
        int height = tensor[0].length;
        int width = tensor[0][0].length;
        double[] result = new double[channels * height * width];
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    result[idx++] = tensor[c][h][w];
                }
            }
        }
        return result;
    }
    
    private double[][][] applyPadding(double[][][] input) {
        if (padding == 0) {
            return input;
        }
        
        int channels = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        
        double[][][] padded = new double[channels][height + 2 * padding][width + 2 * padding];
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    padded[c][h + padding][w + padding] = input[c][h][w];
                }
            }
        }
        
        return padded;
    }
    
    private double[][][] removePadding(double[][][] padded) {
        if (padding == 0) {
            return padded;
        }
        
        int channels = padded.length;
        double[][][] result = new double[channels][inputHeight][inputWidth];
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++) {
                    result[c][h][w] = padded[c][h + padding][w + padding];
                }
            }
        }
        
        return result;
    }
    
    // Getters
    
    public int getInputChannels() { return inputChannels; }
    public int getInputHeight() { return inputHeight; }
    public int getInputWidth() { return inputWidth; }
    public int getNumFilters() { return numFilters; }
    public int getKernelSize() { return kernelSize; }
    public int getStride() { return stride; }
    public int getPadding() { return padding; }
    public int getOutputHeight() { return outputHeight; }
    public int getOutputWidth() { return outputWidth; }
    public ActivationFunction getActivation() { return activation; }
    
    public double[][][][] getFilters() { return filters; }
    public double[] getBiases() { return biases; }
    
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }
    
    /**
     * Get total number of trainable parameters.
     * 
     * @return number of parameters
     */
    public int getNumParameters() {
        return numFilters * inputChannels * kernelSize * kernelSize + numFilters;
    }
    
    @Override
    public String toString() {
        return String.format("Conv2D(in=%d, filters=%d, kernel=%dx%d, stride=%d, padding=%d) -> [%d, %d, %d]",
            inputChannels, numFilters, kernelSize, kernelSize, stride, padding,
            numFilters, outputHeight, outputWidth);
    }
}
