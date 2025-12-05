package com.mindforge.neural;

import java.io.Serializable;

/**
 * 2D Average Pooling layer for Convolutional Neural Networks.
 * 
 * Performs average pooling operation over spatial dimensions to reduce 
 * the spatial size of feature maps by computing the average value in each window.
 * 
 * Input shape: [channels, height, width] flattened to 1D array
 * Output shape: [channels, outHeight, outWidth] flattened to 1D array
 * 
 * @author MindForge Team
 * @version 1.2.0
 */
public class AveragePooling2DLayer extends Layer implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // Layer configuration
    private final int inputChannels;
    private final int inputHeight;
    private final int inputWidth;
    private final int poolSize;
    private final int stride;
    
    // Output dimensions
    private final int outputHeight;
    private final int outputWidth;
    
    /**
     * Creates a 2D average pooling layer.
     *
     * @param inputChannels number of input channels
     * @param inputHeight height of input feature maps
     * @param inputWidth width of input feature maps
     * @param poolSize size of the pooling window (square)
     */
    public AveragePooling2DLayer(int inputChannels, int inputHeight, int inputWidth, int poolSize) {
        this(inputChannels, inputHeight, inputWidth, poolSize, poolSize);
    }
    
    /**
     * Creates a 2D average pooling layer with custom stride.
     *
     * @param inputChannels number of input channels
     * @param inputHeight height of input feature maps
     * @param inputWidth width of input feature maps
     * @param poolSize size of the pooling window (square)
     * @param stride stride of the pooling operation
     */
    public AveragePooling2DLayer(int inputChannels, int inputHeight, int inputWidth,
                                 int poolSize, int stride) {
        validateParameters(inputChannels, inputHeight, inputWidth, poolSize, stride);
        
        this.inputChannels = inputChannels;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.poolSize = poolSize;
        this.stride = stride;
        
        // Calculate output dimensions
        this.outputHeight = (inputHeight - poolSize) / stride + 1;
        this.outputWidth = (inputWidth - poolSize) / stride + 1;
        
        // Set input/output sizes for Layer interface
        this.inputSize = inputChannels * inputHeight * inputWidth;
        this.outputSize = inputChannels * outputHeight * outputWidth;
    }
    
    private void validateParameters(int inputChannels, int inputHeight, int inputWidth,
                                   int poolSize, int stride) {
        if (inputChannels <= 0) {
            throw new IllegalArgumentException("inputChannels must be positive");
        }
        if (inputHeight <= 0 || inputWidth <= 0) {
            throw new IllegalArgumentException("Input dimensions must be positive");
        }
        if (poolSize <= 0) {
            throw new IllegalArgumentException("poolSize must be positive");
        }
        if (stride <= 0) {
            throw new IllegalArgumentException("stride must be positive");
        }
        if (poolSize > inputHeight || poolSize > inputWidth) {
            throw new IllegalArgumentException("poolSize cannot be larger than input dimensions");
        }
    }
    
    @Override
    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException(
                "Input size mismatch: expected " + inputSize + ", got " + input.length);
        }
        
        // Reshape input to 3D
        double[][][] input3D = reshape1Dto3D(input, inputChannels, inputHeight, inputWidth);
        
        // Perform average pooling
        double[][][] output3D = new double[inputChannels][outputHeight][outputWidth];
        double poolArea = poolSize * poolSize;
        
        for (int c = 0; c < inputChannels; c++) {
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    double sum = 0.0;
                    
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            sum += input3D[c][ih][iw];
                        }
                    }
                    
                    output3D[c][oh][ow] = sum / poolArea;
                }
            }
        }
        
        // Flatten output to 1D
        lastOutput = reshape3Dto1D(output3D);
        lastInput = input.clone();
        return lastOutput;
    }
    
    @Override
    public double[] backward(double[] gradOutput, double learningRate) {
        // Reshape gradient to 3D
        double[][][] gradOutput3D = reshape1Dto3D(gradOutput, inputChannels, outputHeight, outputWidth);
        
        // Initialize gradient for input
        double[][][] gradInput3D = new double[inputChannels][inputHeight][inputWidth];
        double poolArea = poolSize * poolSize;
        
        // Distribute gradient evenly to all positions in the pooling window
        for (int c = 0; c < inputChannels; c++) {
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    double gradShare = gradOutput3D[c][oh][ow] / poolArea;
                    
                    for (int ph = 0; ph < poolSize; ph++) {
                        for (int pw = 0; pw < poolSize; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            gradInput3D[c][ih][iw] += gradShare;
                        }
                    }
                }
            }
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
    
    // Getters
    
    public int getInputChannels() { return inputChannels; }
    public int getInputHeight() { return inputHeight; }
    public int getInputWidth() { return inputWidth; }
    public int getPoolSize() { return poolSize; }
    public int getStride() { return stride; }
    public int getOutputHeight() { return outputHeight; }
    public int getOutputWidth() { return outputWidth; }
    
    @Override
    public String toString() {
        return String.format("AveragePooling2D(pool=%dx%d, stride=%d) -> [%d, %d, %d]",
            poolSize, poolSize, stride, inputChannels, outputHeight, outputWidth);
    }
}
