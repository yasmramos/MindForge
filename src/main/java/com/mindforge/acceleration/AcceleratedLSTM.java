package com.mindforge.acceleration;

import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * High-performance LSTM operations.
 * 
 * Provides optimized implementations of LSTM cell computations including
 * fused gate operations, batched processing, and parallelized sequence handling.
 * 
 * <p>Key optimizations:</p>
 * <ul>
 *   <li>Fused gate computations to reduce memory bandwidth</li>
 *   <li>Batched matrix operations for multiple sequences</li>
 *   <li>Parallel processing of independent sequence elements</li>
 *   <li>Vectorized activation functions</li>
 * </ul>
 * 
 * @author MindForge Team
 * @version 1.2.0
 */
public class AcceleratedLSTM {
    
    private static final AccelerationConfig config = AccelerationConfig.getInstance();
    private static final MemoryPool pool = MemoryPool.getInstance();
    
    // Private constructor - utility class
    private AcceleratedLSTM() {}
    
    // ==================== Single Cell Operations ====================
    
    /**
     * Compute LSTM cell forward pass with fused gate operations.
     * 
     * Combines all gate computations into a single pass for better cache utilization.
     * 
     * @param input input vector [inputSize]
     * @param prevHidden previous hidden state [hiddenSize]
     * @param prevCell previous cell state [hiddenSize]
     * @param weightsIH input-to-hidden weights [4*hiddenSize][inputSize] (i,f,g,o gates)
     * @param weightsHH hidden-to-hidden weights [4*hiddenSize][hiddenSize]
     * @param biases biases [4*hiddenSize]
     * @return array containing [newHidden, newCell, gates] where gates is [4*hiddenSize]
     */
    public static double[][] lstmCellForward(double[] input, double[] prevHidden, double[] prevCell,
                                              double[][] weightsIH, double[][] weightsHH, double[] biases) {
        int hiddenSize = prevHidden.length;
        int inputSize = input.length;
        int gateSize = 4 * hiddenSize;
        
        // Allocate outputs - use exact sizes to avoid dimension issues
        double[] gates = new double[gateSize];
        double[] newHidden = new double[hiddenSize];
        double[] newCell = new double[hiddenSize];
        
        // Fused gate computation: gates = W_ih * x + W_hh * h + b
        // Process all 4 gates (input, forget, cell, output) together
        if (config.shouldParallelize(gateSize * (inputSize + hiddenSize))) {
            computeGatesParallel(input, prevHidden, weightsIH, weightsHH, biases, gates, 
                                hiddenSize, inputSize);
        } else {
            computeGatesSequential(input, prevHidden, weightsIH, weightsHH, biases, gates,
                                  hiddenSize, inputSize);
        }
        
        // Apply gate activations and compute new states
        for (int i = 0; i < hiddenSize; i++) {
            // Input gate (sigmoid)
            double inputGate = sigmoid(gates[i]);
            
            // Forget gate (sigmoid)
            double forgetGate = sigmoid(gates[hiddenSize + i]);
            
            // Cell gate (tanh)
            double cellGate = Math.tanh(gates[2 * hiddenSize + i]);
            
            // Output gate (sigmoid)
            double outputGate = sigmoid(gates[3 * hiddenSize + i]);
            
            // New cell state
            newCell[i] = forgetGate * prevCell[i] + inputGate * cellGate;
            
            // New hidden state
            newHidden[i] = outputGate * Math.tanh(newCell[i]);
            
            // Store activated gates for backward pass
            gates[i] = inputGate;
            gates[hiddenSize + i] = forgetGate;
            gates[2 * hiddenSize + i] = cellGate;
            gates[3 * hiddenSize + i] = outputGate;
        }
        
        return new double[][] { newHidden, newCell, gates };
    }
    
    private static void computeGatesSequential(double[] input, double[] prevHidden,
                                                double[][] weightsIH, double[][] weightsHH,
                                                double[] biases, double[] gates,
                                                int hiddenSize, int inputSize) {
        int gateSize = 4 * hiddenSize;
        
        // Initialize with biases
        System.arraycopy(biases, 0, gates, 0, gateSize);
        
        // Add input contribution: W_ih * x
        for (int g = 0; g < gateSize; g++) {
            double sum = 0.0;
            double[] wRow = weightsIH[g];
            
            // Unrolled loop
            int j = 0;
            int limit = inputSize - 4 + 1;
            for (; j < limit; j += 4) {
                sum += wRow[j] * input[j] + wRow[j + 1] * input[j + 1] +
                       wRow[j + 2] * input[j + 2] + wRow[j + 3] * input[j + 3];
            }
            for (; j < inputSize; j++) {
                sum += wRow[j] * input[j];
            }
            
            gates[g] += sum;
        }
        
        // Add hidden contribution: W_hh * h
        for (int g = 0; g < gateSize; g++) {
            double sum = 0.0;
            double[] wRow = weightsHH[g];
            
            int j = 0;
            int limit = hiddenSize - 4 + 1;
            for (; j < limit; j += 4) {
                sum += wRow[j] * prevHidden[j] + wRow[j + 1] * prevHidden[j + 1] +
                       wRow[j + 2] * prevHidden[j + 2] + wRow[j + 3] * prevHidden[j + 3];
            }
            for (; j < hiddenSize; j++) {
                sum += wRow[j] * prevHidden[j];
            }
            
            gates[g] += sum;
        }
    }
    
    private static void computeGatesParallel(double[] input, double[] prevHidden,
                                              double[][] weightsIH, double[][] weightsHH,
                                              double[] biases, double[] gates,
                                              int hiddenSize, int inputSize) {
        int gateSize = 4 * hiddenSize;
        ForkJoinPool executor = config.getExecutor();
        
        executor.submit(() ->
            IntStream.range(0, gateSize).parallel().forEach(g -> {
                double sum = biases[g];
                
                // Input contribution
                double[] wihRow = weightsIH[g];
                for (int j = 0; j < inputSize; j++) {
                    sum += wihRow[j] * input[j];
                }
                
                // Hidden contribution
                double[] whhRow = weightsHH[g];
                for (int j = 0; j < hiddenSize; j++) {
                    sum += whhRow[j] * prevHidden[j];
                }
                
                gates[g] = sum;
            })
        ).join();
    }
    
    /**
     * Compute LSTM cell backward pass.
     * 
     * @param gradHidden gradient w.r.t. hidden state [hiddenSize]
     * @param gradCell gradient w.r.t. cell state [hiddenSize]
     * @param input input that was used in forward pass [inputSize]
     * @param prevHidden previous hidden state [hiddenSize]
     * @param prevCell previous cell state [hiddenSize]
     * @param newCell new cell state from forward pass [hiddenSize]
     * @param gates activated gates from forward pass [4*hiddenSize]
     * @param weightsIH input-to-hidden weights [4*hiddenSize][inputSize]
     * @param weightsHH hidden-to-hidden weights [4*hiddenSize][hiddenSize]
     * @return array containing [gradInput, gradPrevHidden, gradPrevCell, gradWeightsIH, gradWeightsHH, gradBiases]
     */
    public static Object[] lstmCellBackward(double[] gradHidden, double[] gradCell,
                                            double[] input, double[] prevHidden, double[] prevCell,
                                            double[] newCell, double[] gates,
                                            double[][] weightsIH, double[][] weightsHH) {
        int hiddenSize = prevHidden.length;
        int inputSize = input.length;
        int gateSize = 4 * hiddenSize;
        
        // Allocate gradients - use exact sizes to avoid dimension issues
        double[] gradInput = new double[inputSize];
        double[] gradPrevHidden = new double[hiddenSize];
        double[] gradPrevCell = new double[hiddenSize];
        double[] gradGates = new double[gateSize];
        double[][] gradWeightsIH = new double[gateSize][inputSize];
        double[][] gradWeightsHH = new double[gateSize][hiddenSize];
        double[] gradBiases = new double[gateSize];
        
        // Extract gates
        // gates layout: [inputGate, forgetGate, cellGate, outputGate]
        
        for (int i = 0; i < hiddenSize; i++) {
            double inputGate = gates[i];
            double forgetGate = gates[hiddenSize + i];
            double cellGate = gates[2 * hiddenSize + i];
            double outputGate = gates[3 * hiddenSize + i];
            
            double tanhNewCell = Math.tanh(newCell[i]);
            
            // Gradient through output gate
            double gradOutputGate = gradHidden[i] * tanhNewCell;
            gradOutputGate *= outputGate * (1 - outputGate); // sigmoid derivative
            
            // Gradient through tanh(newCell)
            double gradTanhNewCell = gradHidden[i] * outputGate;
            double gradNewCell = gradTanhNewCell * (1 - tanhNewCell * tanhNewCell) + gradCell[i];
            
            // Gradient through cell state computation
            double gradInputGate = gradNewCell * cellGate;
            gradInputGate *= inputGate * (1 - inputGate); // sigmoid derivative
            
            double gradForgetGate = gradNewCell * prevCell[i];
            gradForgetGate *= forgetGate * (1 - forgetGate); // sigmoid derivative
            
            double gradCellGate = gradNewCell * inputGate;
            gradCellGate *= (1 - cellGate * cellGate); // tanh derivative
            
            gradPrevCell[i] = gradNewCell * forgetGate;
            
            // Store gate gradients
            gradGates[i] = gradInputGate;
            gradGates[hiddenSize + i] = gradForgetGate;
            gradGates[2 * hiddenSize + i] = gradCellGate;
            gradGates[3 * hiddenSize + i] = gradOutputGate;
            
            gradBiases[i] = gradInputGate;
            gradBiases[hiddenSize + i] = gradForgetGate;
            gradBiases[2 * hiddenSize + i] = gradCellGate;
            gradBiases[3 * hiddenSize + i] = gradOutputGate;
        }
        
        // Compute gradients for weights and inputs
        computeWeightGradients(gradGates, input, prevHidden, gradWeightsIH, gradWeightsHH,
                              gradInput, gradPrevHidden, weightsIH, weightsHH,
                              gateSize, inputSize, hiddenSize);
        
        return new Object[] { gradInput, gradPrevHidden, gradPrevCell, 
                             gradWeightsIH, gradWeightsHH, gradBiases };
    }
    
    private static void computeWeightGradients(double[] gradGates, double[] input, double[] prevHidden,
                                                double[][] gradWeightsIH, double[][] gradWeightsHH,
                                                double[] gradInput, double[] gradPrevHidden,
                                                double[][] weightsIH, double[][] weightsHH,
                                                int gateSize, int inputSize, int hiddenSize) {
        // Gradient for input weights: dW_ih = gradGates * input^T
        // Gradient for hidden weights: dW_hh = gradGates * prevHidden^T
        // Gradient for input: dx = W_ih^T * gradGates
        // Gradient for prevHidden: dh = W_hh^T * gradGates
        
        if (config.shouldParallelize(gateSize * (inputSize + hiddenSize))) {
            ForkJoinPool executor = config.getExecutor();
            
            executor.submit(() -> {
                // Weight gradients in parallel
                IntStream.range(0, gateSize).parallel().forEach(g -> {
                    double gradG = gradGates[g];
                    
                    for (int j = 0; j < inputSize; j++) {
                        gradWeightsIH[g][j] = gradG * input[j];
                    }
                    
                    for (int j = 0; j < hiddenSize; j++) {
                        gradWeightsHH[g][j] = gradG * prevHidden[j];
                    }
                });
            }).join();
            
            // Input and hidden gradients
            executor.submit(() -> {
                IntStream.range(0, inputSize).parallel().forEach(j -> {
                    double sum = 0.0;
                    for (int g = 0; g < gateSize; g++) {
                        sum += weightsIH[g][j] * gradGates[g];
                    }
                    gradInput[j] = sum;
                });
            }).join();
            
            executor.submit(() -> {
                IntStream.range(0, hiddenSize).parallel().forEach(j -> {
                    double sum = 0.0;
                    for (int g = 0; g < gateSize; g++) {
                        sum += weightsHH[g][j] * gradGates[g];
                    }
                    gradPrevHidden[j] = sum;
                });
            }).join();
            
        } else {
            // Sequential computation
            for (int g = 0; g < gateSize; g++) {
                double gradG = gradGates[g];
                
                for (int j = 0; j < inputSize; j++) {
                    gradWeightsIH[g][j] = gradG * input[j];
                }
                
                for (int j = 0; j < hiddenSize; j++) {
                    gradWeightsHH[g][j] = gradG * prevHidden[j];
                }
            }
            
            // Input gradients
            for (int j = 0; j < inputSize; j++) {
                double sum = 0.0;
                for (int g = 0; g < gateSize; g++) {
                    sum += weightsIH[g][j] * gradGates[g];
                }
                gradInput[j] = sum;
            }
            
            // Hidden gradients
            for (int j = 0; j < hiddenSize; j++) {
                double sum = 0.0;
                for (int g = 0; g < gateSize; g++) {
                    sum += weightsHH[g][j] * gradGates[g];
                }
                gradPrevHidden[j] = sum;
            }
        }
    }
    
    // ==================== Sequence Processing ====================
    
    /**
     * Process a sequence through LSTM layer.
     * 
     * @param sequence input sequence [seqLength][inputSize]
     * @param initialHidden initial hidden state [hiddenSize]
     * @param initialCell initial cell state [hiddenSize]
     * @param weightsIH input-to-hidden weights [4*hiddenSize][inputSize]
     * @param weightsHH hidden-to-hidden weights [4*hiddenSize][hiddenSize]
     * @param biases biases [4*hiddenSize]
     * @param returnSequences if true, return all hidden states; if false, return only last
     * @return output hidden states [seqLength][hiddenSize] or [1][hiddenSize]
     */
    public static double[][] processSequence(double[][] sequence, 
                                              double[] initialHidden, double[] initialCell,
                                              double[][] weightsIH, double[][] weightsHH, 
                                              double[] biases, boolean returnSequences) {
        int seqLength = sequence.length;
        int hiddenSize = initialHidden.length;
        
        double[][] outputs;
        if (returnSequences) {
            outputs = new double[seqLength][hiddenSize];
        } else {
            outputs = new double[1][hiddenSize];
        }
        
        double[] hidden = initialHidden.clone();
        double[] cell = initialCell.clone();
        
        for (int t = 0; t < seqLength; t++) {
            double[][] result = lstmCellForward(sequence[t], hidden, cell, 
                                                weightsIH, weightsHH, biases);
            hidden = result[0];
            cell = result[1];
            
            if (returnSequences) {
                System.arraycopy(hidden, 0, outputs[t], 0, hiddenSize);
            }
        }
        
        if (!returnSequences) {
            System.arraycopy(hidden, 0, outputs[0], 0, hiddenSize);
        }
        
        return outputs;
    }
    
    /**
     * Process multiple sequences in parallel (batch processing).
     * 
     * @param batch batch of sequences [batchSize][seqLength][inputSize]
     * @param initialHidden initial hidden states [batchSize][hiddenSize]
     * @param initialCell initial cell states [batchSize][hiddenSize]
     * @param weightsIH input-to-hidden weights [4*hiddenSize][inputSize]
     * @param weightsHH hidden-to-hidden weights [4*hiddenSize][hiddenSize]
     * @param biases biases [4*hiddenSize]
     * @param returnSequences if true, return all hidden states
     * @return batch of output hidden states
     */
    public static double[][][] processBatch(double[][][] batch,
                                            double[][] initialHidden, double[][] initialCell,
                                            double[][] weightsIH, double[][] weightsHH,
                                            double[] biases, boolean returnSequences) {
        int batchSize = batch.length;
        int seqLength = batch[0].length;
        int hiddenSize = initialHidden[0].length;
        
        double[][][] outputs;
        if (returnSequences) {
            outputs = new double[batchSize][seqLength][hiddenSize];
        } else {
            outputs = new double[batchSize][1][hiddenSize];
        }
        
        if (config.isParallelizationEnabled() && batchSize >= 2) {
            ForkJoinPool executor = config.getExecutor();
            
            executor.submit(() ->
                IntStream.range(0, batchSize).parallel().forEach(b -> {
                    double[][] result = processSequence(batch[b], initialHidden[b], initialCell[b],
                                                       weightsIH, weightsHH, biases, returnSequences);
                    outputs[b] = result;
                })
            ).join();
        } else {
            for (int b = 0; b < batchSize; b++) {
                outputs[b] = processSequence(batch[b], initialHidden[b], initialCell[b],
                                            weightsIH, weightsHH, biases, returnSequences);
            }
        }
        
        return outputs;
    }
    
    // ==================== Bidirectional Processing ====================
    
    /**
     * Process sequence bidirectionally.
     * 
     * @param sequence input sequence [seqLength][inputSize]
     * @param initialHiddenFwd forward initial hidden state
     * @param initialCellFwd forward initial cell state
     * @param initialHiddenBwd backward initial hidden state
     * @param initialCellBwd backward initial cell state
     * @param weightsIHFwd forward input-to-hidden weights
     * @param weightsHHFwd forward hidden-to-hidden weights
     * @param biasesFwd forward biases
     * @param weightsIHBwd backward input-to-hidden weights
     * @param weightsHHBwd backward hidden-to-hidden weights
     * @param biasesBwd backward biases
     * @param mergeMode how to merge: "concat", "sum", "mul", "avg"
     * @return merged output [seqLength][mergedSize]
     */
    public static double[][] processBidirectional(double[][] sequence,
                                                   double[] initialHiddenFwd, double[] initialCellFwd,
                                                   double[] initialHiddenBwd, double[] initialCellBwd,
                                                   double[][] weightsIHFwd, double[][] weightsHHFwd, double[] biasesFwd,
                                                   double[][] weightsIHBwd, double[][] weightsHHBwd, double[] biasesBwd,
                                                   String mergeMode) {
        int seqLength = sequence.length;
        int hiddenSize = initialHiddenFwd.length;
        
        // Process forward and backward in parallel if enabled
        double[][] forwardOutputs;
        double[][] backwardOutputs;
        
        if (config.isParallelizationEnabled()) {
            ForkJoinPool executor = config.getExecutor();
            
            double[][][] results = new double[2][][];
            
            executor.submit(() -> {
                IntStream.range(0, 2).parallel().forEach(dir -> {
                    if (dir == 0) {
                        results[0] = processSequence(sequence, initialHiddenFwd, initialCellFwd,
                                                    weightsIHFwd, weightsHHFwd, biasesFwd, true);
                    } else {
                        // Reverse sequence for backward pass
                        double[][] reversedSeq = new double[seqLength][];
                        for (int t = 0; t < seqLength; t++) {
                            reversedSeq[t] = sequence[seqLength - 1 - t];
                        }
                        results[1] = processSequence(reversedSeq, initialHiddenBwd, initialCellBwd,
                                                    weightsIHBwd, weightsHHBwd, biasesBwd, true);
                    }
                });
            }).join();
            
            forwardOutputs = results[0];
            backwardOutputs = results[1];
        } else {
            forwardOutputs = processSequence(sequence, initialHiddenFwd, initialCellFwd,
                                            weightsIHFwd, weightsHHFwd, biasesFwd, true);
            
            double[][] reversedSeq = new double[seqLength][];
            for (int t = 0; t < seqLength; t++) {
                reversedSeq[t] = sequence[seqLength - 1 - t];
            }
            backwardOutputs = processSequence(reversedSeq, initialHiddenBwd, initialCellBwd,
                                             weightsIHBwd, weightsHHBwd, biasesBwd, true);
        }
        
        // Merge outputs based on mode
        return mergeOutputs(forwardOutputs, backwardOutputs, mergeMode, seqLength, hiddenSize);
    }
    
    private static double[][] mergeOutputs(double[][] forward, double[][] backward,
                                           String mode, int seqLength, int hiddenSize) {
        double[][] merged;
        
        switch (mode.toLowerCase()) {
            case "concat":
                merged = new double[seqLength][2 * hiddenSize];
                for (int t = 0; t < seqLength; t++) {
                    System.arraycopy(forward[t], 0, merged[t], 0, hiddenSize);
                    System.arraycopy(backward[seqLength - 1 - t], 0, merged[t], hiddenSize, hiddenSize);
                }
                break;
                
            case "sum":
                merged = new double[seqLength][hiddenSize];
                for (int t = 0; t < seqLength; t++) {
                    for (int i = 0; i < hiddenSize; i++) {
                        merged[t][i] = forward[t][i] + backward[seqLength - 1 - t][i];
                    }
                }
                break;
                
            case "mul":
                merged = new double[seqLength][hiddenSize];
                for (int t = 0; t < seqLength; t++) {
                    for (int i = 0; i < hiddenSize; i++) {
                        merged[t][i] = forward[t][i] * backward[seqLength - 1 - t][i];
                    }
                }
                break;
                
            case "avg":
            default:
                merged = new double[seqLength][hiddenSize];
                for (int t = 0; t < seqLength; t++) {
                    for (int i = 0; i < hiddenSize; i++) {
                        merged[t][i] = (forward[t][i] + backward[seqLength - 1 - t][i]) * 0.5;
                    }
                }
                break;
        }
        
        return merged;
    }
    
    // ==================== Utility Methods ====================
    
    /**
     * Fast sigmoid implementation.
     */
    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Initialize LSTM weights using Xavier/Glorot initialization.
     * 
     * @param inputSize input dimension
     * @param hiddenSize hidden dimension
     * @param seed random seed
     * @return array containing [weightsIH, weightsHH, biases]
     */
    public static Object[] initializeWeights(int inputSize, int hiddenSize, long seed) {
        java.util.Random random = new java.util.Random(seed);
        
        int gateSize = 4 * hiddenSize;
        
        double[][] weightsIH = new double[gateSize][inputSize];
        double[][] weightsHH = new double[gateSize][hiddenSize];
        double[] biases = new double[gateSize];
        
        // Xavier initialization
        double scaleIH = Math.sqrt(2.0 / (inputSize + hiddenSize));
        double scaleHH = Math.sqrt(2.0 / (hiddenSize + hiddenSize));
        
        for (int g = 0; g < gateSize; g++) {
            for (int j = 0; j < inputSize; j++) {
                weightsIH[g][j] = random.nextGaussian() * scaleIH;
            }
            for (int j = 0; j < hiddenSize; j++) {
                weightsHH[g][j] = random.nextGaussian() * scaleHH;
            }
        }
        
        // Initialize forget gate bias to 1.0 for better gradient flow
        for (int i = 0; i < hiddenSize; i++) {
            biases[hiddenSize + i] = 1.0;
        }
        
        return new Object[] { weightsIH, weightsHH, biases };
    }
    
    /**
     * Apply gradient clipping to prevent exploding gradients.
     * 
     * @param gradients gradient array
     * @param maxNorm maximum gradient norm
     */
    public static void clipGradients(double[] gradients, double maxNorm) {
        double norm = VectorOps.norm2(gradients);
        
        if (norm > maxNorm) {
            double scale = maxNorm / norm;
            for (int i = 0; i < gradients.length; i++) {
                gradients[i] *= scale;
            }
        }
    }
    
    /**
     * Apply gradient clipping to weight matrix.
     * 
     * @param gradients gradient matrix
     * @param maxNorm maximum gradient norm
     */
    public static void clipGradients(double[][] gradients, double maxNorm) {
        double norm = ParallelMatrix.frobeniusNorm(gradients);
        
        if (norm > maxNorm) {
            double scale = maxNorm / norm;
            int m = gradients.length;
            int n = gradients[0].length;
            
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    gradients[i][j] *= scale;
                }
            }
        }
    }
}
