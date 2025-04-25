package org.example.algorithms.linnear;

import java.util.Arrays;
import java.util.Random;

public class Perceptron {
    private double[] weights;
    private double bias;
    private double learningRate;

    /**
     * Constructor for Perceptron
     * @param numFeatures number of input features
     * @param learningRate learning rate for training
     */
    public Perceptron(int numFeatures, double learningRate) {
        this.weights = new double[numFeatures];
        this.bias = 0.0;
        this.learningRate = learningRate;

        // Initialize weights with small random values
        Random random = new Random(42); // fixed seed for reproducibility
        for (int i = 0; i < numFeatures; i++) {
            this.weights[i] = random.nextDouble() * 0.1 - 0.05; // Small random values between -0.05 and 0.05
        }
    }

    /**
     * Compute the weighted sum of inputs and weights
     * @param inputs input features
     * @return weighted sum plus bias
     */
    private double weightedSum(double[] inputs) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("Input size must match weight vector size");
        }

        double sum = bias;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sum;
    }

    /**
     * Apply step activation function
     * @param weightedSum the weighted sum
     * @return 1 if weighted sum >= 0, otherwise 0
     */
    private int activate(double weightedSum) {
        return weightedSum >= 0 ? 1 : 0;
    }

    /**
     * Predict the class of an input
     * @param inputs input features
     * @return predicted class (0 or 1)
     */
    public int predict(double[] inputs) {
        double sum = weightedSum(inputs);
        return activate(sum);
    }

    /**
     * Train the perceptron using an input and its expected output
     * @param inputs input features
     * @param expectedOutput expected class (0 or 1)
     * @return prediction error (0 if correct, non-zero otherwise)
     */
    public int train(double[] inputs, int expectedOutput) {
        int prediction = predict(inputs);
        int error = expectedOutput - prediction;

        if (error != 0) {
            // Update weights
            for (int i = 0; i < weights.length; i++) {
                weights[i] += learningRate * error * inputs[i];
            }
            // Update bias
            bias += learningRate * error;
        }

        return error;
    }

    /**
     * Train the perceptron for multiple epochs
     * @param trainingData array of training examples
     * @param labels array of expected outputs
     * @param epochs number of training epochs
     */
    public void fit(double[][] trainingData, int[] labels, int epochs) {
        if (trainingData.length != labels.length) {
            throw new IllegalArgumentException("Number of training examples must match number of labels");
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            int totalErrors = 0;

            for (int i = 0; i < trainingData.length; i++) {
                int error = train(trainingData[i], labels[i]);
                totalErrors += Math.abs(error);
            }

            System.out.println("Epoch " + (epoch + 1) + "/" + epochs + ", Total errors: " + totalErrors);

            // Early stopping if all examples are correctly classified
            if (totalErrors == 0) {
                System.out.println("Early stopping at epoch " + (epoch + 1) + ": All examples correctly classified");
                break;
            }
        }
    }

    /**
     * Get the weights
     * @return array of weights
     */
    public double[] getWeights() {
        return weights;
    }

    /**
     * Get the bias
     * @return bias value
     */
    public double getBias() {
        return bias;
    }

    /**
     * Print the model parameters
     */
    public void printModel() {
        System.out.println("Weights: " + Arrays.toString(weights));
        System.out.println("Bias: " + bias);
    }

    public static void main(String[] args) {
        // Example usage: Binary classification for logical OR function
        double[][] trainingData = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };

        int[] labels = {0, 1, 1, 1}; // OR function labels

        // Create perceptron with 2 input features and learning rate of 0.1
        Perceptron perceptron = new Perceptron(2, 0.1);

        // Train the perceptron
        System.out.println("Training perceptron for OR function...");
        perceptron.fit(trainingData, labels, 100);

        // Print final model parameters
        System.out.println("\nFinal model parameters:");
        perceptron.printModel();

        // Test the trained model
        System.out.println("\nTesting perceptron on OR function:");
        for (int i = 0; i < trainingData.length; i++) {
            int prediction = perceptron.predict(trainingData[i]);
            System.out.println("Input: " + Arrays.toString(trainingData[i]) +
                    ", Prediction: " + prediction +
                    ", Expected: " + labels[i]);
        }

        // Example of inference with new data
        System.out.println("\nInference with new data:");
        double[] newInput = {0.5, 0.5};
        int prediction = perceptron.predict(newInput);
        System.out.println("Input: " + Arrays.toString(newInput) + ", Prediction: " + prediction);
    }
}
