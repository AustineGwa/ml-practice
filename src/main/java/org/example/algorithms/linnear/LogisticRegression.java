package org.example.algorithms.linnear;

public class LogisticRegression {
    double[] weights = new double[3]; // [bias, w1, w2]
    double learningRate = 0.1;

    public double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public void train(double[][] X, int[] y, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < X.length; i++) {
                double z = weights[0] + weights[1]*X[i][0] + weights[2]*X[i][1];
                double pred = sigmoid(z);
                double error = y[i] - pred;

                // Gradient update
                weights[0] += learningRate * error;
                weights[1] += learningRate * error * X[i][0];
                weights[2] += learningRate * error * X[i][1];
            }
        }
    }

    public int predict(double[] x) {
        double z = weights[0] + weights[1]*x[0] + weights[2]*x[1];
        return sigmoid(z) >= 0.5 ? 1 : 0;
    }

    public static void main(String[] args) {
        double[][] X = { {1, 2}, {2, 3}, {3, 3}, {4, 5}, {6, 8}, {7, 7}, {8, 8}, {9, 10} };
        int[] y =       {  0,     0,     0,     0,     1,     1,     1,     1  };

        LogisticRegression model = new LogisticRegression();
        model.train(X, y, 1000);

        System.out.println("Logistic prediction for [5, 5]: " + model.predict(new double[]{5, 5}));
    }
}

