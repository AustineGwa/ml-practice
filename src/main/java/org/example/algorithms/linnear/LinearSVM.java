package org.example.algorithms.linnear;

public class LinearSVM {
    double[] weights = new double[3]; // [bias, w1, w2]
    double learningRate = 0.01;
    double C = 1.0; // Regularization strength

    public void train(double[][] X, int[] y, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < X.length; i++) {
                int label = y[i] == 1 ? 1 : -1; // Convert to -1 and +1
                double z = weights[0] + weights[1]*X[i][0] + weights[2]*X[i][1];
                if (label * z >= 1) {
                    // No loss — only regularization
                    weights[1] -= learningRate * 2 * weights[1];
                    weights[2] -= learningRate * 2 * weights[2];
                } else {
                    // Hinge loss gradient
                    weights[0] += learningRate * label * C;
                    weights[1] += learningRate * (label * X[i][0] * C - 2 * weights[1]);
                    weights[2] += learningRate * (label * X[i][1] * C - 2 * weights[2]);
                }
            }
        }
    }

    public int predict(double[] x) {
        double z = weights[0] + weights[1]*x[0] + weights[2]*x[1];
        return z >= 0 ? 1 : 0;
    }

    public static void main(String[] args) {
        double[][] X = { {1, 2}, {2, 3}, {3, 3}, {4, 5}, {6, 8}, {7, 7}, {8, 8}, {9, 10} };
        int[] y =       {  0,     0,     0,     0,     1,     1,     1,     1  };

        LinearSVM model = new LinearSVM();
        model.train(X, y, 1000);

        System.out.println("SVM prediction for [5, 5]: " + model.predict(new double[]{5, 5}));
    }
}

