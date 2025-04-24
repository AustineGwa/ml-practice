package org.example.algorithms.linnear;

import java.util.*;

public class KNN {

    // A simple class to hold a data point
    static class DataPoint {
        double[] features;
        String label;

        DataPoint(double[] features, String label) {
            this.features = features;
            this.label = label;
        }
    }

    // Calculates Euclidean distance between two points
    static double distance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    // Predict the label using KNN
    static String predict(List<DataPoint> trainingData, double[] input, int k) {
        // Create a list of distances
        List<Map.Entry<Double, String>> distances = new ArrayList<>();
        for (DataPoint point : trainingData) {
            double dist = distance(point.features, input);
            distances.add(new AbstractMap.SimpleEntry<>(dist, point.label));
        }

        // Sort by distance
        distances.sort(Comparator.comparing(Map.Entry::getKey));

        // Get the top K labels
        Map<String, Integer> labelCounts = new HashMap<>();
        for (int i = 0; i < k; i++) {
            String label = distances.get(i).getValue();
            labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
        }

        // Return the most common label
        return Collections.max(labelCounts.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    // Sample usage
    public static void main(String[] args) {
        List<DataPoint> trainingData = new ArrayList<>();
        trainingData.add(new DataPoint(new double[]{1.0, 2.0}, "A"));
        trainingData.add(new DataPoint(new double[]{1.5, 1.8}, "A"));
        trainingData.add(new DataPoint(new double[]{5.0, 8.0}, "B"));
        trainingData.add(new DataPoint(new double[]{6.0, 9.0}, "B"));
        trainingData.add(new DataPoint(new double[]{1.0, 0.6}, "A"));
        trainingData.add(new DataPoint(new double[]{9.0, 11.0}, "B"));

        double[] input = new double[]{2.0, 2.0};

        int k = 3;
        String prediction = predict(trainingData, input, k);

        System.out.println("Predicted label: " + prediction);
    }
}
