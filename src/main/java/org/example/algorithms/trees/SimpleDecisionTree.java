package org.example.algorithms.trees;

import java.util.*;

public class SimpleDecisionTree {

    static class Node {
        String feature; // Feature to split on
        Map<String, Node> children = new HashMap<>();
        String label; // Leaf label (if it's a leaf)

        boolean isLeaf() {
            return label != null;
        }
    }

    // Sample dataset
    static List<Map<String, String>> data = List.of(
            Map.of("Outlook", "Sunny", "Temperature", "Hot", "PlayTennis", "No"),
            Map.of("Outlook", "Sunny", "Temperature", "Cool", "PlayTennis", "Yes"),
            Map.of("Outlook", "Overcast", "Temperature", "Hot", "PlayTennis", "Yes"),
            Map.of("Outlook", "Rain", "Temperature", "Mild", "PlayTennis", "Yes"),
            Map.of("Outlook", "Rain", "Temperature", "Cool", "PlayTennis", "No")
    );

    static Set<String> features = Set.of("Outlook", "Temperature");

    public static void main(String[] args) {
        Node root = buildTree(data, new HashSet<>(features));
        printTree(root, "");
        System.out.println("Prediction for {Outlook=Rain, Temperature=Cool}: " +
                predict(root, Map.of("Outlook", "Rain", "Temperature", "Cool")));
    }

    static Node buildTree(List<Map<String, String>> rows, Set<String> remainingFeatures) {
        Node node = new Node();

        String firstLabel = rows.get(0).get("PlayTennis");
        boolean allSame = rows.stream().allMatch(row -> row.get("PlayTennis").equals(firstLabel));
        if (allSame) {
            node.label = firstLabel;
            return node;
        }

        if (remainingFeatures.isEmpty()) {
            node.label = mostCommonLabel(rows);
            return node;
        }

        String bestFeature = chooseBestFeature(rows, remainingFeatures);
        node.feature = bestFeature;

        Map<String, List<Map<String, String>>> partitions = new HashMap<>();
        for (Map<String, String> row : rows) {
            String value = row.get(bestFeature);
            partitions.computeIfAbsent(value, k -> new ArrayList<>()).add(row);
        }

        Set<String> newFeatures = new HashSet<>(remainingFeatures);
        newFeatures.remove(bestFeature);

        for (Map.Entry<String, List<Map<String, String>>> entry : partitions.entrySet()) {
            node.children.put(entry.getKey(), buildTree(entry.getValue(), newFeatures));
        }

        return node;
    }

    static String chooseBestFeature(List<Map<String, String>> rows, Set<String> features) {
        double baseEntropy = entropy(rows);
        String bestFeature = null;
        double bestGain = -1;

        for (String feature : features) {
            double gain = baseEntropy - featureEntropy(rows, feature);
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = feature;
            }
        }

        return bestFeature;
    }

    static double entropy(List<Map<String, String>> rows) {
        Map<String, Integer> counts = new HashMap<>();
        for (Map<String, String> row : rows) {
            counts.merge(row.get("PlayTennis"), 1, Integer::sum);
        }

        double result = 0.0;
        int total = rows.size();
        for (int count : counts.values()) {
            double p = (double) count / total;
            result -= p * Math.log(p) / Math.log(2);
        }
        return result;
    }

    static double featureEntropy(List<Map<String, String>> rows, String feature) {
        Map<String, List<Map<String, String>>> partitions = new HashMap<>();
        for (Map<String, String> row : rows) {
            String value = row.get(feature);
            partitions.computeIfAbsent(value, k -> new ArrayList<>()).add(row);
        }

        double totalEntropy = 0.0;
        int total = rows.size();
        for (List<Map<String, String>> partition : partitions.values()) {
            totalEntropy += ((double) partition.size() / total) * entropy(partition);
        }

        return totalEntropy;
    }

    static String mostCommonLabel(List<Map<String, String>> rows) {
        Map<String, Integer> count = new HashMap<>();
        for (Map<String, String> row : rows) {
            count.merge(row.get("PlayTennis"), 1, Integer::sum);
        }

        return count.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
    }

    static String predict(Node node, Map<String, String> input) {
        while (!node.isLeaf()) {
            String value = input.get(node.feature);
            node = node.children.get(value);
            if (node == null) return "Unknown";
        }
        return node.label;
    }

    static void printTree(Node node, String indent) {
        if (node.isLeaf()) {
            System.out.println(indent + "Label: " + node.label);
        } else {
            for (Map.Entry<String, Node> entry : node.children.entrySet()) {
                System.out.println(indent + node.feature + " = " + entry.getKey() + ":");
                printTree(entry.getValue(), indent + "  ");
            }
        }
    }
}

