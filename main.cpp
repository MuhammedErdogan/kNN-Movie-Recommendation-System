#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

using DistanceFunction = double (*)(const vector<double> &, const vector<double> &);

// Distance functions
double euclidean_distance(const vector<double> &a, const vector<double> &b) {
    double distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        distance += pow(a[i] - b[i], 2);
    }
    return sqrt(distance);
}

double manhattan_distance(const vector<double> &a, const vector<double> &b) {
    double distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        distance += abs(a[i] - b[i]);
    }
    return distance;
}

double chebyshev_distance(const vector<double> &a, const vector<double> &b) {
    double max_distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double distance = abs(a[i] - b[i]);
        max_distance = max(max_distance, distance);
    }
    return max_distance;
}

double minkowski_distance(const vector<double> &a, const vector<double> &b, double p) {
    double distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        distance += pow(abs(a[i] - b[i]), p);
    }
    return pow(distance, 1.0 / p);
}

double minkowski_distance_p2(const vector<double> &a, const vector<double> &b) {
    return minkowski_distance(a, b, 2);
}

double minkowski_distance_p3(const vector<double> &a, const vector<double> &b) {
    return minkowski_distance(a, b, 3);
}

// kNN regression function
double knn_regression(const vector<vector<double>> &dataset, const vector<double> &new_point, int k,
                      DistanceFunction distance_function) {
    vector<pair<double, int>> distances;

    // Compute distances from the new point to all data points in the dataset using the specified distance function
    for (int i = 0; i < dataset.size(); ++i) {
        distances.push_back({distance_function(new_point, dataset[i]), i});
    }

    // Sort the distances vector
    sort(distances.begin(), distances.end());

    // Calculate the average rating of the k nearest neighbors
    double sum_ratings = 0.0;
    for (int i = 0; i < k; i++) {
        int user_idx = distances[i].second;
        sum_ratings += new_point[user_idx];
    }

    return sum_ratings / k;
}

int main() {
    // Dataset: Each row represents a user, and each column represents a movie
    vector<vector<double>> dataset = {
            {4, 0, 0, 5, 1},
            {0, 5, 4, 0, 0},
            {3, 0, 0, 4, 2},
            {0, 4, 5, 0, 0},
            {0, 0, 4, 1, 5},
            {0, 0, 5, 2, 4},
            {5, 2, 0, 0, 0},
            {2, 0, 0, 3, 0},
            {0, 0, 2, 4, 0},
            {0, 3, 0, 0, 2},
            {0, 0, 3, 0, 4},
            {0, 0, 2, 1, 5},
            {5, 4, 5, 0, 0},
    };

    // New data point to predict rating for movie 1
    vector<double> new_point = {4, 5, 0, 0};

    // Number of nearest neighbors to consider (k)
    int k = 3;

    // Create a list of distance functions
    vector<pair<DistanceFunction, string>> distance_functions = {
            {euclidean_distance,    "Euclidean"},
            {manhattan_distance,    "Manhattan"},
            {chebyshev_distance,    "Chebyshev"},
            {minkowski_distance_p2, "Minkowski (p=2)"},
            {minkowski_distance_p3, "Minkowski (p=3)"}
    };

// Compute the predicted rating for each distance function and store the results in a list
    vector<pair<string, double>> results;
    for (const auto &df: distance_functions) {
        double predicted_rating = knn_regression(dataset, new_point, k, df.first);
        results.push_back({df.second, predicted_rating});
    }

    // Display the results
    cout << "Predicted ratings for movie 1 using different distance functions:" << endl;
    for (const auto &result: results) {
        cout << result.first << ": " << result.second << endl;
    }

    return 0;
}