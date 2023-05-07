#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
//#include "matplotlibcpp.h" // Include the matplotlibcpp header
//
//namespace plt = matplotlibcpp; // Create an alias for the matplotlibcpp namespace
using namespace std;

using namespace std;

// Function to compute the Euclidean distance between two points
double euclidean_distance(pair<double, double> a, pair<double, double> b) {
    return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
}

double  manhattan_distance(pair<double, double> a, pair<double, double> b) {
    return abs(a.first - b.first) + abs(a.second - b.second);
}

// kNN classification function
int knn_classification(const vector<pair<pair<double, double>, int>>& dataset, pair<double, double> new_point, int k) {
    vector<pair<double, int>> distances;

    // Compute distances from the new point to all data points in the dataset
    for (const auto& data : dataset) {
        distances.push_back({euclidean_distance(new_point, data.first), data.second});
    }

    // Sort the distances vector
    sort(distances.begin(), distances.end());

    // Find the majority class label among the k nearest neighbors
    vector<int> class_count(2, 0); // Assuming binary classification: 2 classes (0 and 1)
    for (int i = 0; i < k; i++) {
        class_count[distances[i].second]++;
    }

    // Return the class label with the highest count
    return (class_count[0] > class_count[1]) ? 0 : 1;
}

int main() {
    // Sample dataset: Each entry consists of a pair of features (X1, X2) and a class label (Y)
    vector<pair<pair<double, double>, int>> dataset = {
        {{2.0, 3.0}, 0},
        {{4.0, 6.0}, 1},
        {{3.0, 4.0}, 0},
        {{6.0, 8.0}, 1},
        {{5.0, 7.0}, 1},
        {{1.0, 2.0}, 0}
    };

    // New data point to classify
    pair<double, double> new_point = {5.0, 5.0};

    // Number of nearest neighbors to consider (k)
    int k = 3;

    // Classify the new data point using kNN
    int predicted_class = knn_classification(dataset, new_point, k);
    cout << "Predicted class for the new data point: " << predicted_class << endl;

    return 0;
}
