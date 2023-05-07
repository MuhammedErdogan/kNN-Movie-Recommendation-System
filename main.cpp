#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace std;

using DistanceFunction = double (*)(const vector<double> &, const vector<double> &);

void write_dataset_random_user_ratings(const string &file_path, int num_users, int num_movies) {
    ofstream file(file_path);

    if (!file.is_open()) {
        cerr << "Error opening the file: " << file_path << endl;
        return;
    }

    for (int i = 0; i < num_users; ++i) {
        file << "user_to_predict" << i << ":";
        for (int j = 0; j < num_movies; ++j) {
            file << rand() % 10 + 1;
            if (j < num_movies - 1) {
                file << ", ";
            }
        }
        file << "" << endl;
    }

    file.close();
}

vector<vector<double>> read_dataset(const string &file_path) {
    vector<vector<double>> dataset;
    ifstream file(file_path);

    if (!file.is_open()) {
        cerr << "Error opening the file: " << file_path << endl;
        return dataset;
    }

    string line;
    while (getline(file, line) && !line.empty()) {
        istringstream iss(line);
        vector<double> row;
        string temp;
        double value;

        // Discard the user number and colon
        getline(iss, temp, ':');

        // Read the opening brace
        iss >> temp;

        string s = temp.substr(0, 1);
//        cout << s << " ";
        value = stod(s);
        row.push_back(value);

        // Read the values until the closing brace is found
        while (getline(iss, temp, ',')) {
            value = stod(temp);
            row.push_back(value);
//            cout << value << " ";
        }
//        cout << endl;
        dataset.push_back(row);
    }
//    cout << "file closed" << endl;
    file.close();
    return dataset;
}

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
                      DistanceFunction distance_function, int movie_idx) {
    vector<pair<double, int>> distances;

    // Compute distances from the new point to all data points in the dataset using the specified distance function
    for (int i = 0; i < dataset.size(); ++i) {
        distances.push_back({distance_function(new_point, dataset[i]), i});
//        cout << distance_function(new_point, dataset[i]) << ":" << i << endl;
    }

    // Sort the distances vector
    sort(distances.begin(), distances.end());

    // Calculate the average rating of the k nearest neighbors
    double sum_ratings = 0.0;
    for (int i = 0; i < k; i++) {
        int user_idx = distances[i].second;
        sum_ratings += dataset[user_idx][movie_idx];
    }

    return sum_ratings / k;
}

int main() {
    // Read dataset from file
    vector<vector<double>> dataset = read_dataset("dataset.txt");
    vector<vector<double>> user_to_predict = read_dataset("user-to-predict.txt");
    // Dataset: Each row represents a user, and each column represents a movie
    // New data point to predict rating for movie 1
    vector<double> new_point = {4, 7, 2, 10,
                                1, 4, 4, 1,
                                2, 5, 9, 3,
                                3, 4, 3, 2,
                                1, 6, 4,};

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
        double predicted_rating = knn_regression(dataset, new_point, k, df.first, 19);
        results.push_back({df.second, predicted_rating});
    }

    // Display the results
    cout << "Predicted ratings for movie 1 using different distance functions:" << endl;
    for (const auto &result: results) {
        cout << result.first << ": " << result.second << endl;
    }

    return 0;
}