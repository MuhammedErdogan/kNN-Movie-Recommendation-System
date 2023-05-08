#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>
using namespace std;

using DistanceFunction = double (*)(const vector<double> &, const vector<double> &);

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

pair<pair<string, double>, double> find_best_distance_method(
        vector<pair<string, double>> &results,
        double real_rating) {
    vector<pair<DistanceFunction, string>> avg_diff_results;

    pair<pair<string, double>, double> best_method;
    double min_difference = 1000;
    for (const auto &df: results) {
        double total_difference = 0;
        double predicted_rating = df.second;

        double cur_diff = abs(predicted_rating - real_rating);
        total_difference += cur_diff;

        if (cur_diff < min_difference) {
            min_difference = cur_diff;
            best_method = {{df.first, predicted_rating}, cur_diff};
        }
    }

    return best_method;
}

void increment_distance_method_count(map<string, int> &counts, const string &method_name) {
    if (counts.find(method_name) == counts.end()) {
        counts[method_name] = 1;
    } else {
        counts[method_name]++;
    }
}

string get_most_frequent_method(const map<string, int> &counts) {
    string most_frequent_method;
    int max_count = 0;

    for (const auto &entry : counts) {
        if (entry.second > max_count) {
            most_frequent_method = entry.first;
            max_count = entry.second;
        }
    }

    return most_frequent_method;
}

int main() {
    // Read dataset from file
    vector<vector<double>> dataset = read_dataset("dataset.txt");
    vector<vector<double>> user_to_predict = read_dataset("user-to-predict.txt");
    vector<vector<double>> real_ratings = read_dataset("user-real-ratings.txt");
    // Dataset: Each row represents a user, and each column represents a movie
    // New data point to predict rating for movie 1
    vector<string> best_methods;
    vector<double> new_point;

    for (int i = 0; i < user_to_predict.size(); ++i) {
        new_point = user_to_predict[i];
        cout << "User " << i << endl;
        cout << "New point: ";
        for (const auto &value: new_point) {
            cout << value << " ";
        }
        cout << endl;
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
            auto predicted_rating = knn_regression(dataset, new_point, k, df.first, 19);
            results.push_back({df.second, predicted_rating});
        }

        // Display the results
        double real_rating = real_ratings[i][0];
        for (const auto &result: results) {
            cout << result.first << " distance: " << result.second
                 << ", difference: " << abs(result.second - real_rating) << endl;
        }

        auto best_method = find_best_distance_method(results, real_rating);

        cout << "Best distance method: " << best_method.first.first << " with: " << best_method.second << endl;
        best_methods.push_back({best_method.first.first});


        cout << endl;
    }

    map<string, int> distance_method_counts;

    for (const auto &best_method : best_methods) {
        increment_distance_method_count(distance_method_counts, best_method);
    }

    string most_frequent_method = get_most_frequent_method(distance_method_counts);
    cout << "Most frequent best distance method: " << most_frequent_method << endl;
    
    return 0;
}