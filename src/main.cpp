#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cmath>
#include "kmeans.h"
#include "config.h"
#include "profile.h"

int main(int argc, char *argv[])
{

    // Config
    KmeansConfig config = parse_args(argc, argv);
    srand(config.seed);
    if (config.debug)
    {
        print_config(config);
    }

    // Parse the input file to get the number of points and the data points
    std::ifstream input_file(config.inputfilename);
    if (!input_file.is_open())
    {
        std::cerr << "Error: Could not open input file: " << config.inputfilename << std::endl;
        return 1;
    }

    int num_pts;
    input_file >> num_pts;
    std::cout << "num_pts: " << num_pts << std::endl;
    std::vector<std::vector<float>> data_points;
    for (int i = 0; i < num_pts; i++)
    {
        int dummy;
        input_file >> dummy;
        std::vector<float> point;
        for (int j = 0; j < config.dims; j++)
        {
            float value;
            input_file >> value;
            point.push_back(value);
        }
        data_points.push_back(point);
    }
    input_file.close();

    std::vector<int> centroid_indices = kmeans_init_centroids(config.k, num_pts);
    std::vector<std::vector<float>> centroids;

    for (int i = 0; i < centroid_indices.size(); i++)
    {
        std::vector<float> centroid;
        for (int j = 0; j < config.dims; j++)
        {
            centroid.push_back(data_points[centroid_indices[i]][j]);
        }
        centroids.push_back(centroid);
    }

    // Initialize a vector of int indices of size num_pts with 0 filled
    std::vector<int> assignments(num_pts, 0);

    // GPU implementation
    if (config.use_gpu)
    {
        launch_kmeans_kernel(data_points, centroids, num_pts, config.k, config.max_iter, config.dims, config.threshold);
    }
    else
    {
        // CPU implementation
        for (int i = 0; i < config.max_iter; i++)
        {
            // Counts of vectors in each cluster
            std::vector<int> counts(config.k, 0);
            std::vector<std::vector<float>> new_centroids(config.k, std::vector<float>(config.dims, 0));
            float loss = 0.0f;

            // Step 1: assign points to nearest centroid
            for (int j = 0; j < num_pts; j++)
            {
                int min_index = -1;
                float min_distance = std::numeric_limits<float>::max();
                for (int l = 0; l < config.k; l++)
                {
                    float distance = euclidean_distance(data_points[j], centroids[l]);
                    if (distance < min_distance)
                    {
                        min_distance = distance;
                        min_index = l;
                    }
                }
                // Update the loss
                loss += min_distance;
                // Update the assignments
                assignments[j] = min_index;
                counts[min_index]++;
                for (int d = 0; d < config.dims; d++)
                {
                    new_centroids[min_index][d] += data_points[j][d];
                }
            }
            // Step 2: update centroids
            for (int j = 0; j < config.k; j++)
            {
                for (int d = 0; d < config.dims; d++)
                {
                    new_centroids[j][d] /= counts[j];
                }
            }
            // Check shift
            float shift = 0.0f;
            for (int j = 0; j < config.k; j++)
            {
                shift += euclidean_distance(centroids[j], new_centroids[j]);
            }
            // If shift is less than threshold, break
            std::cout << "Shift in round " << i << " is: " << shift << std::endl;
            if (shift < config.threshold)
            {
                break;
            }
            // Update the centroids
            centroids = new_centroids;
        }
    }
    // Check your answer with the "best" answer
    float best_loss = loss(data_points, centroids);
    std::cout << "Best loss is: " << best_loss << std::endl;

    if (config.reference_answer_filename != "")
    {
        std::ifstream best_answer_file(config.reference_answer_filename);
        std::vector<std::vector<float>> best_centroids;
        for (int i = 0; i < config.k; i++)
        {
            int dummy;
            best_answer_file >> dummy;

            std::vector<float> centroid;
            for (int j = 0; j < config.dims; j++)
            {
                float value;
                best_answer_file >> value;
                centroid.push_back(value);
            }
            best_centroids.push_back(centroid);
        }
        best_answer_file.close();
        float best_loss_reference = loss(data_points, best_centroids);
        std::cout << "Best loss reference is: " << best_loss_reference << std::endl;
    }
    // TODO: profiling
    // GPU implementation
    return 0;
}