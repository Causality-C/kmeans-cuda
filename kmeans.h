#pragma once

#include <vector>

extern "C" void launch_kmeans_kernel(std::vector<std::vector<float>> &data_points, std::vector<std::vector<float>> &centroids, int num_pts, int k, int max_iter, int dims, float tolerance);