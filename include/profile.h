#pragma once
#include <vector>
#include <cmath>
#include <iostream>

float euclidean_distance(std::vector<float> point1, std::vector<float> point2);
float loss(std::vector<std::vector<float>> points, std::vector<std::vector<float>> centroids);