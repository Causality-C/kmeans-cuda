
#include "profile.h"

float euclidean_distance(std::vector<float> point1, std::vector<float> point2)
{
    if (point1.size() != point2.size())
    {
        std::cerr << "Error: Points have different dimensions" << std::endl;
        return -1;
    }
    float distance = 0;
    for (int i = 0; i < point1.size(); i++)
    {
        distance += std::pow(point1[i] - point2[i], 2);
    }
    return std::sqrt(distance);
}

float loss(std::vector<std::vector<float>> points, std::vector<std::vector<float>> centroids)
{
    float loss = 0.0f;
    for (int i = 0; i < points.size(); i++)
    {
        float min_distance = std::numeric_limits<float>::max();
        for (int j = 0; j < centroids.size(); j++)
        {
            float distance = euclidean_distance(points[i], centroids[j]);
            if (distance < min_distance)
            {
                min_distance = distance;
            }
        }
        loss += min_distance;
    }
    return loss;
}