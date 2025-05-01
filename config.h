#pragma once
#include <string>
#include <vector>
typedef struct
{
    int k = -1;
    int dims = -1;
    std::string inputfilename = "";
    int max_iter = 300;
    float threshold = 0.0001;
    bool print_centroids = false;
    bool debug = false;
    int seed = 8675309;
    bool use_gpu = false;
    std::string reference_answer_filename;
} KmeansConfig;

KmeansConfig parse_args(int argc, char *argv[]);
void print_config(KmeansConfig config);
float rand_float();
std::vector<int> kmeans_init_centroids(int num_clusters, int num_pts);