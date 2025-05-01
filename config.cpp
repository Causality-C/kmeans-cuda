#include "config.h"
#include <iostream>
#include <cstdlib>

KmeansConfig parse_args(int argc, char *argv[])
{
    KmeansConfig config;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-k" && i + 1 < argc)
        {
            config.k = std::atoi(argv[++i]);
        }
        else if (arg == "-d" && i + 1 < argc)
        {
            config.dims = std::atoi(argv[++i]);
        }
        else if (arg == "-i" && i + 1 < argc)
        {
            config.inputfilename = argv[++i];
        }
        else if (arg == "-m" && i + 1 < argc)
        {
            config.max_iter = std::atoi(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc)
        {
            config.threshold = std::atof(argv[++i]);
        }
        else if (arg == "-b" && i + 1 < argc)
        {
            config.reference_answer_filename = argv[++i];
        }
        else if (arg == "-c")
        {
            config.print_centroids = true;
        }
        else if (arg == "-s" && i + 1 < argc)
        {
            config.seed = std::atoi(argv[++i]);
        }
        else if (arg == "-g")
        {
            config.use_gpu = true;
        }
        else if (arg == "-v")
        {
            config.debug = true;
        }
        else
        {
            std::cerr << "Unknown or malformed option: " << arg << "\n";
        }
    }

    if (config.k == -1 || config.dims == -1 || config.inputfilename == "")
    {
        std::cerr << "Missing required arguments\n";
        exit(1);
    }

    return config;
}

void print_config(KmeansConfig config)
{
    std::cout << "k: " << config.k << "\n";
    std::cout << "dims: " << config.dims << "\n";
    std::cout << "input: " << config.inputfilename << "\n";
    std::cout << "reference answer: " << config.reference_answer_filename << "\n";
    std::cout << "max_iter: " << config.max_iter << "\n";
    std::cout << "threshold: " << config.threshold << "\n";
    std::cout << "print centroids: " << (config.print_centroids ? "yes" : "no") << "\n";
    std::cout << "seed: " << config.seed << "\n";
    std::cout << "use GPU: " << (config.use_gpu ? "yes" : "no") << "\n";
    std::cout << "debug: " << (config.debug ? "yes" : "no") << "\n";
}

/*
 * kmeans initialization algorithms:
 *  seed: the seed provided via command line arguments
 *  num_pts: the number of points in the input
 *  num_centroids: the number of centroids specified
 *  D(x): distance between the xth point and its nearest centroid
 */
float rand_float()
{
    // Returns a float in [0.0, 1.0)
    return static_cast<float>(rand()) / static_cast<float>((long long)RAND_MAX + 1);
}

std::vector<int> kmeans_init_centroids(int num_clusters, int num_pts)
{
    std::vector<int> centroid_indices;
    for (int i = 0; i < num_clusters; i++)
    {
        int index = (int)(rand_float() * num_pts); // the index of the point that will be used for this centroid
        centroid_indices.push_back(index);
    }
    return centroid_indices;
}
