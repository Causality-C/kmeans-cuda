#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cmath>
extern "C" void launch_kmeans_kernel(std::vector<std::vector<float>> &data_points, std::vector<std::vector<float>> &centroids, int num_pts, int k, int max_iter, int dims);
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

int main(int argc, char *argv[])
{
    int k = -1;
    int dims = -1;
    std::string inputfilename;
    int max_iter = 300;
    float threshold = 0.001;
    bool print_centroids = false;
    int seed = 42;
    bool use_gpu = false;
    bool use_gpu_shared = false;
    bool use_kmeanspp = false;
    std::string reference_answer_filename;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "-k" && i + 1 < argc)
        {
            k = std::atoi(argv[++i]);
        }
        else if (arg == "-d" && i + 1 < argc)
        {
            dims = std::atoi(argv[++i]);
        }
        else if (arg == "-i" && i + 1 < argc)
        {
            inputfilename = argv[++i];
        }
        else if (arg == "-m" && i + 1 < argc)
        {
            max_iter = std::atoi(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc)
        {
            threshold = std::atof(argv[++i]);
        }
        else if (arg == "-b" && i + 1 < argc)
        {
            reference_answer_filename = argv[++i];
        }
        else if (arg == "-c")
        {
            print_centroids = true;
        }
        else if (arg == "-s" && i + 1 < argc)
        {
            seed = std::atoi(argv[++i]);
        }
        else if (arg == "-g")
        {
            use_gpu = true;
        }
        else if (arg == "-f")
        {
            use_gpu_shared = true;
        }
        else if (arg == "-p")
        {
            use_kmeanspp = true;
        }
        else
        {
            std::cerr << "Unknown or malformed option: " << arg << "\n";
        }
    }

    // Seed the RNG
    srand(seed);

    // Debug print to verify parsing
    std::cout << "k: " << k << "\n";
    std::cout << "dims: " << dims << "\n";
    std::cout << "input: " << inputfilename << "\n";
    std::cout << "reference answer: " << reference_answer_filename << "\n";
    std::cout << "max_iter: " << max_iter << "\n";
    std::cout << "threshold: " << threshold << "\n";
    std::cout << "print centroids: " << (print_centroids ? "yes" : "no") << "\n";
    std::cout << "seed: " << seed << "\n";
    std::cout << "use GPU: " << (use_gpu ? "yes" : "no") << "\n";
    std::cout << "use shared GPU: " << (use_gpu_shared ? "yes" : "no") << "\n";
    std::cout << "use KMeans++: " << (use_kmeanspp ? "yes" : "no") << "\n";

    // Parse the input file to get the number of points and the data points
    std::ifstream input_file(inputfilename);
    if (!input_file.is_open())
    {
        std::cerr << "Error: Could not open input file: " << inputfilename << std::endl;
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
        for (int j = 0; j < dims; j++)
        {
            float value;
            input_file >> value;
            point.push_back(value);
        }
        data_points.push_back(point);
    }
    input_file.close();

    // Main logic
    std::cout << "Launching hello kernel" << std::endl;

    // CPU implementation
    std::vector<int> centroid_indices = kmeans_init_centroids(k, num_pts);
    std::vector<std::vector<float>> centroids;

    for (int i = 0; i < centroid_indices.size(); i++)
    {
        std::vector<float> centroid;
        for (int j = 0; j < dims; j++)
        {
            centroid.push_back(data_points[centroid_indices[i]][j]);
        }
        centroids.push_back(centroid);
    }

    // Initialize a vector of int indices of size num_pts with 0 filled
    std::vector<int> assignments(num_pts, 0);

    // Iterations
    if (use_gpu)
    {
        launch_kmeans_kernel(data_points, centroids, num_pts, k, max_iter, dims);
    }
    else
    {
        for (int i = 0; i < max_iter; i++)
        {
            // Counts of vectors in each cluster
            std::vector<int> counts(k, 0);
            std::vector<std::vector<float>> new_centroids(k, std::vector<float>(dims, 0));
            float loss = 0.0f;

            // Step 1: assign points to nearest centroid
            for (int j = 0; j < num_pts; j++)
            {
                int min_index = -1;
                float min_distance = std::numeric_limits<float>::max();
                for (int l = 0; l < k; l++)
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
                for (int d = 0; d < dims; d++)
                {
                    new_centroids[min_index][d] += data_points[j][d];
                }
            }
            // Step 2: update centroids
            for (int j = 0; j < k; j++)
            {
                for (int d = 0; d < dims; d++)
                {
                    new_centroids[j][d] /= counts[j];
                }
            }
            // Check shift
            float shift = 0.0f;
            for (int j = 0; j < k; j++)
            {
                shift += euclidean_distance(centroids[j], new_centroids[j]);
            }
            // If shift is less than threshold, break
            std::cout << "Shift in round " << i << " is: " << shift << std::endl;
            if (shift < threshold)
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

    if (reference_answer_filename != "")
    {
        std::ifstream best_answer_file(reference_answer_filename);
        std::vector<std::vector<float>> best_centroids;
        for (int i = 0; i < k; i++)
        {
            int dummy;
            best_answer_file >> dummy;

            std::vector<float> centroid;
            for (int j = 0; j < dims; j++)
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