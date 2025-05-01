// kernel.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

__global__ void distanceForPoints(float *data_points, float *centroids, float *distance_for_points, int num_pts, int k, int dims)
{
    // Computes the distance for each point / centroid pair
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = num_pts * k;

    if (tid < totalThreads)
    {
        int i = tid / k; // point index
        int j = tid % k; // centroid index

        float *point = &data_points[i * dims];
        float *centroid = &centroids[j * dims];

        float dist = 0.0;
        for (int d = 0; d < dims; d++)
        {
            dist += (point[d] - centroid[d]) * (point[d] - centroid[d]);
        }
        distance_for_points[tid] = sqrt(dist);
    }
}
__global__ void sumNewCentroids(float *data_points, float *new_centroids, int *assignments, int num_pts, int k, int dims)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = num_pts * dims;

    if (tid < totalThreads)
    {
        int i = tid / dims; // point index
        int j = tid % dims; // dimension index

        int centroid_idx = assignments[i];

        float value = data_points[i * dims + j];
        atomicAdd(&new_centroids[centroid_idx * dims + j], value);
    }
}
__global__ void normalizeNewCentroids(float *new_centroids, int *counts, int k, int dims)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = k * dims;
    if (tid < totalThreads)
    {
        int centroid_idx = tid / dims;
        int dimension_idx = tid % dims;

        new_centroids[centroid_idx * dims + dimension_idx] /= counts[centroid_idx];
    }
}

__global__ void checkConvergence(float *centroids, float *new_centroids, float *loss, int k, int dims)
{
    extern __shared__ float shared_loss[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float local_loss = 0.0f;
    if (gid < k * dims)
    {
        local_loss = (centroids[gid] - new_centroids[gid]) * (centroids[gid] - new_centroids[gid]);
    }

    shared_loss[tid] = local_loss;
    __syncthreads();

    // Reduction via strides
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_loss[tid] += shared_loss[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(loss, shared_loss[0]);
    }
}
extern "C" void launch_kmeans_kernel(std::vector<std::vector<float>> &data_points, std::vector<std::vector<float>> &centroids, int num_pts, int k, int max_iter, int dims, float tolerance)
{
    float *d_data_points;
    float *d_centroids;
    float *d_new_centroids;
    int *d_assignments;
    float *d_distance_for_points;
    float *d_loss;

    // flatten the data points
    std::vector<float> flat_data_points;
    for (const auto &pt : data_points)
    {
        flat_data_points.insert(flat_data_points.end(), pt.begin(), pt.end());
    }
    assert(flat_data_points.size() == num_pts * dims);
    std::vector<float> flat_centroids;
    for (const auto &ct : centroids)
    {
        flat_centroids.insert(flat_centroids.end(), ct.begin(), ct.end());
    }
    assert(flat_centroids.size() == k * dims);

    // Allocate memory on the device
    cudaMalloc(&d_data_points, sizeof(float) * num_pts * dims);
    cudaMalloc(&d_centroids, sizeof(float) * k * dims);
    cudaMalloc(&d_assignments, sizeof(int) * num_pts);
    cudaMalloc(&d_distance_for_points, sizeof(float) * num_pts * k); // distance for each point to each centroid
    cudaMalloc(&d_new_centroids, sizeof(float) * k * dims);
    cudaMalloc(&d_loss, sizeof(float));

    // Copy data to the device
    cudaMemcpy(d_data_points, flat_data_points.data(), sizeof(float) * num_pts * dims, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, flat_centroids.data(), sizeof(float) * k * dims, cudaMemcpyHostToDevice);
    cudaMemset(d_assignments, 0, sizeof(int) * num_pts);

    // Required for min reduction
    int *d_offsets;
    std::vector<int> offsets(num_pts + 1);
    for (int i = 0; i <= num_pts; i++)
    {
        offsets[i] = i * k;
    }
    cudaMalloc(&d_offsets, sizeof(int) * (num_pts + 1));
    cudaMemcpy(d_offsets, offsets.data(), sizeof(int) * (num_pts + 1), cudaMemcpyHostToDevice);
    cub::KeyValuePair<int, float> *d_argmin_output;
    cudaMalloc(&d_argmin_output, sizeof(cub::KeyValuePair<int, float>) * num_pts);
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_distance_for_points, d_argmin_output, num_pts, d_offsets, d_offsets + 1);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Required for histogram reduction
    int *d_cluster_sizes;
    cudaMalloc(&d_cluster_sizes, sizeof(int) * k);
    void *d_temp_histogram_storage = nullptr;
    size_t temp_histogram_storage_bytes = 0;
    cub::DeviceHistogram::HistogramEven(d_temp_histogram_storage, temp_histogram_storage_bytes, d_assignments, d_cluster_sizes, k + 1, 0, k, num_pts);
    cudaMalloc(&d_temp_histogram_storage, temp_histogram_storage_bytes);

    // Determine how many threads to launch
    int totalThreads = num_pts * k;
    std::cout << "totalThreads: " << totalThreads << std::endl;
    int threads_per_block = 1024;
    int blocks_per_grid = (totalThreads + threads_per_block - 1) / threads_per_block; // ceil of num_pts / threads_per_block

    // Threads for the divide operation
    int totalThreadsDivide = k * dims;
    int blocks_per_grid_divide = (totalThreadsDivide + threads_per_block - 1) / threads_per_block;

    // Threads for the sum operation
    int totalThreadsSum = num_pts * dims;
    int blocks_per_grid_sum = (totalThreadsSum + threads_per_block - 1) / threads_per_block;

    // Threads for the check convergence operation
    int threads = 256;
    int total = k * dims;
    int blocks = (total + threads - 1) / threads;
    float sharedMemSize = threads * sizeof(float);
    float loss;

    for (int i = 0; i < max_iter; i++)
    {
        // 0. Zero out the new centroids
        cudaMemset(d_new_centroids, 0, sizeof(float) * k * dims);
        cudaMemset(d_cluster_sizes, 0, sizeof(int) * k);
        cudaMemset(d_loss, 0, sizeof(float));

        // 1. Distance for each point to each centroid
        distanceForPoints<<<blocks_per_grid, threads_per_block>>>(d_data_points, d_centroids, d_distance_for_points, num_pts, k, dims);

        // 2. Get the minimum distance for each point -- reduction from num_pts * k -> num_pts (This requires two steps: https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html#_CPPv4N3cub12DeviceReduceE)
        cub::DeviceSegmentedReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_distance_for_points, d_argmin_output, num_pts, d_offsets, d_offsets + 1);
        thrust::transform(thrust::device, d_argmin_output, d_argmin_output + num_pts, d_assignments, [] __device__(cub::KeyValuePair<int, float> kvp)
                          { return kvp.key; });

        // 3. Count number of points assigned to each centroid
        cub::DeviceHistogram::HistogramEven(d_temp_histogram_storage, temp_histogram_storage_bytes, d_assignments, d_cluster_sizes, k + 1, 0, k, num_pts);

        // 4. Sum the group of points assigned to each centroid
        sumNewCentroids<<<blocks_per_grid_sum, threads_per_block>>>(d_data_points, d_new_centroids, d_assignments, num_pts, k, dims);

        // 5. Normalize the new centroids
        normalizeNewCentroids<<<blocks_per_grid_divide, threads_per_block>>>(d_new_centroids, d_cluster_sizes, k, dims);

        // 6. Check convergence
        checkConvergence<<<blocks, threads, sharedMemSize>>>(d_centroids, d_new_centroids, d_loss, k, dims);

        // 7. Copy shit
        cudaMemcpy(d_centroids, d_new_centroids, sizeof(float) * k * dims, cudaMemcpyDeviceToDevice);

        cudaDeviceSynchronize();
        cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        if (sqrt(loss) < tolerance)
        {
            printf("Converged in %d iterations\n", i);
            break;
        }
    }

    // Copy centroids back to the host
    std::vector<float> host_centroids(k * dims);
    cudaMemcpy(host_centroids.data(), d_centroids, sizeof(float) * k * dims, cudaMemcpyDeviceToHost);

    // Copy assignments to centorids
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < dims; j++)
        {
            centroids[i][j] = host_centroids[i * dims + j];
        }
    }

    // return vector of vectors
    cudaFree(d_data_points);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_cluster_sizes);
    cudaFree(d_distance_for_points);
    cudaFree(d_temp_storage);
    cudaFree(d_temp_histogram_storage);
    cudaFree(d_offsets);
    cudaFree(d_new_centroids);
    cudaFree(d_argmin_output);
    cudaFree(d_temp_storage);
    return;
}