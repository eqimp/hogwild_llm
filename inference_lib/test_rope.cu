#include "q_rope.h"
#include "rope.cuh"
#include "cuda_check.cuh"
#include <vector>
#include <random>

std::vector<float> random_vector(std::size_t n_elements, int seed) {
    std::vector<float> out(n_elements);
    std::uniform_real_distribution<float> dist(-2.0, 2.0);
    std::mt19937 rng(seed);
    for(int i = 0; i < n_elements; ++i) {
        out[i] = dist(rng);
    }
    return out;
}

int main() {
    int F = 4;
    int W = 2;
    int Hq = 40;
    int S = 1;
    int E = 128;

    std::vector<float> queries = random_vector(W * Hq * S * E, 42);
    std::vector<float> expected(F*W*Hq*S*E);
    std::vector<float> cosines = random_vector(F * W * S * E, 453);
    std::vector<float> sines = random_vector(F * W * S * E, 837);

    rope_cpu(expected.data(), queries.data(), cosines.data(), sines.data(), F, W, Hq, S, E);

    // GPU version
    float* d_queries;
    float* d_result;
    float* d_cosines;
    float* d_sines;
    CUDA_CHECK_THROW(cudaMalloc(&d_queries, queries.size()*sizeof(float)));
    CUDA_CHECK_THROW(cudaMemcpy(d_queries, queries.data(), queries.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_THROW(cudaMalloc(&d_result, expected.size()*sizeof(float)));
    CUDA_CHECK_THROW(cudaMalloc(&d_cosines, cosines.size()*sizeof(float)));
    CUDA_CHECK_THROW(cudaMemcpy(d_cosines, cosines.data(), cosines.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_THROW(cudaMalloc(&d_sines, sines.size()*sizeof(float)));
    CUDA_CHECK_THROW(cudaMemcpy(d_sines, sines.data(), sines.size()*sizeof(float), cudaMemcpyHostToDevice));

    rope_gpu(d_result, d_queries, d_cosines, d_sines, F, W, Hq, S, E);
    CUDA_CHECK_THROW(cudaGetLastError());

    std::vector<float> result(expected.size());
    CUDA_CHECK_THROW(cudaMemcpy(result.data(), d_result, result.size()*sizeof(float), cudaMemcpyDeviceToHost));

    float* h_ptr = expected.data();
    float* d_ptr = result.data();
    for(int f = 0; f < F; ++f) {
        for(int w = 0; w < W; ++w) {
            for(int s = 0; s < S; ++s) {
                for(int h = 0; h < Hq; ++h) {
                    for(int e = 0; e < E; ++e) {
                        if(fabsf(*h_ptr - *d_ptr) > 1e-5 || std::isnan(*h_ptr) || std::isnan(*d_ptr)) {
                            printf("[%d %d %d %d %d] %f %f\n", f, w, s, h, e, *h_ptr, *d_ptr);
                            exit(1);
                        }
                        ++h_ptr;
                        ++d_ptr;
                    }
                }
            }
        }
    }
}
