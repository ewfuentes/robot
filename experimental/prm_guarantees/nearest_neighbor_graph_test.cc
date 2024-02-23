
#include "gtest/gtest.h"

#include "annoy/annoylib.h"
#include "annoy/kissrandom.h"

#include <random>
#include <iostream>

namespace robot::experimental::prm_guarantee {
using AnnoyIndex = Annoy::AnnoyIndex<int, double, Annoy::Euclidean, Annoy::Kiss32Random, Annoy::AnnoyIndexMultiThreadedBuildPolicy>;

TEST(NearestNeighborGraphTest, dim_20_1m_pts) {
    // Sample 1m points in a unit hypercube [0, 1]^n and add them to the index
    constexpr int NUM_PTS = 1000000;
    constexpr int NUM_DIMS = 20;
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    auto annoy_index = AnnoyIndex(NUM_DIMS);

    std::array<double, NUM_DIMS> data = {};
    for (int i = 0; i < NUM_PTS; i++) {
        for (int j = 0; j < NUM_DIMS; j++) {
            data[j] = dist(gen);
        }

        // To avoid reallocating repeatedly, pass in the indices in reverse order
        const int idx = NUM_PTS - i - 1;
        // The index makes a copy of the data, so it's okay overwrite it on the next iteration
        annoy_index.add_item(idx, data.data());

        std::cout << "Adding items to index: " << static_cast<double>(i) / NUM_PTS * 100.0 << " %\r";
    }
    std::cout << std::endl;

    // Build the trees from the added points
    constexpr int NUM_TREES = 2 * NUM_DIMS;
    constexpr int USE_MAX_CORES = -1;
    std::cout << "Building trees" << std::endl;
    annoy_index.build(NUM_TREES, USE_MAX_CORES);
    std::cout << "End building trees" << std::endl;

    // You can save the index to a file with:
    // annoy_index.save(filename)
    // and load an index with:
    // annoy_index.load(filename)

    // Build the nearest neighbor graph
    constexpr int SEARCH_K = -1;
    constexpr int IDX = 0;
    constexpr int NUM_NEIGHBORS = 256;

    std::vector<std::vector<int>> neighbors_by_idx;
    neighbors_by_idx.reserve(NUM_PTS);


    for (int i = 0; i < NUM_PTS; i++) {
        neighbors_by_idx.push_back({});
        neighbors_by_idx.back().reserve(NUM_NEIGHBORS);
        annoy_index.get_nns_by_item(IDX, NUM_NEIGHBORS, SEARCH_K, &neighbors_by_idx.back(), nullptr);

        if (i % 1000 == 0) {
            std::cout << "Computing nearest neighbors: " << static_cast<float>(i) / NUM_PTS * 100.0 << " %\r" << std::flush;
        }
    }
    std::cout << std::endl;

}
}
