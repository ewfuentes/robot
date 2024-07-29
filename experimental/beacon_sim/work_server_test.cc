

#include "experimental/beacon_sim/work_server.hh"

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {

TEST(WorkServerTest, test_table_construction) {
    // Setup
    const std::filesystem::path db_path = "/home/erick/scratch/beacon_sim/20240716/oracle_results.db";
    const std::filesystem::path results_dir = "/home/erick/scratch/beacon_sim/20240716/results";
    const std::filesystem::path experiment_config_path = "/home/erick/scratch/beacon_sim/20240716/experiment_configs";
    WorkServer work_server(db_path, results_dir, experiment_config_path);

    // Action

    // Verification

}

}
