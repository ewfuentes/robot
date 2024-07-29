
#pragma once

#include <filesystem>

#include "common/sqlite3/sqlite3.hh"
#include "experimental/beacon_sim/work_server_service.grpc.pb.h"

namespace robot::experimental::beacon_sim {
class WorkServer : public proto::WorkServer::Service {
   public:
    explicit WorkServer(const std::filesystem::path &db_path,
                        const std::filesystem::path &results_dir,
                        const std::filesystem::path &experiment_config_path);

   private:
    sqlite3::Database db_;
    std::filesystem::path experiment_config_path_;
};
}  // namespace robot::experimental::beacon_sim
