
#pragma once

#include <filesystem>

#include "common/sqlite3/sqlite3.hh"
#include "experimental/beacon_sim/work_server_service.grpc.pb.h"

namespace robot::experimental::beacon_sim {
class WorkServer : public proto::WorkServer::Service {
   public:
    WorkServer(const std::filesystem::path &db_path,
                        const std::filesystem::path &results_dir,
                        const std::filesystem::path &experiment_config_path);
    ~WorkServer();
   WorkServer(const WorkServer &)=delete;
   WorkServer &operator=(const WorkServer &)=delete;

    grpc::Status get_job(grpc::ServerContext *context, const proto::Worker *request,
                         proto::JobInputs *response) override;

    grpc::Status update_job_status(grpc::ServerContext *context, const proto::JobStatusUpdate *request,
                         proto::JobStatusUpdateResponse *response) override;

    grpc::Status submit_job_result(grpc::ServerContext *context, const proto::JobResult *request,
                         proto::JobResultResponse *response) override;

    grpc::Status get_progress(grpc::ServerContext *context, const proto::ProgressRequest *request,
                         proto::ProgressResponse *response) override;

   private:
    sqlite3::Database db_;
    std::filesystem::path experiment_config_path_;
};
}  // namespace robot::experimental::beacon_sim
