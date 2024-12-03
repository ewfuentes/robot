
#include "experimental/beacon_sim/work_server.hh"

#include <filesystem>
#include <iostream>
#include <sstream>

#include "common/check.hh"
#include "common/proto/load_from_file.hh"
#include "experimental/beacon_sim/work_server_service.pb.h"

namespace robot::experimental::beacon_sim {
namespace {
const std::string JOB_TABLE_NAME = "job_table";
constexpr int CHUNK_SIZE = 200;

std::vector<proto::JobInputs> create_inputs_from_result(
    const std::filesystem::path &result_path, const std::filesystem::path &experiment_config_path) {
    const auto maybe_proto = robot::proto::load_from_file<proto::ExperimentResult>(result_path);
    ROBOT_CHECK(maybe_proto.has_value(), "Failed to load proto", result_path);

    const int num_eval_trials = maybe_proto.value().experiment_config().num_eval_trials();
    std::vector<proto::JobInputs> out;
    out.reserve(num_eval_trials);
    for (int i = 0; i < num_eval_trials; i += CHUNK_SIZE) {
        proto::JobInputs inputs;
        inputs.set_results_file(result_path.string());
        inputs.set_experiment_config_path(experiment_config_path.string());
        inputs.set_eval_start_idx(i);
        inputs.set_eval_end_idx(std::min(num_eval_trials, i + CHUNK_SIZE));
        out.emplace_back(std::move(inputs));
    }
    return out;
}

std::vector<proto::JobInputs> create_inputs(const std::filesystem::path &results_dir,
                                            const std::filesystem::path &experiment_configs_dir) {
    std::vector<proto::JobInputs> out;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(results_dir)) {
        const auto &path = entry.path();
        if (path.extension() == ".pb") {
            const auto result_chunks =
                create_inputs_from_result(path, experiment_configs_dir / path.stem());
            out.insert(out.end(), result_chunks.begin(), result_chunks.end());
        }
    }
    return out;
}
}  // namespace

WorkServer::~WorkServer() = default;

WorkServer::WorkServer(const std::filesystem::path &db_path,
                       const std::filesystem::path &results_dir,
                       const std::filesystem::path &experiment_configs_dir)
    : db_{db_path}, experiment_config_path_(experiment_configs_dir) {
    // Check if the table exists
    const auto rows = db_.query("SELECT count(*) FROM sqlite_master WHERE type='table' and name='" +
                                JOB_TABLE_NAME + "'");
    ROBOT_CHECK(rows.size() == 1);
    const bool does_table_exist = std::get<int>(rows.at(0).value(0));
    if (does_table_exist) {
        // Table exists
        return;
    }

    // Get the chunks
    std::vector<proto::JobInputs> inputs = create_inputs(results_dir, experiment_configs_dir);

    // Insert them into the table
    db_.query("CREATE TABLE " + JOB_TABLE_NAME +
              " (id INTEGER PRIMARY KEY ASC, job_inputs BLOB NOT NULL, job_status BLOB, job_result "
              "BLOB) STRICT;");

    const int num_chunks = inputs.size();
    constexpr int MAX_INSERTION_SIZE = 1000;
    for (int i = 0; i < num_chunks; i += MAX_INSERTION_SIZE) {
        std::ostringstream oss;
        std::unordered_map<std::string, sqlite3::Database::Value> to_bind;
        oss << "INSERT INTO " + JOB_TABLE_NAME + " (rowid, job_inputs) VALUES ";
        for (int j = 0; j < std::min(MAX_INSERTION_SIZE, num_chunks - i); j++) {
            const int idx = i + j;
            auto &job_input = inputs.at(idx);
            std::vector<unsigned char> blob(job_input.ByteSizeLong());
            job_input.SerializeToArray(blob.data(), blob.size());

            if (j > 0) {
                oss << ",";
            }
            const std::string row_id = ":row_id" + std::to_string(j);
            const std::string blob_id = ":inputs" + std::to_string(j);
            oss << "(" << row_id << "," << blob_id << ")";
            to_bind.insert({row_id, sqlite3::Database::Value(idx + 1)});
            to_bind.insert({blob_id, sqlite3::Database::Value(blob)});
        }
        const auto statement = db_.prepare(oss.str());
        db_.bind(statement, to_bind);
        db_.step(statement);
    }
}

grpc::Status WorkServer::get_job(grpc::ServerContext *, const proto::GetJobRequest *request,
                                 proto::GetJobResponse *response) {
    // Find the first job
    const auto rows = db_.query("SELECT id, job_inputs FROM " + JOB_TABLE_NAME +
                                " WHERE job_status IS NULL LIMIT 1;");
    if (rows.empty()) {
        response->Clear();
        return grpc::Status::OK;
    }

    const int job_id = std::get<int>(rows.at(0).value(0));
    std::cout << "allocating job id " << job_id << std::endl;
    {
        const std::vector<unsigned char> &blob =
            std::get<std::vector<unsigned char>>(rows.at(0).value(1));
        proto::JobInputs job_inputs;
        job_inputs.ParseFromArray(blob.data(), blob.size());

        response->set_job_id(job_id);
        response->mutable_job_inputs()->CopyFrom(job_inputs);
    }

    // Set an initial job status so it doesn't get requested again
    {
        proto::JobStatusUpdate update;
        update.mutable_worker()->CopyFrom(request->worker());
        std::vector<unsigned char> blob(update.ByteSizeLong());
        update.SerializeToArray(blob.data(), blob.size());

        const auto statement = db_.prepare("UPDATE " + JOB_TABLE_NAME +
                                           " SET job_status = :job_status WHERE id = :id;");
        db_.bind(statement, {{":id", job_id}, {":job_status", blob}});
        db_.step(statement);
    }

    return grpc::Status::OK;
}

grpc::Status WorkServer::update_job_status(grpc::ServerContext *,
                                           const proto::JobStatusUpdateRequest *request,
                                           proto::JobStatusUpdateResponse *response) {
    std::cout << "updating job status for row: " << request->DebugString() << std::endl;
    const int job_id = request->job_id();
    const proto::JobStatusUpdate &update = request->update();
    std::vector<unsigned char> blob(update.ByteSizeLong());
    update.SerializeToArray(blob.data(), blob.size());

    const auto statement =
        db_.prepare("UPDATE " + JOB_TABLE_NAME + " SET job_status = :job_status WHERE id = :id;");
    db_.bind(statement, {{":id", job_id}, {":job_status", blob}});
    db_.step(statement);

    response->Clear();
    return grpc::Status::OK;
}

grpc::Status WorkServer::submit_job_result(grpc::ServerContext *,
                                           const proto::JobResultRequest *request,
                                           proto::JobResultResponse *response) {
    const int job_id = request->job_id();
    std::cout << "updating job result for row " << job_id << std::endl;
    const proto::JobResult &result = request->job_result();
    std::vector<unsigned char> blob(result.ByteSizeLong());
    result.SerializeToArray(blob.data(), blob.size());

    const auto statement =
        db_.prepare("UPDATE " + JOB_TABLE_NAME + " SET job_result = :job_result WHERE id = :id;");
    db_.bind(statement, {{":id", job_id}, {":job_result", blob}});
    db_.step(statement);

    response->Clear();

    return grpc::Status::OK;
}

grpc::Status WorkServer::get_progress(grpc::ServerContext *, const proto::ProgressRequest *,
                                      proto::ProgressResponse *response) {
    int jobs_completed;
    int jobs_remaining;
    int jobs_in_progress;

    {
        const auto rows =
            db_.query("SELECT count(*) FROM " + JOB_TABLE_NAME + " WHERE job_result IS NOT NULL;");
        ROBOT_CHECK(!rows.empty());
        jobs_completed = std::get<int>(rows.at(0).value(0));
    }
    {
        const auto rows =
            db_.query("SELECT count(*) FROM " + JOB_TABLE_NAME + " WHERE job_status IS NULL;");
        ROBOT_CHECK(!rows.empty());
        jobs_remaining = std::get<int>(rows.at(0).value(0));
    }
    {
        const auto rows = db_.query("SELECT count(*) FROM " + JOB_TABLE_NAME +
                                    " WHERE job_status IS NOT NULL AND job_result IS NULL;");
        ROBOT_CHECK(!rows.empty());
        jobs_in_progress = std::get<int>(rows.at(0).value(0));
    }
    response->set_jobs_completed(jobs_completed);
    response->set_jobs_remaining(jobs_remaining);
    response->set_jobs_in_progress(jobs_in_progress);

    return grpc::Status::OK;
}
}  // namespace robot::experimental::beacon_sim
