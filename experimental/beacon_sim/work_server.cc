
#include "experimental/beacon_sim/work_server.hh"

#include <filesystem>
#include <iterator>
#include <sstream>

#include "common/check.hh"
#include "experimental/beacon_sim/work_server_service.pb.h"
#include "common/proto/load_from_file.hh"

namespace robot::experimental::beacon_sim {
namespace {
const std::string JOB_TABLE_NAME = "job_table";
constexpr int CHUNK_SIZE = 1;

std::vector<proto::JobInputs> create_inputs_from_result(
    [[maybe_unused]] const std::filesystem::path &result_path,
    [[maybe_unused]] const std::filesystem::path &experiment_config_path) {

    const auto maybe_proto = robot::proto::load_from_file<proto::ExperimentResult>(result_path);
    CHECK(maybe_proto.has_value(), "Failed to load proto", result_path);

    const int num_eval_trials = maybe_proto.value().experiment_config().num_eval_trials();
    std::vector<proto::JobInputs> out;
    out.reserve(num_eval_trials);
    for (int i = 0; i < num_eval_trials; i += CHUNK_SIZE) {
        proto::JobInputs inputs;
        inputs.set_results_file(result_path.string());
        inputs.set_experiment_config_path(experiment_config_path.string());
        inputs.set_start_idx(i);
        inputs.set_end_idx(std::min(num_eval_trials, i+CHUNK_SIZE));
        out.emplace_back(std::move(inputs));
    }
    return out;
}

std::vector<proto::JobInputs> create_inputs(
    const std::filesystem::path &results_dir,
    const std::filesystem::path &experiment_configs_dir) {
    std::vector<proto::JobInputs> out;
    for (const auto &entry: std::filesystem::recursive_directory_iterator(results_dir)) {
        const auto &path = entry.path();
        if (path.extension() == ".pb") {
            const auto result_chunks = create_inputs_from_result(path, experiment_configs_dir / path.stem());
            out.insert(out.end(), result_chunks.begin(), result_chunks.end());
        }

    }
    return out;
}
}  // namespace

WorkServer::WorkServer(const std::filesystem::path &db_path,
                       const std::filesystem::path &results_dir,
                       const std::filesystem::path &experiment_configs_dir)
    : db_{db_path}, experiment_config_path_(experiment_configs_dir) {
    // Check if the table exists
    const auto rows = db_.query("SELECT count(*) FROM sqlite_master WHERE type='table' and name='" +
                                JOB_TABLE_NAME + "'");
    CHECK(rows.size() == 1);
    const bool does_table_exist = std::get<int>(rows.at(0).value(0));
    if (does_table_exist) {
        // Table exists
        std::cout << "table exists" << std::endl;
        return;
    }

    // Get the chunks
    std::vector<proto::JobInputs> inputs = create_inputs(results_dir, experiment_configs_dir);
    std::cout << "num inputs: " << inputs.size() << std::endl;

    // Insert them into the table
    std::cout << "table does not exist" << std::endl;
    db_.query("CREATE TABLE " + JOB_TABLE_NAME +
              " (id INTEGER PRIMARY KEY ASC, job_inputs BLOB NOT NULL, job_status BLOB, result "
              "BLOB) STRICT;");


    std::ostringstream oss;
    std::unordered_map<std::string, sqlite3::Database::Value> to_bind;
    oss << "INSERT INTO " + JOB_TABLE_NAME + " (rowid, job_inputs) VALUES ";
    for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
        std::cout << "Inserting " << i << " / " << inputs.size() << "\r";
        auto &job_input = inputs.at(i+1);
        job_input.set_job_id(i+1);
        std::string blob;
        job_input.SerializeToString(&blob);
        std::vector<unsigned char> blob_vec(blob.begin(), blob.end());

        if (i > 0) {
            oss << ",";
        }
        const std::string row_id = ":row_id" + std::to_string(i);
        const std::string blob_id = ":inputs" + std::to_string(i);
        oss << "(" << row_id << "," << blob_id << ")";
        to_bind[row_id] = sqlite3::Database::Value(i+1);
        to_bind[blob_id] = sqlite3::Database::Value(blob_vec);
    }
    std::cout << std::endl;
    const auto statement = db_.prepare(oss.str());
    db_.bind(statement, to_bind);
    db_.step(statement);
}

}  // namespace robot::experimental::beacon_sim
