
#include <filesystem>
#include <limits>
#include <optional>
#include <random>

#include "BS_thread_pool.hpp"
#include "common/argument_wrapper.hh"
#include "common/check.hh"
#include "common/math/matrix_to_proto.hh"
#include "common/proto/load_from_file.hh"
#include "common/sqlite3/sqlite3.hh"
#include "cxxopts.hpp"
#include "experimental/beacon_sim/beacon_potential.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/experiment_results.pb.h"
#include "experimental/beacon_sim/make_belief_updater.hh"
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"
#include "experimental/beacon_sim/robot_belief.hh"
#include "experimental/beacon_sim/world_map_config_to_proto.hh"
#include "planning/belief_road_map.hh"
#include "planning/road_map.hh"
#include "planning/road_map_to_proto.hh"

namespace robot::experimental::beacon_sim {
namespace {
struct StartGoal {
    Eigen::Vector2d start;
    Eigen::Vector2d goal;
};

EkfSlam load_ekf_slam(const MappedLandmarks &mapped_landmarks) {
    EkfSlam ekf(
        {
            .max_num_beacons = static_cast<int>(mapped_landmarks.beacon_ids.size()),
            .initial_beacon_uncertainty_m = 100,
            .along_track_process_noise_m_per_rt_meter = 5e-2,
            .cross_track_process_noise_m_per_rt_meter = 1e-9,
            .pos_process_noise_m_per_rt_s = 1e-3,
            .heading_process_noise_rad_per_rt_meter = 1e-3,
            .heading_process_noise_rad_per_rt_s = 1e-10,
            .beacon_pos_process_noise_m_per_rt_s = 1e-3,
            .range_measurement_noise_m = 0.1,
            .bearing_measurement_noise_rad = 0.01,
            .on_map_load_position_uncertainty_m = 1.0,
            .on_map_load_heading_uncertainty_rad = 1.0,
        },
        {});

    constexpr bool LOAD_OFF_DIAGONALS = true;
    ekf.load_map(mapped_landmarks, LOAD_OFF_DIAGONALS);

    return ekf;
}

std::tuple<BeaconPotential, planning::RoadMap, EkfSlam> load_environment(
    const proto::ExperimentConfig &config, const std::filesystem::path &config_path) {
    // Load Map Config
    const auto maybe_map_config = robot::proto::load_from_file<proto::WorldMapConfig>(
        config_path.parent_path() / config.map_config_path());
    CHECK(maybe_map_config.has_value());
    const WorldMap world_map(unpack_from(maybe_map_config.value()));

    // Load EKF State
    const auto maybe_mapped_landmarks = robot::proto::load_from_file<proto::MappedLandmarks>(
        config_path.parent_path() / config.ekf_state_path());
    CHECK(maybe_mapped_landmarks.has_value());
    const auto mapped_landmarks = unpack_from(maybe_mapped_landmarks.value());

    // Create a road map
    const auto maybe_road_map = robot::proto::load_from_file<planning::proto::RoadMap>(
        config_path.parent_path() / config.road_map_path());
    CHECK(maybe_road_map.has_value());
    const planning::RoadMap road_map = unpack_from(maybe_road_map.value());

    return {world_map.beacon_potential(), road_map, load_ekf_slam(mapped_landmarks)};
}

std::vector<std::vector<proto::PlanMetrics>> load_oracle_metrics(
    const std::filesystem::path &oracle_db_path, const std::string &experiment_name) {
    sqlite3::Database oracle_db(oracle_db_path);

    const int exp_id = [&oracle_db, &experiment_name]() -> int {
        const auto statement =
            oracle_db.prepare("SELECT rowid FROM experiment_names WHERE exp_name = :exp_name");
        oracle_db.bind(statement, {{":exp_name", experiment_name}});
        return std::get<int>(oracle_db.step(statement).value().value(0));
    }();

    const sqlite3::Database::Statement stmt = oracle_db.prepare(
        "SELECT trial_id, eval_id, prob_mass_in_region FROM oracle_results WHERE exp_type = "
        ":exp_id ORDER BY eval_id ASC, trial_id ASC;");
    oracle_db.bind(stmt, {{":exp_id", exp_id}});
    std::optional<sqlite3::Database::Row> maybe_row;
    std::vector<std::vector<proto::PlanMetrics>> out;
    while ((maybe_row = oracle_db.step(stmt)).has_value()) {
        const auto &row = maybe_row.value();
        const int &trial_id = std::get<int>(row.value(0));
        const int &eval_trial_id = std::get<int>(row.value(1));
        const double &prob_mass_in_region = std::get<double>(row.value(2));
        if (trial_id == 0) {
            out.push_back({});
        }
        CHECK(out.size() == eval_trial_id +1);
        
        out.back().push_back({});
        auto &to_fill = out.back().back();
        to_fill.set_prob_mass_in_region(prob_mass_in_region);
    }
    return out;
}
}  // namespace

void reprocess_result(const proto::ExperimentResult &result,
                      const std::filesystem::path &experiment_config_path,
                      const std::filesystem::path &output_path,
                      const std::filesystem::path &oracle_db_path, const int num_threads) {
    proto::ExperimentResult out = result;
    const auto &experiment_config = result.experiment_config();
    // Load the beacon potential, road map and ekf from the config file
    const auto &[beacon_potential, road_map_, ekf_] =
        load_environment(experiment_config, experiment_config_path);
    planning::RoadMap road_map = road_map_;
    EkfSlam ekf = ekf_;

    // Create the uncertainty size metrics
    const auto expected_det_metric =
        make_uncertainty_size<RobotBelief>(ExpectedDeterminant{.position_only = false});
    const auto expected_pos_det_metric =
        make_uncertainty_size<RobotBelief>(ExpectedDeterminant{.position_only = true});
    const auto prob_mass_in_region_metric = make_uncertainty_size<RobotBelief>(ProbMassInRegion{
        .position_x_half_width_m = 0.5,
        .position_y_half_width_m = 0.5,
        .heading_half_width_rad = 6.0,
    });

    const auto expected_trace_metric =
        make_uncertainty_size<RobotBelief>(ExpectedTrace{.position_only = false});
    const auto expected_position_trace_metric =
        make_uncertainty_size<RobotBelief>(ExpectedTrace{.position_only = true});

    // Recompute the metrics for each run
    std::vector<std::vector<int>> samples;
    std::mt19937 gen(experiment_config.evaluation_base_seed());
    for (int trial_id = 0; trial_id < experiment_config.num_eval_trials(); trial_id++) {
        samples.push_back(beacon_potential.sample(make_in_out(gen)));
        const auto &sample = samples.back();

        std::string sample_string(beacon_potential.members().size(), '0');
        for (const auto member : beacon_potential.members()) {
            const auto iter = std::find(sample.begin(), sample.end(), member);
            if (iter != sample.end()) {
                const auto idx = std::distance(sample.begin(), iter);
                sample_string[idx] = '1';
            }
        }
        out.add_landmark_eval_samples(std::move(sample_string));
    }

    // Get a list of the start goals
    std::vector<StartGoal> start_goals;
    for (const auto &proto_sg : result.start_goal()) {
        start_goals.push_back({
            .start = unpack_from<Eigen::Vector2d>(proto_sg.start()),
            .goal = unpack_from<Eigen::Vector2d>(proto_sg.goal()),
        });
    }

    const auto &oracle_metrics_by_eval_trial_id =
        load_oracle_metrics(oracle_db_path, experiment_config.name());

    BS::thread_pool pool(num_threads);
    std::vector<proto::PlannerResult> new_results(out.results_size());
    for (int i = 0; i < out.results_size(); i++) {
        new_results.at(i).CopyFrom(result.results(i));
        pool.detach_task([=, &samples, &experiment_result = result, &out = new_results.at(i),
                          &experiment_config, &oracle_metrics_by_eval_trial_id]() mutable {
            double expected_det = 0.0;
            double expected_pos_det = 0.0;
            double expected_prob = 0.0;
            double expected_trace = 0.0;
            double expected_position_trace = 0.0;

            double expected_prob_regret = 0.0;
            out.mutable_plan()->mutable_sampled_plan_metrics()->Reserve(
                experiment_config.num_eval_trials());
            for (int eval_trial_id = 0; eval_trial_id < experiment_config.num_eval_trials();
                 eval_trial_id++) {
                const auto &present_beacons = samples.at(eval_trial_id);
                const Eigen::Vector2d start_in_map = unpack_from<Eigen::Vector2d>(
                    experiment_result.start_goal(out.trial_id()).start());
                const Eigen::Vector2d goal_in_map = unpack_from<Eigen::Vector2d>(
                    experiment_result.start_goal(out.trial_id()).goal());
                road_map.add_start_goal(
                    {.start = start_in_map,
                     .goal = goal_in_map,
                     .connection_radius_m = experiment_config.start_goal_connection_radius_m()});
                const liegroups::SE2 map_from_robot = liegroups::SE2::trans(start_in_map);
                ekf.estimate().local_from_robot(map_from_robot);
                RobotBelief belief{
                    .local_from_robot = ekf.estimate().local_from_robot(),
                    .cov_in_robot = ekf.estimate().robot_cov(),
                };

                const auto belief_updater =
                    make_belief_updater(road_map, experiment_config.max_sensor_range_m(), ekf,
                                        present_beacons, TransformType::COVARIANCE);

                for (int plan_node_idx = 0; plan_node_idx < out.plan().nodes_size() - 1;
                     plan_node_idx++) {
                    belief = belief_updater(belief, out.plan().nodes(plan_node_idx),
                                            out.plan().nodes(plan_node_idx + 1));
                }

                const auto &oracle_metrics =
                    oracle_metrics_by_eval_trial_id.at(eval_trial_id).at(out.trial_id());
                const double sampled_expected_det = expected_det_metric(belief);
                const double sampled_expected_pos_det = expected_pos_det_metric(belief);
                const double sampled_prob_mass_in_region = prob_mass_in_region_metric(belief);
                const double sampled_expected_trace = expected_trace_metric(belief);
                const double sampled_expected_pos_trace = expected_position_trace_metric(belief);

                const double prob_regret =
                    sampled_prob_mass_in_region - oracle_metrics.prob_mass_in_region();

                expected_det += sampled_expected_det / experiment_config.num_eval_trials();
                expected_pos_det += sampled_expected_pos_det / experiment_config.num_eval_trials();
                expected_prob += sampled_prob_mass_in_region / experiment_config.num_eval_trials();
                expected_trace += sampled_expected_trace / experiment_config.num_eval_trials();
                expected_position_trace +=
                    sampled_expected_pos_trace / experiment_config.num_eval_trials();

                expected_prob_regret += (prob_regret) / experiment_config.num_eval_trials();

                auto &sampled_metrics = *out.mutable_plan()->add_sampled_plan_metrics();
                sampled_metrics.set_expected_determinant(sampled_expected_det);
                sampled_metrics.set_expected_position_determinant(sampled_expected_pos_det);
                sampled_metrics.set_prob_mass_in_region(sampled_prob_mass_in_region);
                sampled_metrics.set_expected_trace(sampled_expected_trace);
                sampled_metrics.set_expected_position_trace(sampled_expected_pos_trace);
                sampled_metrics.set_prob_mass_in_region_regret(prob_regret);
            }
            // curr_result.mutable_plan()->clear_expected_size();
            out.mutable_plan()->mutable_average_plan_metrics()->set_expected_determinant(
                expected_det);
            out.mutable_plan()->mutable_average_plan_metrics()->set_expected_position_determinant(
                expected_pos_det);
            out.mutable_plan()->mutable_average_plan_metrics()->set_prob_mass_in_region(
                expected_prob);
            out.mutable_plan()->mutable_average_plan_metrics()->set_expected_trace(expected_trace);
            out.mutable_plan()->mutable_average_plan_metrics()->set_expected_position_trace(
                expected_position_trace);
            out.mutable_plan()->mutable_average_plan_metrics()->set_prob_mass_in_region_regret(
                expected_prob_regret);
            std::cout << out.trial_id() << " " << out.planner_id() << std::endl;
        });
    }

    pool.wait();
    out.clear_results();
    out.mutable_results()->Add(new_results.begin(), new_results.end());

    // Write out the file
    std::filesystem::create_directories(output_path.parent_path());
    {
        std::ofstream os(output_path);
        CHECK(os.good());
        out.SerializeToOstream(&os);
    }
}
}  // namespace robot::experimental::beacon_sim

int main(int argc, const char **argv) {
    // clang-format off
    const std::string DEFAULT_NUM_THREADS = "0";
    cxxopts::Options options("reprocess_result",
                             "Add additional uncertainty metrics to ExperimentResult protos");
    options.add_options()
        ("results_file", "Path to results file", cxxopts::value<std::string>())
        ("experiment_config_path", "Path to Experiment Config file", cxxopts::value<std::string>())
        ("output_path", "Output path", cxxopts::value<std::string>())
        ("oracle_db", "Path to Oracle DB", cxxopts::value<std::string>())
        ("num_threads", "Number of threads to use",
            cxxopts::value<int>()->default_value(DEFAULT_NUM_THREADS))
        ("help", "Print usage");
    // clang-format on

    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    const auto check_required = [&](const std::string &opt) {
        if (args.count(opt) == 0) {
            std::cout << "Missing " << opt << " argument" << std::endl;
            std::cout << options.help() << std::endl;
            std::exit(1);
        }
    };

    check_required("results_file");
    check_required("experiment_config_path");
    check_required("output_path");
    check_required("oracle_db");

    const std::filesystem::path results_file_path = args["results_file"].as<std::string>();
    const auto maybe_results_file =
        robot::proto::load_from_file<robot::experimental::beacon_sim::proto::ExperimentResult>(
            results_file_path);
    CHECK(maybe_results_file.has_value());

    robot::experimental::beacon_sim::reprocess_result(
        maybe_results_file.value(),
        std::filesystem::path(args["experiment_config_path"].as<std::string>()),
        std::filesystem::path(args["output_path"].as<std::string>()),
        std::filesystem::path(args["oracle_db"].as<std::string>()), args["num_threads"].as<int>());
}
