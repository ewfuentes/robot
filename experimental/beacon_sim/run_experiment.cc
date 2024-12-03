
#include <algorithm>
#include <execution>
#include <filesystem>
#include <limits>
#include <optional>
#include <random>

#include "BS_thread_pool.hpp"
#include "Eigen/Core"
#include "common/argument_wrapper.hh"
#include "common/check.hh"
#include "common/math/matrix_to_proto.hh"
#include "common/proto/load_from_file.hh"
#include "common/time/robot_time.hh"
#include "common/time/robot_time_to_proto.hh"
#include "cxxopts.hpp"
#include "experimental/beacon_sim/beacon_potential.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/experiment_config.pb.h"
#include "experimental/beacon_sim/experiment_results.pb.h"
#include "experimental/beacon_sim/make_belief_updater.hh"
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"
#include "experimental/beacon_sim/robot_belief.hh"
#include "experimental/beacon_sim/world_map_config_to_proto.hh"
#include "planning/road_map_to_proto.hh"

using robot::experimental::beacon_sim::proto::ExperimentConfig;

namespace robot::experimental::beacon_sim {
namespace {

struct StartGoal {
    Eigen::Vector2d start;
    Eigen::Vector2d goal;
};

struct PlannerResult {
    struct Plan {
        std::vector<int> nodes;
        double log_prob_mass_tracked;
    };
    time::RobotTimestamp::duration elapsed_time;
    std::optional<Plan> plan;
};

std::vector<std::tuple<int, StartGoal>> sample_start_goal(const planning::RoadMap &map,
                                                          const int num_trials,
                                                          InOut<std::mt19937> gen) {
    Eigen::Vector2d min = {
        std::min_element(
            map.points().begin(), map.points().end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) { return a.x() < b.x(); })
            ->x(),
        std::min_element(
            map.points().begin(), map.points().end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) { return a.y() < b.y(); })
            ->y(),
    };
    Eigen::Vector2d max = {
        std::max_element(
            map.points().begin(), map.points().end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) { return a.x() < b.x(); })
            ->x(),
        std::max_element(
            map.points().begin(), map.points().end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) { return a.y() < b.y(); })
            ->y(),
    };

    std::vector<std::tuple<int, StartGoal>> out;
    std::uniform_real_distribution<> x_sampler(min.x(), max.x());
    std::uniform_real_distribution<> y_sampler(min.y(), max.y());
    for (int i = 0; i < num_trials; i++) {
        out.push_back(std::make_tuple(i, StartGoal{.start = {x_sampler(*gen), y_sampler(*gen)},
                                                   .goal = {x_sampler(*gen), y_sampler(*gen)}}));
    }
    return out;
}

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

UncertaintySizeOptions from_proto(const proto::UncertaintySize &config) {
    switch (config.uncertainty_size_oneof_case()) {
        case proto::UncertaintySize::kExpectedDeterminant: {
            const auto &us_config = config.expected_determinant();
            return ExpectedDeterminant{
                .position_only = us_config.position_only(),
            };
        }
        case proto::UncertaintySize::kExpectedTrace: {
            const auto &us_config = config.expected_trace();
            return ExpectedTrace{
                .position_only = us_config.position_only(),
            };
        }
        case proto::UncertaintySize::kValueAtRiskDeterminant: {
            const auto &us_config = config.value_at_risk_determinant();
            return ValueAtRiskDeterminant{
                .percentile = us_config.percentile(),
            };
        }
        case proto::UncertaintySize::kProbMassInRegion: {
            const auto &us_config = config.prob_mass_in_region();
            return ProbMassInRegion{
                .position_x_half_width_m = us_config.position_x_half_width_m(),
                .position_y_half_width_m = us_config.position_y_half_width_m(),
                .heading_half_width_rad = us_config.heading_half_width_rad(),
            };
        }
        case proto::UncertaintySize::UNCERTAINTY_SIZE_ONEOF_NOT_SET: {
            ROBOT_CHECK(false, "uncertainty size not set");
            return ExpectedDeterminant{
                .position_only = false,
            };
        }
    }
    ROBOT_CHECK(false, "uncertainty size not set");
    return ExpectedDeterminant{.position_only = false};
}

PlannerResult run_planner(const planning::RoadMap &road_map, const EkfSlam &ekf,
                          const BeaconPotential &beacon_potential,
                          const proto::LandmarkBRMPlanner &config, const double max_sensor_range_m,
                          const std::optional<time::RobotTimestamp::duration> &timeout) {
    const time::RobotTimestamp start_time = time::current_robot_time();

    const auto uncertainty_options = from_proto(config.uncertainty_size());

    const auto plan = compute_landmark_belief_road_map_plan(
        road_map, ekf, beacon_potential,
        {.max_sensor_range_m = max_sensor_range_m,
         .uncertainty_size_options = uncertainty_options,
         .sampled_belief_options =
             config.has_max_num_components()
                 ? std::make_optional(LandmarkBeliefRoadMapOptions::SampledBeliefOptions{
                       .max_num_components = config.max_num_components(), .seed = 12345})
                 : std::nullopt,
         .timeout = timeout});
    const time::RobotTimestamp::duration elapsed_time = time::current_robot_time() - start_time;

    // std::cout << "log probability mass tracked: "
    //           << plan->beliefs.back().log_probability_mass_tracked << std::endl;

    return {.elapsed_time = elapsed_time,
            .plan = plan.has_value() ? std::make_optional(PlannerResult::Plan{
                                           .nodes = plan->nodes,
                                           .log_prob_mass_tracked =
                                               plan->beliefs.back().log_probability_mass_tracked,
                                       })
                                     : std::nullopt};
}

PlannerResult run_planner(const planning::RoadMap &road_map, const EkfSlam &ekf,
                          const BeaconPotential &beacon_potential,
                          const proto::ExpectedBRMPlanner &config, const double max_sensor_range_m,
                          const std::optional<time::RobotTimestamp::duration> &timeout) {
    const ExpectedBeliefRoadMapOptions options = {
        .num_configuration_samples = config.num_configuration_samples(),
        .seed = 12345,
        .timeout = timeout,
        .brm_options =
            {
                .max_sensor_range_m = max_sensor_range_m,
                .uncertainty_tolerance = std::nullopt,
                .max_num_edge_transforms = std::numeric_limits<int>::max(),
                .timeout = timeout,
                .uncertainty_size_options = from_proto(config.uncertainty_size()),
            },
    };

    const time::RobotTimestamp start_time = time::current_robot_time();
    const auto maybe_plan =
        compute_expected_belief_road_map_plan(road_map, ekf, beacon_potential, options);
    const time::RobotTimestamp::duration elapsed_time = time::current_robot_time() - start_time;

    return {
        .elapsed_time = elapsed_time,
        .plan = maybe_plan.has_value()
                    ? std::make_optional(PlannerResult::Plan{
                          .nodes = maybe_plan->nodes,
                          .log_prob_mass_tracked = maybe_plan->log_probability_mass_tracked,
                      })
                    : std::nullopt,
    };
}

PlannerResult run_planner(const planning::RoadMap &road_map, const EkfSlam &ekf,
                          const BeaconPotential &beacon_potential,
                          const proto::OptimisticBRMPlanner &config,
                          const double max_sensor_range_m,
                          const std::optional<time::RobotTimestamp::duration> &timeout) {
    const BeliefRoadMapOptions options = BeliefRoadMapOptions{
        .max_sensor_range_m = max_sensor_range_m,
        .uncertainty_tolerance = std::nullopt,
        .max_num_edge_transforms = std::numeric_limits<int>::max(),
        .timeout = timeout,
        .uncertainty_size_options = from_proto(config.uncertainty_size()),
    };
    const time::RobotTimestamp start_time = time::current_robot_time();
    const auto maybe_plan = compute_belief_road_map_plan(road_map, ekf, {}, options);
    const time::RobotTimestamp::duration elapsed_time = time::current_robot_time() - start_time;

    const double log_prob = beacon_potential.log_prob(beacon_potential.members());

    return {.elapsed_time = elapsed_time,
            .plan = maybe_plan.has_value() ? std::make_optional(PlannerResult::Plan{
                                                 .nodes = maybe_plan->nodes,
                                                 .log_prob_mass_tracked = log_prob,
                                             })
                                           : std::nullopt};
}

struct EvaluationResult {
    struct PlanStatistics {
        struct Plan {
            std::vector<int> nodes;
            double log_prob_mass_tracked;
            double expected_size;
        };
        time::RobotTimestamp::duration elapsed_time;
        std::optional<Plan> plan;
    };
    std::unordered_map<std::string, PlanStatistics> results;
    StartGoal start_goal;
};

template <typename UncertaintySize>
EvaluationResult evaluate_results(const StartGoal &start_goal, const WorldMap &world_map,
                                  const planning::RoadMap &road_map, const EkfSlam &ekf,
                                  const std::unordered_map<std::string, PlannerResult> &results,
                                  const int evaluation_seed, const int num_eval_trials,
                                  const double max_sensor_range_m,
                                  const UncertaintySize &uncertainty_size) {
    using Plan = EvaluationResult::PlanStatistics::Plan;
    std::mt19937 gen(evaluation_seed);

    std::unordered_map<std::string, EvaluationResult::PlanStatistics> results_map;
    for (int i = 0; i < num_eval_trials; i++) {
        const auto present_beacons = world_map.beacon_potential().sample(make_in_out(gen));
        const auto belief_updater = make_belief_updater(road_map, max_sensor_range_m, ekf,
                                                        present_beacons, TransformType::COVARIANCE);
        const RobotBelief initial_belief = {
            .local_from_robot = ekf.estimate().local_from_robot(),
            .cov_in_robot = ekf.estimate().robot_cov(),
        };

        for (const auto &[name, result] : results) {
            if (results_map.find(name) == results_map.end()) {
                results_map[name] = {
                    .elapsed_time = result.elapsed_time,
                    .plan = result.plan.has_value()
                                ? std::make_optional(Plan{
                                      .nodes = result.plan->nodes,
                                      .log_prob_mass_tracked = result.plan->log_prob_mass_tracked,
                                      .expected_size = 0})
                                : std::nullopt};
            }

            if (result.plan.has_value()) {
                RobotBelief belief = initial_belief;
                const auto &plan = result.plan->nodes;
                for (int i = 1; i < static_cast<int>(plan.size()); i++) {
                    belief = belief_updater(belief, plan.at(i - 1), plan.at(i));
                }

                results_map[name].plan->expected_size += uncertainty_size(belief) / num_eval_trials;
            }
        }
    }

    return {
        .results = results_map,
        .start_goal = start_goal,
    };
}

proto::ExperimentResult to_proto(const ExperimentConfig &config,
                                 const std::vector<EvaluationResult> &in) {
    proto::ExperimentResult out;
    out.mutable_experiment_config()->CopyFrom(config);

    std::vector<std::string> names;
    for (const auto &planner_config : config.planner_configs()) {
        out.add_planner_names(planner_config.name());
        names.push_back(planner_config.name());
    }

    for (int trial_idx = 0; trial_idx < static_cast<int>(in.size()); trial_idx++) {
        std::cout << "working on trial idx: " << trial_idx << std::endl;
        const auto &proto_start_goal = out.add_start_goal();
        const auto &result = in.at(trial_idx);
        pack_into(result.start_goal.start, proto_start_goal->mutable_start());
        pack_into(result.start_goal.goal, proto_start_goal->mutable_goal());

        for (int planner_idx = 0; planner_idx < static_cast<int>(names.size()); planner_idx++) {
            std::cout << "Working on planner: " << names.at(planner_idx) << std::endl;
            ROBOT_CHECK(result.results.contains(names.at(planner_idx)), "uh oh!", result.results);

            const auto &trial_result = result.results.at(names.at(planner_idx));
            proto::PlannerResult &planner_result = *out.add_results();
            planner_result.set_trial_id(trial_idx);
            planner_result.set_planner_id(planner_idx);
            pack_into(trial_result.elapsed_time, planner_result.mutable_elapsed_time());

            if (trial_result.plan.has_value()) {
                proto::Plan &plan = *planner_result.mutable_plan();
                plan.mutable_nodes()->Add(trial_result.plan->nodes.begin(),
                                          trial_result.plan->nodes.end());
                plan.set_log_prob_mass(trial_result.plan->log_prob_mass_tracked);
                plan.set_expected_size(trial_result.plan->expected_size);
            }
        }
    }
    return out;
}

}  // namespace

void run_experiment(const proto::ExperimentConfig &config, const std::filesystem::path &base_path,
                    const std::filesystem::path &output_path, const int num_threads) {
    std::cout << config.DebugString() << std::endl;
    if (std::filesystem::exists(output_path)) {
        std::cout << "Output file already exists. Bailing early" << std::endl;
        std::exit(0);
    }

    // Load Map Config
    const auto maybe_map_config =
        robot::proto::load_from_file<proto::WorldMapConfig>(base_path / config.map_config_path());
    ROBOT_CHECK(maybe_map_config.has_value());
    const WorldMap world_map(unpack_from(maybe_map_config.value()));

    // Load EKF State
    const auto maybe_mapped_landmarks =
        robot::proto::load_from_file<proto::MappedLandmarks>(base_path / config.ekf_state_path());
    ROBOT_CHECK(maybe_mapped_landmarks.has_value());
    const auto mapped_landmarks = unpack_from(maybe_mapped_landmarks.value());

    // Create a road map
    const auto maybe_road_map =
        robot::proto::load_from_file<planning::proto::RoadMap>(base_path / config.road_map_path());
    ROBOT_CHECK(maybe_road_map.has_value());
    const planning::RoadMap road_map = unpack_from(maybe_road_map.value());

    std::mt19937 gen(config.start_goal_seed());
    const std::vector<std::tuple<int, StartGoal>> start_goals =
        sample_start_goal(road_map, config.num_trials(), make_in_out(gen));

    EkfSlam ekf = load_ekf_slam(mapped_landmarks);

    std::atomic<int> remaining = start_goals.size();
    const std::optional<time::RobotTimestamp::duration> timeout =
        config.has_plan_timeout_s() ? std::make_optional(time::as_duration(config.plan_timeout_s()))
                                    : std::nullopt;

    std::vector<EvaluationResult> all_statistics(start_goals.size());
    BS::thread_pool pool(num_threads);

    const auto uncertainty_size =
        make_uncertainty_size<RobotBelief>(from_proto(config.uncertainty_metric()));

    for (const auto &start_goal : start_goals) {
        pool.detach_task([=, &remaining, eval_base_seed = config.evaluation_base_seed(),
                          num_eval_trials = config.num_eval_trials(),
                          max_sensor_range_m = config.max_sensor_range_m(), &all_statistics,
                          elem = start_goal, road_map = road_map]() mutable {
            const auto &[idx, start_goal] = elem;
            // Add the start goal to the road map
            road_map.add_start_goal(
                {.start = start_goal.start,
                 .goal = start_goal.goal,
                 .connection_radius_m = config.start_goal_connection_radius_m()});

            // Set the initial pose
            const liegroups::SE2 local_from_robot = liegroups::SE2::trans(start_goal.start);
            ekf.estimate().local_from_robot(local_from_robot);

            std::unordered_map<std::string, PlannerResult> results;

            for (const auto &planner_config : config.planner_configs()) {
                ROBOT_CHECK(planner_config.planner_config_oneof_case() !=
                            proto::PlannerConfig::PLANNER_CONFIG_ONEOF_NOT_SET);
                switch (planner_config.planner_config_oneof_case()) {
                    case proto::PlannerConfig::kLandmarkBrmConfig: {
                        results[planner_config.name()] = run_planner(
                            road_map, ekf, world_map.beacon_potential(),
                            planner_config.landmark_brm_config(), max_sensor_range_m, timeout);
                        break;
                    }
                    case proto::PlannerConfig::kOptimisticBrmConfig: {
                        results[planner_config.name()] = run_planner(
                            road_map, ekf, world_map.beacon_potential(),
                            planner_config.optimistic_brm_config(), max_sensor_range_m, timeout);
                        break;
                    }
                    case proto::PlannerConfig::kExpectedBrmConfig: {
                        results[planner_config.name()] = run_planner(
                            road_map, ekf, world_map.beacon_potential(),
                            planner_config.expected_brm_config(), max_sensor_range_m, timeout);
                        break;
                    }
                    default: {
                        ROBOT_CHECK(false, "Unhandled Planner Config type",
                                    planner_config.planner_config_oneof_case());
                    }
                }
            }

            all_statistics.at(idx) = evaluate_results(start_goal, world_map, road_map, ekf, results,
                                                      eval_base_seed + idx, num_eval_trials,
                                                      max_sensor_range_m, uncertainty_size);
            std::cout << "remaining: " << --remaining << std::endl;
        });
    }
    pool.wait();

    // Write out the results
    const auto experiment_results_proto = to_proto(config, all_statistics);

    std::filesystem::create_directories(output_path.parent_path());
    std::ofstream file_out(output_path, std::ios::binary | std::ios::out);
    experiment_results_proto.SerializeToOstream(&file_out);
}
}  // namespace robot::experimental::beacon_sim

int main(int argc, const char **argv) {
    // clang-format off
    const std::string DEFAULT_NUM_THREADS = "0";
    cxxopts::Options options("run_experiment", "Run experiments for paper");
    options.add_options()
        ("config_file", "Path to experiment config file", cxxopts::value<std::string>())
        ("output_path", "Output directory for experiment results", cxxopts::value<std::string>())
        ("num_threads", "number of experiments to run in parallel",
            cxxopts::value<int>()->default_value(DEFAULT_NUM_THREADS))
        ("help", "Print usage");
    // clang-format on

    auto args = options.parse(argc, argv);

    const auto check_required = [&](const std::string &opt) {
        if (args.count(opt) == 0) {
            std::cout << "Missing " << opt << " argument" << std::endl;
            std::cout << options.help() << std::endl;
            std::exit(1);
        }
    };

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    check_required("config_file");
    check_required("output_path");

    std::filesystem::path config_file_path = args["config_file"].as<std::string>();
    const auto maybe_config_file = robot::proto::load_from_file<ExperimentConfig>(config_file_path);
    ROBOT_CHECK(maybe_config_file.has_value());

    robot::experimental::beacon_sim::run_experiment(
        maybe_config_file.value(), config_file_path.remove_filename(),
        args["output_path"].as<std::string>(), args["num_threads"].as<int>());
}
