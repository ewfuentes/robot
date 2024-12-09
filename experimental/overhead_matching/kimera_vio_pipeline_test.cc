#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <future>
#include <memory>
#include <utility>

#include "experimental/overhead_matching/kimera_spectacular_data_provider.hh"
#include "kimera-vio/dataprovider/DataProviderInterface.h"
#include "kimera-vio/logging/Logger.h"
#include "kimera-vio/pipeline/Pipeline.h"
#include "kimera-vio/pipeline/RgbdImuPipeline.h"
#include "kimera-vio/utils/Statistics.h"
#include "kimera-vio/utils/Timer.h"

DEFINE_string(params_folder_path, "../params/Euroc",
              "Path to the folder containing the yaml files with the VIO parameters.");
DEFINE_string(dataset_path, "/Users/Luca/data/MH_01_easy",
              "Path of dataset (i.e. Euroc, /Users/Luca/data/MH_01_easy).");

int main(int argc, char* argv[]) {
    // Initialize Google's flags library.
    google::ParseCommandLineFlags(&argc, &argv, true);
    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);

    // Parse VIO parameters from gflags.
    VIO::VioParams vio_params(FLAGS_params_folder_path);

    // Build dataset parser.
    VIO::DataProviderInterface::Ptr dataset_parser =
        std::make_unique<robot::experimental::overhead_matching::SpectacularDataProviderInterface>(
            vio_params);

    CHECK(dataset_parser);

    VIO::Pipeline::Ptr vio_pipeline;

    vio_pipeline = std::make_unique<VIO::RgbdImuPipeline>(vio_params);

    // Register callback to shutdown data provider in case VIO pipeline
    // shutsdown.
    vio_pipeline->registerShutdownCallback(
        std::bind(&VIO::DataProviderModule::shutdown, dataset_parser));

    // Register callback to vio pipeline.
    dataset_parser->registerImuSingleCallback(
        std::bind(&VIO::Pipeline::fillSingleImuQueue, vio_pipeline, std::placeholders::_1));
    // We use blocking variants to avoid overgrowing the input queues (use
    // the non-blocking versions with real sensor streams)
    dataset_parser->registerLeftFrameCallback(
        std::bind(&VIO::Pipeline::fillLeftFrameQueue, vio_pipeline, std::placeholders::_1));

    dataset_parser->registerDepthFrameCallback(
        std::bind(&VIO::RgbdImuPipeline::fillDepthFrameQueue, vio_pipeline, std::placeholders::_1));

    // Spin dataset.
    auto tic = VIO::utils::Timer::tic();
    bool is_pipeline_successful = false;
    while (dataset_parser->spin() && vio_pipeline->spin()) {
        continue;
    };
    vio_pipeline->shutdown();
    is_pipeline_successful = true;

    // Output stats.
    auto spin_duration = VIO::utils::Timer::toc(tic);
    LOG(WARNING) << "Spin took: " << spin_duration.count() << " ms.";
    LOG(INFO) << "Pipeline successful? " << (is_pipeline_successful ? "Yes!" : "No!");

    if (is_pipeline_successful) {
        // Log overall time of pipeline run.
        VIO::PipelineLogger logger;
        logger.logPipelineOverallTiming(spin_duration);
    }

    return is_pipeline_successful ? EXIT_SUCCESS : EXIT_FAILURE;
}
