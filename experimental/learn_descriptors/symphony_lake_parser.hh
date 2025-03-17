#pragma once
#include <coroutine>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "symphony_lake_dataset/ImagePoint.h"
#include "symphony_lake_dataset/SurveyVector.h"

namespace robot::experimental::learn_descriptors {
class SymphonyLakeCamParams {
   public:
    constexpr static double fx = 759.308, fy = 690.44;
    constexpr static double px = 370.915, py = 250.909;

    constexpr static double k1 = -0.302805, k2 = 0.171088, k3 = 0.0;
    constexpr static double p1 = 0.001151, p2 = -0.00038;
};
class DataParser {
   public:
    template <typename T>
    struct Generator {
        struct promise_type {
            T value;
            std::suspend_always yield_value(T v) {
                value = v;
                return {};
            }
            std::suspend_always initial_suspend() { return {}; }
            std::suspend_always final_suspend() noexcept { return {}; }
            Generator get_return_object() { return Generator{handle_type::from_promise(*this)}; }
            void return_void() {}
            void unhandled_exception() { std::terminate(); }
        };

        using handle_type = std::coroutine_handle<promise_type>;

        Generator(handle_type h) : handle(h) {}
        ~Generator() {
            if (handle) handle.destroy();
        }

        struct iterator {
            handle_type handle;
            bool operator!=(std::default_sentinel_t) const { return !handle.done(); }
            iterator &operator++() {
                handle.resume();
                return *this;
            }
            T operator*() const { return handle.promise().value; }
        };

        iterator begin() { return iterator{handle}; }
        std::default_sentinel_t end() { return {}; }

       private:
        handle_type handle;
    };

    Generator<cv::Mat> image_generator(const symphony_lake_dataset::SurveyVector &survey_vector) {
        for (int i = 0; i < static_cast<int>(survey_vector.getNumSurveys()); i++) {
            const symphony_lake_dataset::Survey &survey = survey_vector.get(i);
            for (int j = 0; j < static_cast<int>(survey.getNumImages()); j++) {
                co_yield survey.loadImageByImageIndex(j);
            }
        }
    }

    static const Eigen::Vector3d t_boat_cam;
    static const Eigen::Isometry3d T_boat_gps;    
    static const Eigen::Isometry3d T_boat_imu;

    DataParser(const std::filesystem::path &image_root_dir,
               const std::vector<std::string> &survey_list);
    ~DataParser();

    Eigen::Affine3d get_T_world_camera(size_t survey_idx, size_t image_idx, bool use_gps = false,
                                       bool use_compass = false);
    static const Eigen::Isometry3d get_T_boat_camera(const symphony_lake_dataset::ImagePoint &img_pt);
    static const Eigen::Isometry3d get_T_boat_camera(double theta_pan, double theta_tilt);
    /// @brief get_R_world_boat assuming z_axis_boat dot z_axis_world ~ -1
    /// @param theta_compass in radians
    /// @return R_world_boat
    static const Eigen::Matrix3d get_R_world_boat(double theta_compass);
    /// @brief get_T_world_boat assuming z_axis_boat dot z_axis_world ~ -1
    /// @param img_pt 
    /// @return T_world_boat
    static const Eigen::Isometry3d get_T_world_boat(const symphony_lake_dataset::ImagePoint &img_pt);

    const symphony_lake_dataset::SurveyVector &get_surveys() const { return surveys_; };
    Generator<cv::Mat> create_img_generator() { return image_generator(surveys_); };

   private:
    std::filesystem::path image_root_dir_;
    std::vector<std::string> survey_list_;
    symphony_lake_dataset::SurveyVector surveys_;
};
class SymphonyLakeDatasetTestHelper {
   public:
    static constexpr const char *test_image_root_dir =
        "external/symphony_lake_snippet/symphony_lake";
    static constexpr std::array<const char *, 1> test_survey_list = {"140106_snippet"};
    static bool images_equal(cv::Mat img1, cv::Mat img2) {
        if (img1.size() != img2.size() || img1.type() != img2.type()) {
            return false;
        }
        cv::Mat diff;
        cv::absdiff(img1, img2, diff);
        diff = diff.reshape(1);
        return cv::countNonZero(diff) == 0;
    };
    static DataParser get_test_parser() {
        return DataParser(get_test_iamge_root_dir(), get_test_survey_list());
    };
    static std::filesystem::path get_test_iamge_root_dir() {
        return std::filesystem::path(test_image_root_dir);
    };
    static std::vector<std::string> get_test_survey_list() {
        return std::vector<std::string>(test_survey_list.begin(), test_survey_list.end());
    };
};
}  // namespace robot::experimental::learn_descriptors