
#include <fmt/core.h>
#include "gtest/gtest.h"

#include "fmt/format.h"

#include "osmium/io/pbf_input.hpp"
#include "osmium/handler.hpp"
#include "osmium/visitor.hpp"

#include <filesystem>
#include <limits>

namespace robot::openstreetmap {

TEST(OpenstreetmapTest, can_open_pbf_file) {
    // Setup
    struct TestHandler : public osmium::handler::Handler {
        int node_counter = 0;
        std::pair<double, double> min{std::numeric_limits<double>::max(),std::numeric_limits<double>::max()};
        std::pair<double, double> max{std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};
        void node(const osmium::Node &node) {
            const auto location = node.location();


            if (location.valid()) {
                min = std::make_pair(std::min(min.first, location.lat()), std::min(min.second, location.lon()));
                max = std::make_pair(std::max(max.first, location.lat()), std::max(max.second, location.lon()));
            }
            if (!node.visible()) {
                return;
            }

            fmt::print("{} id: {} location: ({}, {}) tags: {{", node_counter++, node.id(), location.lat(), location.lon());
            for (const auto &tag: node.tags()) {
                fmt::print("{}: {}, ", tag.key(), tag.value());
            }
            std::cout << "}" << std::endl;
        }
    
        void way(const osmium::Way &way) {

        }
    };
    const std::filesystem::path osm_pbf_path("external/openstreetmap_snippet/us-virgin-islands-latest.osm.pbf");
    osmium::io::Reader reader(osm_pbf_path, osmium::osm_entity_bits::node | osmium::osm_entity_bits::way);

    // Action
    auto handler = TestHandler();
    osmium::apply(reader, handler);

    fmt::print("min: ({}, {}) max: ({}, {})\r\n", handler.min.first, handler.min.second, handler.max.first, handler.max.second);
    
    // Verification
    EXPECT_TRUE(std::filesystem::exists(osm_pbf_path));
}

}

