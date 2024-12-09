
#include <fmt/core.h>

#include <filesystem>
#include <limits>

#include "fmt/format.h"
#include "gtest/gtest.h"
#include "osmium/handler.hpp"
#include "osmium/handler/dump.hpp"
#include "osmium/io/pbf_input.hpp"
#include "osmium/visitor.hpp"

namespace robot::openstreetmap {
TEST(OpenstreetmapTest, can_open_pbf_file) {
    // Setup
    struct TestHandler : public osmium::handler::Handler {
        int node_counter = 0;
        int way_counter = 0;
        int relation_counter = 0;

        void node(const osmium::Node &node) { node_counter++; }

        void way(const osmium::Way &way) { way_counter++; }

        void relation(const osmium::Relation &relation) { relation_counter++; }
    };
    const std::filesystem::path osm_pbf_path(
        "external/openstreetmap_snippet/us-virgin-islands-latest.osm.pbf");
    osmium::io::Reader reader(osm_pbf_path, osmium::osm_entity_bits::node |
                                                osmium::osm_entity_bits::way |
                                                osmium::osm_entity_bits::relation);

    // Action
    auto handler = TestHandler();
    osmium::apply(reader, handler);

    // Verification
    EXPECT_GT(handler.node_counter, 0);
    EXPECT_GT(handler.way_counter, 0);
    EXPECT_GT(handler.relation_counter, 0);
}

}  // namespace robot::openstreetmap
