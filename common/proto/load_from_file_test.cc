

#include "common/proto/load_from_file.hh"

#include <cstdlib>

#include "common/proto/sample_message.pb.h"
#include "gtest/gtest.h"

namespace robot::proto {
namespace {
std::filesystem::path get_tmp_dir() {
    const char* maybe_tmp_dir = std::getenv("TEST_TMPDIR");
    if (maybe_tmp_dir) {
        return maybe_tmp_dir;
    }
    return "/tmp";
}

SampleMessage get_sample_message() {
    SampleMessage out;
    out.set_test_int(123);
    out.set_test_string("my_string");
    out.mutable_test_nested()->set_test_bool(true);
    return out;
}
}  // namespace

TEST(LoadFromFileTest, load_text_proto) {
    // Setup
    const std::filesystem::path text_proto_path = get_tmp_dir() / "test_proto.pbtxt";
    const auto sample_msg = get_sample_message();
    std::string text_proto;
    google::protobuf::TextFormat::PrintToString(sample_msg, &text_proto);
    {
        std::ofstream file_out(text_proto_path);
        file_out << text_proto;
    }

    // Action
    const std::optional<SampleMessage> maybe_msg = load_from_file<SampleMessage>(text_proto_path);

    // Verification
    EXPECT_TRUE(maybe_msg.has_value());
    const auto& msg = maybe_msg.value();
    EXPECT_EQ(msg.test_int(), 123);
    EXPECT_EQ(msg.test_string(), "my_string");
    EXPECT_TRUE(msg.has_test_nested());
    EXPECT_TRUE(msg.test_nested().test_bool());
}

TEST(LoadFromFileTest, load_binary_proto) {
    // Setup
    const std::filesystem::path binary_proto_path = get_tmp_dir() / "test_proto.pb";
    const auto sample_msg = get_sample_message();
    {
        std::ofstream file_out(binary_proto_path);
        sample_msg.SerializeToOstream(&file_out);
    }

    // Action
    const std::optional<SampleMessage> maybe_msg = load_from_file<SampleMessage>(binary_proto_path);

    // Verification
    EXPECT_TRUE(maybe_msg.has_value());
    const auto& msg = maybe_msg.value();
    EXPECT_EQ(msg.test_int(), 123);
    EXPECT_EQ(msg.test_string(), "my_string");
    EXPECT_TRUE(msg.has_test_nested());
    EXPECT_TRUE(msg.test_nested().test_bool());
}

TEST(LoadFromFileTest, not_existing_path_returns_nullopt) {
    // Setup
    const std::filesystem::path fake_proto_path = get_tmp_dir() / "fake_proto.pb";

    // Action
    const std::optional<SampleMessage> maybe_msg = load_from_file<SampleMessage>(fake_proto_path);

    // Verification
    EXPECT_FALSE(maybe_msg.has_value());
}

TEST(LoadFromFileTest, non_text_proto_returns_nullopt) {
    // Setup
    const std::filesystem::path not_proto_path = get_tmp_dir() / "not_proto.pbtxt";
    {
        std::ofstream file_out(not_proto_path);
        file_out << "Hello World!";
    }

    // Action
    const std::optional<SampleMessage> maybe_msg = load_from_file<SampleMessage>(not_proto_path);

    // Verification
    EXPECT_FALSE(maybe_msg.has_value());
}
}  // namespace robot::proto
