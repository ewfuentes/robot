
#include <filesystem>
#include <fstream>
#include <optional>

#include "google/protobuf/text_format.h"

namespace robot::proto {

template <typename MsgType>
std::optional<MsgType> load_from_file(const std::filesystem::path &path) {
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }

    std::ifstream file_in(path, std::ios::binary | std::ios::in);
    std::stringstream sstream;
    sstream << file_in.rdbuf();

    MsgType out;

    if (out.ParseFromString(sstream.str())) {
        return out;
    }

    if (google::protobuf::TextFormat::ParseFromString(sstream.str(), &out)) {
        return out;
    }

    return std::nullopt;
}

}  // namespace robot::proto
