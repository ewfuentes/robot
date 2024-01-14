
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>

#include "google/protobuf/text_format.h"

namespace robot::proto {

template <typename MsgType>
std::optional<MsgType> load_from_file(const std::filesystem::path &path) {
    std::cout << "Trying to read from: " << path << std::endl;
    if (!std::filesystem::exists(path)) {
        std::cout << "Does not exist!" << std::endl;
        return std::nullopt;
    }

    std::ifstream file_in(path, std::ios::binary | std::ios::in);
    std::stringstream sstream;
    sstream << file_in.rdbuf();

    MsgType out;

    if (out.ParseFromString(sstream.str())) {
        std::cout << "Found binary proto" << std::endl;
        return out;
    }

    if (google::protobuf::TextFormat::ParseFromString(sstream.str(), &out)) {
        std::cout << "Found text proto" << std::endl;
        return out;
    }

    std::cout << "couldn't parse as binary or text proto" << std::endl;
    return std::nullopt;
}

}  // namespace robot::proto
