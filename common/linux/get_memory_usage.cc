
#include <fstream>
#include <regex>
#include <sstream>

#include "common/check.hh"

namespace robot::linux {

int get_memory_usage_kb() {
    std::ifstream in("/proc/self/smaps_rollup");
    std::stringstream stream;
    stream << in.rdbuf();
    std::regex memory_matcher("Rss:\\s+(\\d+)");
    std::smatch match;
    std::string contents = stream.str();
    const bool match_found = std::regex_search(contents, match, memory_matcher);
    CHECK(match_found);
    return std::stoi(match[1]);
}
}  // namespace robot::linux
