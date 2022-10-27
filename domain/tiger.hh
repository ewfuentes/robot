
#include <random>

namespace robot::domain {
struct TigerConfig {
    double consistent_observation_probability;
    double listening_reward;
    double treasure_reward;
    double tiger_reward;
};

class Tiger {
   public:
    enum class Action { OPEN_LEFT, OPEN_RIGHT, LISTEN };
    enum class Observation { INVALID, GROWL_LEFT, GROWL_RIGHT, TREASURE, TIGER };

    struct Result {
        double reward;
        Observation observation;
        bool is_done;
    };

    Tiger(const TigerConfig &config, const std::size_t seed = 0);

    Result step(const Action &action);

    bool is_tiger_left() { return is_tiger_left_; }

   private:
    TigerConfig config_;
    std::mt19937 gen_;
    bool is_done_;
    bool is_tiger_left_;
};

}  // namespace robot::domain
