
#pragma once

#include <functional>

namespace robot::domain {

auto make_private_info(const auto owner) {
    return [owner](const auto player) { return player == owner; };
}

template <typename T, typename PlayerT>
class Fog {
   public:
    using VisibilityFunc = std::function<bool(const PlayerT &)>;

    Fog() : has_value_{false}, is_visible_{}, value_{} {}

    Fog(const T &action, const VisibilityFunc &f)
        : has_value_{true}, is_visible_{f}, value_(action) {}

    bool is_visible_to(const PlayerT &player) const { return is_visible_(player); }
    void update_visibility(const VisibilityFunc &f) { is_visible_ = f; }

    bool has_value() const { return has_value_; }

    T &value() { return value_; }
    const T &value() const { return value_; }

   private:
    bool has_value_;
    VisibilityFunc is_visible_;
    T value_;
};
}  // namespace robot::domain
