
#pragma once

#include <functional>

namespace robot::domain {

template <typename T, typename PlayerT>
class Fog {
   public:
    using VisibilityFunc = std::function<bool(const PlayerT &)>;

    Fog() : has_value_{false}, is_visible_{}, value_{} {}

    Fog(const T &action, const VisibilityFunc &f)
        : has_value_{true}, is_visible_{f}, value_(action) {}

    bool is_visible_to(const PlayerT &player) const { return is_visible_(player); }

    bool has_value() const { return has_value_; }

    T &value() { return value_; }
    const T &value() const { return value_; }

   private:
    bool has_value_;
    VisibilityFunc is_visible_;
    T value_;
};
}  // namespace robot::domain
