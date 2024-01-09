
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"
#include "common/check.hh"

namespace robot::experimental::beacon_sim {
class BeaconPotential;
namespace proto {
class BeaconPotential;
beacon_sim::BeaconPotential unpack_from(const BeaconPotential &);
BeaconPotential pack_into(const beacon_sim::BeaconPotential &);
}  // namespace proto

struct LogMarginal {
    std::vector<int> present_beacons;
    double log_marginal;
};

// A probability distribution over beacon presences/absences
class BeaconPotential {
   public:
    BeaconPotential() = default;
    BeaconPotential(const BeaconPotential &other)
        : impl_(other.impl_ ? other.impl_->clone_() : nullptr) {}
    template <typename T>
    BeaconPotential(T other) : impl_(std::make_unique<Model<T>>(other)) {}

    double log_prob(const std::unordered_map<int, bool> &assignments,
                    const bool allow_partial_assignment = false) const {
        CHECK(impl_ != nullptr);
        return impl_->log_prob_(assignments, allow_partial_assignment);
    };
    double log_prob(const std::vector<int> &present_beacons) const {
        CHECK(impl_ != nullptr);
        return impl_->log_prob_(present_beacons);
    }

    BeaconPotential operator*(const BeaconPotential &other) const;

    std::vector<LogMarginal> log_marginals(const std::vector<int> &remaining) const {
        CHECK(impl_ != nullptr);
        return impl_->log_marginals_(remaining);
    };

    const std::vector<int> members() const {
        CHECK(impl_ != nullptr);
        return impl_->members_();
    };

    friend BeaconPotential proto::unpack_from(const proto::BeaconPotential &);
    friend proto::BeaconPotential proto::pack_into(const BeaconPotential &);

   private:
    struct Concept {
        virtual ~Concept() = default;
        virtual std::unique_ptr<Concept> clone_() const = 0;
        virtual double log_prob_(const std::unordered_map<int, bool> &assignments,
                                 const bool allow_partial_assignment = false) const = 0;
        virtual double log_prob_(const std::vector<int> &present_beacons) const = 0;
        virtual std::vector<int> members_() const = 0;
        virtual std::vector<LogMarginal> log_marginals_(
            const std::vector<int> &remaining) const = 0;
    };

    template <typename T>
    struct Model final : Concept {
        Model(T arg) : data_(arg) {}
        std::unique_ptr<Concept> clone_() const override { return std::make_unique<Model>(data_); }
        double log_prob_(const std::unordered_map<int, bool> &assignments,
                         const bool allow_partial_assignment = false) const override {
            return compute_log_prob(data_, assignments, allow_partial_assignment);
        }

        double log_prob_(const std::vector<int> &present_beacons) const override {
            return compute_log_prob(data_, present_beacons);
        }

        std::vector<LogMarginal> log_marginals_(const std::vector<int> &remaining) const override {
            return compute_log_marginals(data_, remaining);
        }

        std::vector<int> members_() const override { return get_members(data_); }

        T data_;
    };

    std::unique_ptr<Concept> impl_;
};

}  // namespace robot::experimental::beacon_sim
