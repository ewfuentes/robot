
#pragma once

#include <concepts>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"
#include "common/argument_wrapper.hh"
#include "common/check.hh"
#include "experimental/beacon_sim/log_marginal.hh"

namespace robot::experimental::beacon_sim {
class BeaconPotential;
struct ConditionedPotential;
void recondition_on(ConditionedPotential &pot, const std::unordered_map<int, bool> &assignments);

struct CorrelatedBeaconPotential;
CorrelatedBeaconPotential condition_on(const CorrelatedBeaconPotential &pot,
                                       const std::unordered_map<int, bool> &);
void recondition_on(const CorrelatedBeaconPotential &pot, const std::unordered_map<int, bool> &);

namespace proto {
class BeaconPotential;
beacon_sim::BeaconPotential unpack_from(const BeaconPotential &);
void pack_into(const beacon_sim::BeaconPotential &, BeaconPotential *);
}  // namespace proto

template <typename T>
concept Potential = requires(T pot, bool b, std::unordered_map<int, bool> assignment,
                             std::vector<int> present_beacons, proto::BeaconPotential *bp_proto,
                             InOut<std::mt19937> gen) {
                        { get_members(pot) } -> std::same_as<const std::vector<int> &>;
                        { compute_log_prob(pot, assignment, b) } -> std::same_as<double>;
                        {
                            compute_log_marginals(pot, present_beacons)
                            } -> std::same_as<std::vector<LogMarginal>>;
                        { pack_into_potential(pot, bp_proto) } -> std::same_as<void>;
                        { pack_into_potential(pot, bp_proto) } -> std::same_as<void>;
                        { generate_sample(pot, gen) } -> std::same_as<std::vector<int>>;
                    };

// A probability distribution over beacon presences/absences
class BeaconPotential {
   public:
    BeaconPotential() = default;
    BeaconPotential(const BeaconPotential &other)
        : impl_(other.impl_ ? other.impl_->clone_() : nullptr) {}

    BeaconPotential &operator=(const BeaconPotential &other) {
        if (other.impl_) {
            impl_ = other.impl_->clone_();
        } else {
            impl_.reset();
        }
        return *this;
    }

    template <Potential T>
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

    std::vector<LogMarginal> log_marginals(const std::vector<int> &remaining) const {
        return impl_ ? impl_->log_marginals_(remaining)
                     : std::vector<LogMarginal>{{.present_beacons = {}, .log_marginal = 0.0}};
    };

    std::vector<int> sample(InOut<std::mt19937> gen) const {
        return impl_ ? impl_->sample_(gen) : std::vector<int>{};
    }

    const std::vector<int> &members() const {
        const static std::vector<int> EMPTY_MEMBERS{};
        return impl_ ? impl_->members_() : EMPTY_MEMBERS;
    };

    BeaconPotential conditioned_on(const std::unordered_map<int, bool> &assignments) const {
        return impl_ ? impl_->condition_on_(assignments) : *this;
    };

    void reconditioned_on(const std::unordered_map<int, bool> &assignments) {
        if (impl_) {
            impl_->recondition_on_(assignments);
        }
    };

    friend BeaconPotential proto::unpack_from(const proto::BeaconPotential &);
    friend void proto::pack_into(const BeaconPotential &, proto::BeaconPotential *);

   private:
    struct Concept {
        virtual ~Concept() = default;
        virtual std::unique_ptr<Concept> clone_() const = 0;
        virtual double log_prob_(const std::unordered_map<int, bool> &assignments,
                                 const bool allow_partial_assignment = false) const = 0;
        virtual double log_prob_(const std::vector<int> &present_beacons) const = 0;
        virtual const std::vector<int> &members_() const = 0;
        virtual std::vector<LogMarginal> log_marginals_(
            const std::vector<int> &remaining) const = 0;
        virtual std::vector<int> sample_(InOut<std::mt19937> gen) const = 0;
        virtual void pack_into_(proto::BeaconPotential *out) const = 0;
        virtual BeaconPotential condition_on_(
            const std::unordered_map<int, bool> &assignments) const = 0;
        virtual void recondition_on_(const std::unordered_map<int, bool> &assignments) = 0;
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
            std::unordered_map<int, bool> assignment;
            for (const int member_id : members_()) {
                const auto iter =
                    std::find(present_beacons.begin(), present_beacons.end(), member_id);
                assignment[member_id] = iter == present_beacons.end() ? false : true;
            }

            constexpr bool DONT_ALLOW_PARTIAL_ASSIGNMENTS = false;
            return log_prob_(assignment, DONT_ALLOW_PARTIAL_ASSIGNMENTS);
        }

        std::vector<LogMarginal> log_marginals_(const std::vector<int> &remaining) const override {
            return compute_log_marginals(data_, remaining);
        }

        std::vector<int> sample_(InOut<std::mt19937> gen) const override {
            return generate_sample(data_, gen);
        }

        const std::vector<int> &members_() const override { return get_members(data_); }

        void pack_into_(proto::BeaconPotential *out) const override {
            pack_into_potential(data_, out);
        }

        BeaconPotential condition_on_(
            const std::unordered_map<int, bool> &assignments) const override {
            constexpr bool has_conditioning_support =
                requires(T p, std::unordered_map<int, bool> & a) {
                    { condition_on(p, a) } -> std::same_as<T>;
                };
            if constexpr (has_conditioning_support) {
                return BeaconPotential(condition_on(data_, assignments));
            } else {
                return BeaconPotential(ConditionedPotential(data_, assignments));
            }
        }

        void recondition_on_(const std::unordered_map<int, bool> &assignments) override {
            constexpr bool has_reconditioning_support =
                requires(T p, std::unordered_map<int, bool> & a) {
                    { recondition_on(p, a) } -> std::same_as<void>;
                };

            if constexpr (has_reconditioning_support) {
                recondition_on(data_, assignments);
            }
        }

        T data_;
    };

    std::unique_ptr<Concept> impl_;
};

struct CombinedPotential {
    explicit CombinedPotential(std::vector<BeaconPotential> pots);
    std::vector<BeaconPotential> pots;
    std::vector<int> members;
};
double compute_log_prob(const CombinedPotential &pot,
                        const std::unordered_map<int, bool> &assignments,
                        const bool allow_partial_assignments);
std::vector<LogMarginal> compute_log_marginals(const CombinedPotential &pot,
                                               const std::vector<int> &remaining);
const std::vector<int> &get_members(const CombinedPotential &pot);
void pack_into_potential(const CombinedPotential &in, proto::BeaconPotential *out);
std::vector<int> generate_sample(const CombinedPotential &pot, InOut<std::mt19937> gen);
CombinedPotential condition_on(const CombinedPotential &pot,
                               const std::unordered_map<int, bool> &assignments);
void recondition_on(CombinedPotential &pot, const std::unordered_map<int, bool> &assignments);

BeaconPotential operator*(const BeaconPotential &a, const BeaconPotential &b);

// A ConditionedPotential is meant to represent a BeaconPotential where
// certain beacons are assumed to take a given value. A conditioned potential
// maintains the same set of members as the underlying distribution, but
// if compute_log_prob is queried with a conflicting assignment, a log probability
// of -infinity is returned. `compute_log_marginals` will contain the unconditioned
// variables and the conditioned variables with the given assignment.
struct ConditionedPotential {
    ConditionedPotential(BeaconPotential pot, std::unordered_map<int, bool> assignments);
    BeaconPotential underlying_pot;
    std::unordered_map<int, bool> conditioned_members;
    // The probability of the conditioned assignment on `underlying_pot`
    double log_normalizer;
};

double compute_log_prob(const ConditionedPotential &pot,
                        const std::unordered_map<int, bool> &assignments,
                        const bool allow_partial_assignments);

std::vector<LogMarginal> compute_log_marginals(const ConditionedPotential &pot,
                                               const std::vector<int> &remaining);
const std::vector<int> &get_members(const ConditionedPotential &pot);
void pack_into_potential(const ConditionedPotential &in, proto::BeaconPotential *out);
std::vector<int> generate_sample(const ConditionedPotential &pot, InOut<std::mt19937> gen);

}  // namespace robot::experimental::beacon_sim
