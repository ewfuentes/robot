
#pragma once

#include <algorithm>
#include <array>
#include <vector>

#include "common/check.hh"

namespace robot::math {

// Interpolate between data points using piecewise cubic polynomials
template <typename T = double, typename X = double>
class CubicHermiteSpline {
   public:
    CubicHermiteSpline(std::vector<T> ts, std::vector<X> xs)
        : ts_(std::move(ts)), xs_(std::move(xs)) {
        CHECK(std::is_sorted(ts_.begin(), ts_.end()), "Input times must be sorted!", ts_);
    }

    X operator()(const T &query_time) const {
        CHECK(query_time >= ts_.front() && query_time <= ts_.back(), "query_time is out of bounds",
              ts_.front(), ts_.back(), query_time);

        const auto iter = std::lower_bound(ts_.begin(), ts_.end(), query_time);
        if (iter == ts_.end()) {
            // Since we know that query time is less than or equal to the max time and
            // there is no element t \in ts_ for which query_time < t evaluates to true,
            // then query_time must equal ts_.back(). Therefore, query_time == ts_.back()
            return xs_.back();
        } else if (iter == ts_.begin()) {
            // Similarly, since we know that query time is greater than or equal to the min time and
            // there is no element t \in ts_ for which query_time < t evaluates to false,
            // then query_time must equal ts_.front(). Therefore, query_time == ts_.front()
            return xs_.front();
        }

        // A cubic polynomial has 4 degrees of freedom. To constrain the cubic polynomial
        // between t_i and t_{i+1}, we require that f(t_i) = x_i, f(t_{i+1}) = x_{i+1} and
        // that f'(t_j) = 1/2 ((x_{j+1} - x_j)/(t_{j+1} - t_j) + (x_j - x_{j-1})/(t_j - t_{j-1}))
        // at t_i and t_{i+1}.

        const T &end_t = *iter;
        const T &start_t = *(iter - 1);
        const T segment_length = end_t - start_t;
        const X &end_val = xs_.at(std::distance(ts_.begin(), iter));
        const X &start_val = xs_.at(std::distance(ts_.begin(), iter - 1));

        const X segment_slope = (end_val - start_val) / segment_length;

        const X start_slope = [&]() {
            if (iter - 1 == ts_.begin()) {
                return segment_slope;
            }

            const T &pre_t = *(iter - 2);
            const X &pre_val = xs_.at(std::distance(ts_.begin(), iter - 2));
            const X pre_slope = (start_val - pre_val) / (start_t - pre_t);

            return 0.5 * (pre_slope + segment_slope);
        }();

        const X end_slope = [&]() {
            if (iter + 1 == ts_.end()) {
                return segment_slope;
            }

            const T &post_t = *(iter + 1);
            const X &post_val = xs_.at(std::distance(ts_.begin(), iter + 1));
            const X post_slope = (post_val - end_val) / (post_t - end_t);

            return 0.5 * (post_slope + segment_slope);
        }();

        const double segment_t = (query_time - start_t) / (end_t - start_t);
        const double segment_t2 = segment_t * segment_t;
        const double segment_t3 = segment_t2 * segment_t;

        std::array<double, 4> coeffs{
            2 * segment_t3 - 3 * segment_t2 + 1,
            segment_t3 - 2 * segment_t2 + segment_t,
            -2 * segment_t3 + 3 * segment_t2,
            segment_t3 - segment_t2,
        };

        return coeffs[0] * start_val + coeffs[1] * segment_length * start_slope +
               coeffs[2] * end_val + coeffs[3] * segment_length * end_slope;
    }

    const std::vector<T> &ts() const { return ts_; }
    const std::vector<X> &xs() const { return xs_; }

   private:
    std::vector<T> ts_;
    std::vector<X> xs_;
};
}  // namespace robot::math
