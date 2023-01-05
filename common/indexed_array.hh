
#pragma once

#include <iterator>
#include <type_traits>
#include <unordered_map>

#include "wise_enum.h"

namespace robot {

namespace detail {
template <typename EnumT>
constexpr bool is_contiguous_and_zero_indexed() {
    static_assert(wise_enum::is_wise_enum_v<EnumT>, "Indexing Type must be a wise_enum");
    int expected_idx = 0;
    for (const auto value_and_name : wise_enum::range<EnumT>) {
        if (static_cast<std::underlying_type_t<EnumT>>(value_and_name.value) != expected_idx) {
            return false;
        }
        expected_idx++;
    }
    return true;
};
}  // namespace detail

template <typename T, typename EnumT>
class IndexedArray {
   public:
    using container_type = std::array<T, wise_enum::size<EnumT>>;

    constexpr IndexedArray() = default;
    constexpr IndexedArray(const T& value) {
        for (const auto& [idx, _] : wise_enum::range<EnumT>) {
            (*this)[idx] = value;
        }
    };
    constexpr IndexedArray(const std::initializer_list<std::pair<EnumT, T>>& init) {
        for (const auto& [idx, value] : init) {
            (*this)[idx] = value;
        }
    };

    constexpr const T& operator[](const EnumT& index) const {
        if constexpr (detail::is_contiguous_and_zero_indexed<EnumT>()) {
            return data_[static_cast<int>(index)];
        } else {
            const auto iter = std::find_if(
                wise_enum::range<EnumT>.begin(), wise_enum::range<EnumT>.end(),
                [index](const auto& enum_and_name) { return enum_and_name.value == index; });
            const int idx = std::distance(wise_enum::range<EnumT>.begin(), iter);
            return data_[idx];
        }
    }

    constexpr T& operator[](const EnumT& index) {
        if constexpr (detail::is_contiguous_and_zero_indexed<EnumT>()) {
            return data_[static_cast<int>(index)];
        } else {
            const auto iter = std::find_if(
                wise_enum::range<EnumT>.begin(), wise_enum::range<EnumT>.end(),
                [index](const auto& enum_and_name) { return enum_and_name.value == index; });
            const int idx = std::distance(wise_enum::range<EnumT>.begin(), iter);
            return data_[idx];
        }
    }

    constexpr int size() const { return wise_enum::size<EnumT>; }

    struct ConstIterator {
        using enum_iter = decltype(wise_enum::range<EnumT>.cbegin());
        using data_iter = decltype(std::array<T, wise_enum::size<EnumT>>{}.cbegin());
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<EnumT, const T&>;

        ConstIterator(const enum_iter& enum_iter, const data_iter& data_iter)
            : enum_iter_{enum_iter}, data_iter_{data_iter} {}

        bool operator!=(const ConstIterator& other) { return enum_iter_ != other.enum_iter_; }

        ConstIterator& operator++() {
            enum_iter_++;
            data_iter_++;
            return *this;
        }

        value_type operator*() const { return {enum_iter_->value, *data_iter_}; }

       private:
        enum_iter enum_iter_;
        data_iter data_iter_;
    };

    struct Iterator {
        using enum_iter = decltype(wise_enum::range<EnumT>.begin());
        using data_iter = decltype(std::array<T, wise_enum::size<EnumT>>{}.begin());
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<EnumT, T&>;

        Iterator(const enum_iter& enum_iter, const data_iter& data_iter)
            : enum_iter_(enum_iter), data_iter_(data_iter) {}

        bool operator!=(const Iterator& other) { return enum_iter_ != other.enum_iter_; }

        Iterator& operator++() {
            enum_iter_++;
            data_iter_++;
            return *this;
        }

        value_type operator*() const { return {enum_iter_->value, *data_iter_}; }

       private:
        enum_iter enum_iter_;
        data_iter data_iter_;
    };

    constexpr ConstIterator begin() const {
        return ConstIterator(wise_enum::range<EnumT>.begin(), data_.cbegin());
    };
    constexpr ConstIterator end() const {
        return ConstIterator(wise_enum::range<EnumT>.end(), data_.cend());
    };
    constexpr Iterator begin() { return Iterator(wise_enum::range<EnumT>.begin(), data_.begin()); };
    constexpr Iterator end() { return Iterator(wise_enum::range<EnumT>.end(), data_.end()); };

   private:
    container_type data_;
};
}  // namespace robot
