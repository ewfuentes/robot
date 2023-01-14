
#pragma once

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include "wise_enum.h"

namespace robot {
template <typename T, typename T2 = void>
struct IndexSize {};

template <typename T>
struct IndexSize<T, std::enable_if_t<wise_enum::is_wise_enum_v<T>>> {
    static constexpr int value = wise_enum::size<T>;
};

template <typename T, typename T2 = void>
struct Range {};

template <typename T>
struct Range<T, std::enable_if_t<wise_enum::is_wise_enum_v<T>>> {
    static constexpr auto value = wise_enum::range<T>;
};

template <typename T>
concept Indexable = requires {
    IndexSize<T>::value;
};

template <Indexable EnumT>
constexpr bool is_contiguous_and_zero_indexed() {
    if constexpr (wise_enum::is_wise_enum_v<EnumT>) {
        int expected_idx = 0;
        for (const auto value_and_name : Range<EnumT>::value) {
            if (static_cast<std::underlying_type_t<EnumT>>(value_and_name.value) != expected_idx) {
                return false;
            }
            expected_idx++;
        }
        return true;
    } else {
        return false;
    }
};

template <typename T, Indexable EnumT>
class IndexedArray {
   public:
    using container_type = std::array<T, IndexSize<EnumT>::value>;

    constexpr IndexedArray() = default;
    constexpr IndexedArray(const T& value) {
        for (const auto& [idx, _] : Range<EnumT>::value) {
            (*this)[idx] = value;
        }
    };
    constexpr IndexedArray(const std::initializer_list<std::pair<EnumT, T>>& init) {
        for (const auto& [idx, value] : init) {
            (*this)[idx] = value;
        }
    };

    constexpr const T& operator[](const EnumT& index) const {
        if constexpr (is_contiguous_and_zero_indexed<EnumT>()) {
            return data_[static_cast<int>(index)];
        } else {
            const auto iter = std::find_if(Range<EnumT>::value.begin(), Range<EnumT>::value.end(),
                                           [index](const auto& enum_and_name) {
                                               const auto& [value, name] = enum_and_name;
                                               return value == index;
                                           });
            const int idx = std::distance(Range<EnumT>::value.begin(), iter);
            return data_[idx];
        }
    }

    constexpr T& operator[](const EnumT& index) {
        if constexpr (is_contiguous_and_zero_indexed<EnumT>()) {
            return data_[static_cast<int>(index)];
        } else if constexpr (wise_enum::is_wise_enum_v<EnumT>) {
            const auto iter = std::find_if(
                Range<EnumT>::value.begin(), Range<EnumT>::value.end(),
                [index](const auto& enum_and_name) { return enum_and_name.value == index; });
            const int idx = std::distance(Range<EnumT>::value.begin(), iter);
            return data_[idx];
        } else {
            const auto iter = std::find_if(Range<EnumT>::value.begin(), Range<EnumT>::value.end(),
                                           [index](const auto& enum_and_name) {
                                               const auto& [value, name] = enum_and_name;
                                               return value == index;
                                           });
            const int idx = std::distance(Range<EnumT>::value.begin(), iter);
            return data_[idx];
        }
    }

    constexpr int size() const { return IndexSize<EnumT>::value; }

    struct ConstIterator {
        using enum_iter = decltype(Range<EnumT>::value.cbegin());
        using data_iter = decltype(std::array<T, IndexSize<EnumT>::value>{}.cbegin());
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<EnumT, const T&>;
        using reference = std::pair<EnumT, const T&>;
        using pointer = std::pair<EnumT, const T&>*;
        using difference_type = std::ptrdiff_t;

        ConstIterator(const enum_iter& enum_iter, const data_iter& data_iter)
            : enum_iter_{enum_iter}, data_iter_{data_iter} {}

        bool operator!=(const ConstIterator& other) { return enum_iter_ != other.enum_iter_; }

        ConstIterator& operator++() {
            enum_iter_++;
            data_iter_++;
            return *this;
        }

        value_type operator*() const {
            const auto& [value, name] = *enum_iter_;
            return {value, *data_iter_};
        }

       private:
        enum_iter enum_iter_;
        data_iter data_iter_;
    };

    struct Iterator {
        using enum_iter = decltype(Range<EnumT>::value.begin());
        using data_iter = decltype(std::array<T, IndexSize<EnumT>::value>{}.begin());
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<EnumT, T&>;
        using reference = std::pair<EnumT, T&>;
        using pointer = std::pair<EnumT, T&>*;
        using difference_type = std::ptrdiff_t;

        Iterator(const enum_iter& enum_iter, const data_iter& data_iter)
            : enum_iter_(enum_iter), data_iter_(data_iter) {}

        bool operator!=(const Iterator& other) { return enum_iter_ != other.enum_iter_; }

        Iterator& operator++() {
            enum_iter_++;
            data_iter_++;
            return *this;
        }

        value_type operator*() const {
            const auto& [value, name] = *enum_iter_;
            return {value, *data_iter_};
        }

       private:
        enum_iter enum_iter_;
        data_iter data_iter_;
    };

    constexpr ConstIterator begin() const {
        return ConstIterator(Range<EnumT>::value.begin(), data_.cbegin());
    };
    constexpr ConstIterator end() const {
        return ConstIterator(Range<EnumT>::value.end(), data_.cend());
    };
    constexpr Iterator begin() { return Iterator(Range<EnumT>::value.begin(), data_.begin()); };
    constexpr Iterator end() { return Iterator(Range<EnumT>::value.end(), data_.end()); };

   private:
    container_type data_;
};
}  // namespace robot
