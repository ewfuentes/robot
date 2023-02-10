
#pragma once

#include "Eigen/Core"
#include "common/math/matrix.pb.h"

namespace robot::math::proto {
// Declarations
template <typename Derived>
void pack_into(const Eigen::MatrixBase<Derived> &in, Matrix *out);

template <typename MatrixType>
MatrixType unpack_from(const Matrix &in);

// Definitions
template <typename Derived>
void pack_into(const Eigen::MatrixBase<Derived> &in, Matrix *out) {
    out->set_num_rows(in.rows());
    out->set_num_cols(in.cols());
    auto &data = *out->mutable_data();
    data.Clear();
    data.Reserve(in.rows() * in.cols());
    for (int i = 0; i < in.rows(); i++) {
        for (int j = 0; j < in.cols(); j++) {
            data.AddAlreadyReserved(in(i, j));
        }
    }
}

template <typename MatrixType>
MatrixType unpack_from(const Matrix &in) {
    MatrixType out;
    constexpr bool IS_ROWS_DYNAMIC = MatrixType::RowsAtCompileTime == Eigen::Dynamic;
    constexpr bool IS_COLS_DYNAMIC = MatrixType::ColsAtCompileTime == Eigen::Dynamic;
    constexpr bool IS_DYNAMIC_VECTOR = IS_ROWS_DYNAMIC != IS_COLS_DYNAMIC;
    constexpr bool IS_DYNAMIC_MATRIX = IS_ROWS_DYNAMIC && IS_COLS_DYNAMIC;
    if constexpr (IS_DYNAMIC_VECTOR) {
        out.resize(std::max(in.num_rows(), in.num_cols()));
    } else if constexpr (IS_DYNAMIC_MATRIX) {
        out.resize(in.num_rows(), in.num_cols());
    }
    // Check number of rows/columns are equal
    for (int i = 0; i < in.num_rows(); i++) {
        for (int j = 0; j < in.num_cols(); j++) {
            out(i, j) = in.data(i * in.num_cols() + j);
        }
    }
    return out;
}
}  // namespace robot::math::proto
