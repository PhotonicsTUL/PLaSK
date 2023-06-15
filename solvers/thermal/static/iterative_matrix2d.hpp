/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__MODULE_THERMAL_ITERATIVE_MATRIX2D_H
#define PLASK__MODULE_THERMAL_ITERATIVE_MATRIX2D_H

#include <algorithm>
#include <plask/plask.hpp>

#include "conjugate_gradient.hpp"


namespace plask { namespace thermal { namespace tstatic {

struct SparseBandMatrix2D {
    static constexpr size_t nd = 5;
    static constexpr size_t kd = 4;
    static constexpr size_t ld = 4;

    const int size;         ///< Order of the matrix, i.e. number of columns or rows
    int bno[nd];            ///< Vector of non-zero band numbers (shift from diagonal)

    double* data;           ///< Data stored in the matrix

    NspcgSolver<SparseBandMatrix2D> matrix_solver;

    /**
     * Create matrix.
     * \param size size of the matrix
     * \param major shift of nodes to the next row (mesh[x,y+1])
     */
    SparseBandMatrix2D(size_t size, size_t major): size(size) {
        bno[0] = 0;  bno[1] = 1;  bno[2] = major - 1;  bno[3] = major;  bno[4] = major + 1;
        data = aligned_malloc<double>(nd*size);
        clear();
    }

    SparseBandMatrix2D(const SparseBandMatrix2D&) = delete;

    SparseBandMatrix2D(SparseBandMatrix2D&& src): size(src.size), data(src.data), matrix_solver(std::move(src.matrix_solver)) {
        std::copy_n(src.bno, nd, bno);
        src.data = nullptr;
    }

    ~SparseBandMatrix2D() {
        aligned_free<double>(data);
    }

    /**
     * Return reference to array element.
     * \param r index of the element row
     * \param c index of the element column
     * \return reference to array element
     **/
    double& operator()(size_t r, size_t c) {
        if (r < c) std::swap(r, c);
        size_t i = std::find(bno, bno+nd, r-c) - bno;
        assert(i != nd);
        return data[c+size*i];
    }

    /// Clear the matrix
    void clear() {
        std::fill_n(data, nd*size, 0.);
    }
};

}}} // namespace plask::thermal::tstatic

#endif // PLASK__MODULE_THERMAL_ITERATIVE_MATRIX2D_H
