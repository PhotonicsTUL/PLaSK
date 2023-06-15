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
#ifndef PLASK__MODULE_THERMAL_ITERATIVE_MATRIX3D_H
#define PLASK__MODULE_THERMAL_ITERATIVE_MATRIX3D_H

#include <algorithm>
#include <plask/plask.hpp>

#include "conjugate_gradient.hpp"


namespace plask { namespace thermal { namespace tstatic {

struct SparseBandMatrix3D {
    static constexpr size_t nd = 14;
    static constexpr size_t kd = 13;
    static constexpr size_t ld = 13;

    const int size;         ///< Order of the matrix, i.e. number of columns or rows
    int bno[nd];            ///< Vector of non-zero band numbers (shift from diagonal)

    double* data;           ///< Data stored in the matrix

    NspcgSolver<SparseBandMatrix3D> matrix_solver;

    /**
     * Create matrix.
     * \param size size of the matrix
     * \param major shift of nodes to the next major row (mesh[x,y,z+1])
     * \param minor shift of nodes to the next minor row (mesh[x,y+1,z])
     */
    SparseBandMatrix3D(size_t size, size_t major, size_t minor): size(size) {
                                      bno[0]  =             0;  bno[1]  =                 1;
        bno[2]  =         minor - 1;  bno[3]  =         minor;  bno[4]  =         minor + 1;
        bno[5]  = major - minor - 1;  bno[6]  = major - minor;  bno[7]  = major - minor + 1;
        bno[8]  = major         - 1;  bno[9]  = major        ;  bno[10] = major         + 1;
        bno[11] = major + minor - 1;  bno[12] = major + minor;  bno[13] = major + minor + 1;

        data = aligned_malloc<double>(nd*size);
    }

    SparseBandMatrix3D(const SparseBandMatrix3D&) = delete;


    SparseBandMatrix3D(SparseBandMatrix3D&& src): size(src.size), data(src.data), matrix_solver(std::move(src.matrix_solver)) {
        std::copy_n(src.bno, nd, bno);
        src.data = nullptr;
    }

    ~SparseBandMatrix3D() {
        aligned_free<double>(data);
    }

    /**
     * Return reference to array element.
     * @param r index of the element row
     * @param c index of the element column
     * @return reference to array element
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

}}} // namespace plask::thermal::tstatict3d

#endif // PLASK__MODULE_THERMAL_ITERATIVE_MATRIX3D_H
