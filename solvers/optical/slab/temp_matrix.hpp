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
#ifndef PLASK__SOLVER_SLAB_TEMPMATRIX_H
#define PLASK__SOLVER_SLAB_TEMPMATRIX_H

#include <plask/plask.hpp>

#ifdef OPENMP_FOUND
#   include <omp.h>
#endif

#include "matrices.hpp"

namespace plask { namespace optical { namespace slab {

struct TempMatrix;

struct TempMatrixPool {
  private:
    cmatrix* tmpmx;             ///< Matrices for temporary storage

    #ifdef OPENMP_FOUND
        omp_nest_lock_t* tmplx; ///< Locks of allocated temporary matrices
    #endif

    friend struct TempMatrix;

  public:

    TempMatrixPool() {
        #ifdef OPENMP_FOUND
            const int nthr = omp_get_max_threads();
            tmpmx = new cmatrix[nthr];
            tmplx = new omp_nest_lock_t[nthr];
            for (int i = 0; i != nthr; ++i) {
                omp_init_nest_lock(tmplx+i);
            }
        #else
            tmpmx = new cmatrix();
        #endif
    }

    ~TempMatrixPool() {
        #ifdef OPENMP_FOUND
            write_debug("destroying temporary matrices");
            const int nthr = omp_get_max_threads();
            for (int i = 0; i != nthr; ++i) {
                omp_destroy_nest_lock(tmplx+i);
            }
            delete[] tmpmx;
            delete[] tmplx;
        #else
            write_debug("destroying temporary matrix");
            delete tmpmx;
        #endif
    }

    TempMatrix get(size_t rows, size_t cols);

    void reset() {
        #ifdef OPENMP_FOUND
            write_debug("freeing temporary matrices");
            const int nthr = omp_get_max_threads();
            for (int i = 0; i != nthr; ++i) {
                tmpmx[i].reset();
            }
        #else
            write_debug("freeing temporary matrix");
            tmpmx->reset();
        #endif
    }


};


struct TempMatrix {
    TempMatrixPool* pool;
    size_t rows, cols;
    #ifdef OPENMP_FOUND
        int mn;
    #endif

    #ifdef OPENMP_FOUND
        TempMatrix(TempMatrixPool* pool, size_t rows, size_t cols): pool(pool), rows(rows), cols(cols) {
            const int nthr = omp_get_max_threads();
            int l;
            for (mn = 0; mn != nthr; ++mn) {
                l = omp_test_nest_lock(pool->tmplx+mn);
                if (l) break;
            }
            assert(mn != nthr);
            size_t NN = rows * cols;
            if (pool->tmpmx[mn].rows() * pool->tmpmx[mn].cols() < NN) {
                write_debug("allocating temporary matrix {}", mn);
                pool->tmpmx[mn].reset(rows, cols);
            }
            write_debug("acquiring temporary matrix {} in thread {} ({})", mn, omp_get_thread_num(), l);
        }
        TempMatrix(TempMatrix&& src): pool(src.pool), mn(src.mn) { src.pool = nullptr; }
        TempMatrix(const TempMatrix& src) = delete;
    #else
        TempMatrix(TempMatrixPool* pool, size_t rows, size_t cols): pool(pool), rows(rows), cols(cols) {
            if (pool->tmpmx->rows() * pool->tmpmx->cols() < rows * cols) {
                write_debug("allocating temporary matrix");
                pool->tmpmx->reset(rows, cols);
            }
        }
    #endif

    #ifdef OPENMP_FOUND
        ~TempMatrix() {
            if (pool) {
                write_debug("releasing temporary matrix {} in thread {}", mn, omp_get_thread_num());
                omp_unset_nest_lock(pool->tmplx+mn);
            }
        }
    #endif

    #ifdef OPENMP_FOUND
        operator cmatrix() {
            if (pool->tmpmx[mn].rows() == rows && pool->tmpmx[mn].cols() == cols)
                return pool->tmpmx[mn];
            else
                return cmatrix(rows, cols, pool->tmpmx[mn].data());
        }

        dcomplex* data() { return pool->tmpmx[mn].data(); }
    #else
        operator cmatrix() {
            if (pool->tmpmx->rows() == rows && pool->tmpmx->cols() == cols)
                return *pool->tmpmx;
            else
                return cmatrix(rows, cols, pool->tmpmx->data());
        }

        dcomplex* data() { return pool->tmpmx->data(); }
    #endif
};

inline TempMatrix TempMatrixPool::get(size_t rows, size_t cols) {
    return TempMatrix(this, rows, cols);
}


}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_TEMPMATRIX_H
