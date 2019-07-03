#ifndef PLASK__SOLVER_SLAB_TEMPMATRIX_H
#define PLASK__SOLVER_SLAB_TEMPMATRIX_H

#include <plask/plask.hpp>

#ifdef OPENMP_FOUND
#   include <omp.h>
#endif

#include "matrices.h"

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

    TempMatrix get(size_t N);

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
    #ifdef OPENMP_FOUND
        int mn;
    #endif

    #ifdef OPENMP_FOUND
        TempMatrix(TempMatrixPool* pool, size_t N): pool(pool) {
            const int nthr = omp_get_max_threads();
            int l;
            for (mn = 0; mn != nthr; ++mn) {
                l = omp_test_nest_lock(pool->tmplx+mn);
                if (l) break;
            }
            assert(mn != nthr);
            if (pool->tmpmx[mn].rows() != N) {
                write_debug("allocating temporary matrix {}", mn);
                pool->tmpmx[mn].reset(N, N);
            }
            write_debug("acquiring temporary matrix {} in thread {} ({})", mn, omp_get_thread_num(), l);
        }
        TempMatrix(TempMatrix&& src): pool(src.pool), mn(src.mn) { src.pool = nullptr; }
        TempMatrix(const TempMatrix& src) = delete;
    #else
        TempMatrix(TempMatrixPool* pool, size_t N): pool(pool) {
            if (pool->tmpmx->rows() != N) {
                write_debug("allocating temporary matrix");
                pool->tmpmx->reset(N, N);
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
        operator cmatrix() { return pool->tmpmx[mn]; }

        dcomplex* data() { return pool->tmpmx[mn].data(); }
    #else
        operator cmatrix() { return *pool->tmpmx; }

        dcomplex* data() { return pool->tmpmx->data(); }
    #endif
};

inline TempMatrix TempMatrixPool::get(size_t N) {
    return TempMatrix(this, N);
}


}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_TEMPMATRIX_H

