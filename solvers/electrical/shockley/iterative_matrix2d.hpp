#ifndef PLASK__MODULE_ELECTRICAL_ITERATIVE_MATRIX2D_H
#define PLASK__MODULE_ELECTRICAL_ITERATIVE_MATRIX2D_H

#include <algorithm>
#include <plask/plask.hpp>

namespace plask { namespace electrical { namespace shockley {

#undef LDA
#define LDA 8

struct SparseBandMatrix2D {
    const ptrdiff_t size;   ///< Order of the matrix, i.e. number of columns or rows
    ptrdiff_t bno[5];      ///< Vector of non-zero band numbers (shift from diagonal)

    double* data;           ///< Data stored in the matrix

    static constexpr size_t kd = 4;
    static constexpr size_t ld = LDA-1;

    /**
     * Create matrix.
     * \param size size of the matrix
     * \param major shift of nodes to the next row (mesh[x,y+1])
     */
    SparseBandMatrix2D(size_t size, size_t major): size(size) {
        bno[0] = 0;  bno[1] = 1;  bno[2] = major - 1;  bno[3] = major;  bno[4] = major + 1;
        data = aligned_malloc<double>(LDA*size);
    }

    SparseBandMatrix2D(const SparseBandMatrix2D&) = delete;

    SparseBandMatrix2D(SparseBandMatrix2D&& src): size(src.size), data(src.data) {
        src.data = nullptr;
        std::move(std::begin(src.bno), std::end(src.bno), bno);
    }

    ~SparseBandMatrix2D() {
        if (data) aligned_free<double>(data);
    }

    /**
     * Return reference to array element.
     * \param r index of the element row
     * \param c index of the element column
     * \return reference to array element
     **/
    double& operator()(size_t r, size_t c) {
        if (r < c) std::swap(r, c);
        size_t i = std::find(bno, bno+5, r-c) - bno;
        assert(i != 5);
        return data[LDA*c+i];
    }

    /// Clear the matrix
    void clear() {
        std::fill_n(data, LDA*size, 0.);
    }

    /**
     * Multiply matrix by vector
     * \param vector vector to multiply
     * \param result multiplication result
     */
    void mult(const DataVector<const double>& vector, DataVector<double>& result) {
        multiply(vector.data(), result.data());
    }

    /**
     * Multiplication functor for symmetric banded matrix
     */
    void multiply(const double* x, double* y) const { // y = A x
        #pragma omp parallel for
        for (ptrdiff_t r = 0; r < size; ++r) {
            double* datar = data + LDA*r;
            double v = 0.;
            // below diagonal
            for (ptrdiff_t i = 4; i > 0; --i) {
                ptrdiff_t c = r - bno[i];
                if (c >= 0) v += data[LDA*c+i] * x[c];
            }
            // above diagonal
            for (ptrdiff_t i = 0; i < 5; ++i) {
                ptrdiff_t c = r + bno[i];
                if (c < size) v += datar[i] * x[c];
            }
            y[r] = v;
        }
    }

    inline void noUpdate(double*) {}
};


/**
 * Jacobi preconditioner for symmetric banded matrix (i.e. diagonal scaling)
 */
struct PrecondJacobi2D {

    const SparseBandMatrix2D& matrix;

    DataVector<double> diag;

    PrecondJacobi2D(const SparseBandMatrix2D& A): matrix(A), diag(A.size) {
        for (double *last = matrix.data + A.size*LDA, *m = matrix.data, *d = diag.data(); m < last; m += LDA, ++d)
            *d = 1. / *m;
    }

    void operator()(double* z, double* r) const { // z = inv(M) r
        const double* d = diag.data(), *zend = z + matrix.size-4;
        for (; z < zend; z += 4, r += 4, d += 4) {
            z[0] = r[0] * d[0];
            z[1] = r[1] * d[1];
            z[2] = r[2] * d[2];
            z[3] = r[3] * d[3];
        }
        for (zend += 4; z != zend; ++z, ++r, ++d)
            *z = *r * *d;
    }
};

}}} // namespaces

#endif // PLASK__MODULE_ELECTRICAL_ITERATIVE_MATRIX2D_H
