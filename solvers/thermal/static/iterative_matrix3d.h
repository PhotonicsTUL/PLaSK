#ifndef PLASK__MODULE_THERMAL_ITERATIVE_MATRIX3D_H
#define PLASK__MODULE_THERMAL_ITERATIVE_MATRIX3D_H

#include <algorithm>
#include <plask/plask.hpp>

namespace plask { namespace thermal { namespace tstatic {

#undef LDA
#define LDA 16

struct SparseBandMatrix3D {
    const ptrdiff_t size;   ///< Order of the matrix, i.e. number of columns or rows
    ptrdiff_t bno[14];      ///< Vector of non-zero band numbers (shift from diagonal)

    double* data;           ///< Data stored in the matrix

    static constexpr size_t kd = 13;
    static constexpr size_t ld = LDA-1;

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

        data = aligned_malloc<double>(LDA*size);
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
        size_t i = std::find(bno, bno+14, r-c) - bno;
        assert(i != 14);
        return data[LDA*c+i];
    }

    /// Clear the matrix
    void clear() {
        std::fill_n(data, LDA*size, 0.);
    }

    /**
     * Multiplication functor for symmetric banded matrix
     */
    void multiply(double* x, double* y) const { // y = A x
        #pragma omp parallel for
        for (ptrdiff_t r = 0; r < size; ++r) {
            double* datar = data + LDA*r;
            double v = 0.;
            // below diagonal
            for (ptrdiff_t i = 13; i > 0; --i) {
                ptrdiff_t c = r - bno[i];
                if (c >= 0) v += data[LDA*c+i] * x[c];
            }
            // above diagonal
            for (ptrdiff_t i = 0; i < 14; ++i) {
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
struct PrecondJacobi3D {

    const SparseBandMatrix3D& matrix;

    DataVector<double> diag;

    PrecondJacobi3D(const SparseBandMatrix3D& A): matrix(A), diag(A.size) {
        for (double *last = matrix.data + A.size*LDA, *m = matrix.data, *d = diag.data(); m < last; m += LDA, ++d)
            *d = 1. / *m;
    }

    void operator()(double* z, double* r) const { // z = inv(M) r
        const double* d = diag.data(), *zend = z + matrix.size-4;
        for (; z < zend; z += 4, r += 4, d += 4) {
            // *reinterpret_cast<v2double*>(z) = *reinterpret_cast<const v2double*>(r) * *reinterpret_cast<const v2double*>(d);
            // *reinterpret_cast<v2double*>(z+2) = *reinterpret_cast<const v2double*>(r+2) * *reinterpret_cast<const v2double*>(d+2);
            z[0] = r[0] * d[0];
            z[1] = r[1] * d[1];
            z[2] = r[2] * d[2];
            z[3] = r[3] * d[3];
        }
        for (zend += 4; z != zend; ++z, ++r, ++d)
            *z = *r * *d;
    }
};

}}} // namespace plask::thermal::statict3d

#endif // PLASK__MODULE_THERMAL_ITERATIVE_MATRIX3D_H
