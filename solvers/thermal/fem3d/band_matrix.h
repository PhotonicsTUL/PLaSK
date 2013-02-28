#ifndef PLASK__MODULE_THERMAL_BAND_MATRIX_H
#define PLASK__MODULE_THERMAL_BAND_MATRIX_H

#include <cstddef>
#include <plask/plask.hpp>

#define UPLO 'L'

namespace plask { namespace solvers { namespace thermal3d {

/**
 * Oversimple symmetric band matrix structure. It only offers easy access to elements and nothing more.
 * Data is stored in LAPACK format.
 */
struct DpbMatrix {

    const size_t size;  ///< Order of the matrix, i.e. number of columns or rows
    const size_t band1; ///< Size of the band reduced by one
    double* data;       ///< Pointer to data

    DpbMatrix(size_t rank, size_t major, size_t minor):
        size(rank), band1(minor * (major+1) + 1), data(new double[rank*(band1+1)]) {}
    DpbMatrix(const DpbMatrix&) = delete; // this object is non-copyable
    ~DpbMatrix() { delete[] data; }

    /**
     * Return index in data array
     * \param r index of the element row
     * \param c index of the element column
     */
    size_t index(size_t r, size_t c) {
        assert(r < size && c < size);
        if (r < c) {
            assert(c - r <= band1);
#           if UPLO == 'U'
                return band1 * c + r + band1;
#           else
                return band1 * r + c;
#           endif
        } else {
            assert(r - c <= band1);
#           if UPLO == 'U'
                return band1 * r + c + band1;
#           else
                return band1 * c + r;
#           endif
        }
    }

    /**
     * Return reference to array element
     * \param r index of the element row
     * \param c index of the element column
     **/
    double& operator()(size_t r, size_t c) {
        return data[index(r,c)];
    }
};

struct SparseBandMatrix {
    const size_t size;  ///< Order of the matrix, i.e. number of columns or rows
    size_t bno[14];     ///< Vector of non-zero band numbers

    double* data[14];   ///< Data stored in the matrix

    SparseBandMatrix(size_t size, size_t major, size_t minor): size(size) {

        for (size_t i = 0; i < 14; ++i) {
            data[i] = new double[size];     //TODO (maybe) allocate shorter bands
        }
    }
};

}}} // namespace plask::solver::thermal3d

#endif // PLASK__MODULE_THERMAL_BAND_MATRIX_H
