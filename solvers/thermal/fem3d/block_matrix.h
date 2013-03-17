#ifndef PLASK__MODULE_THERMAL_BLOCK_MATRIX_H
#define PLASK__MODULE_THERMAL_BLOCK_MATRIX_H

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
    const size_t bands; ///< Size of the band reduced by one
    double* data;       ///< Pointer to data

    /**
     * Create matrix
     * \param rank size of the matrix
     * \param major shift of nodes to the next major row (mesh[x,y,z+1])
     * \param minor shift of nodes to the next minor row (mesh[x,y+1,z])
     */
    DpbMatrix(size_t rank, size_t major, size_t minor):
        size(rank), bands(major+minor+1), data(new double[rank*(major+minor+2)]) {}

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
            assert(c - r <= bands);
#           if UPLO == 'U'
                return bands * c + r + bands;
#           else
                return bands * r + c;
#           endif
        } else {
            assert(r - c <= bands);
#           if UPLO == 'U'
                return bands * r + c + bands;
#           else
                return bands * c + r;
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

    /// Clear the matrix
    void clear() {
        std::fill_n(data, size * (bands+1), 0.);
    }
};

}}} // namespace plask::solver::thermal3d

#endif // PLASK__MODULE_THERMAL_BLOCK_MATRIX_H
