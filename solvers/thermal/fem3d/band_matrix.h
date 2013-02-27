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

    std::size_t size;   ///< Order of the matrix, i.e. number of columns or rows
    std::size_t band1;  ///< Size of the band reduced by one
    double* data;       ///< Pointer to data

    DpbMatrix(): size(0), band1(0), data(nullptr) {}
    DpbMatrix(std::size_t rank, std::size_t band): size(rank), band1(band-1), data(new double[rank*band]) {}
    DpbMatrix(const DpbMatrix&) = delete; // this object is non-copyable
    ~DpbMatrix() { delete[] data; }

    /// Init the existing matrix
    void init(std::size_t rank, std::size_t band) {
        if (data) throw CriticalException("DpbMatrix already initialized");
        size = rank;
        band1 = band - 1;
        data = new double[rank*band];
    }

    /**
     * Return index in data array
     * \param r index of the element row
     * \param c index of the element column
     */
    size_t index(std::size_t r, std::size_t c) {
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
    double& operator()(std::size_t r, std::size_t c) {
        return data[index(r,c)];
    }
};

struct SparseBandMatrix {
    std::size_t size;   ///< Order of the matrix, i.e. number of columns or rows
    std::size_t bands;  ///< Number of the nonzero bands
    std::size_t *band;  ///< Vector of non-zero bad numbers

    double** data;      ///< Data stored in the matrix
};

}}} // namespace plask::solver::thermal3d

#endif // PLASK__MODULE_THERMAL_BAND_MATRIX_H
