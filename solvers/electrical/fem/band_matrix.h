#ifndef PLASK__MODULE_THERMAL_BAND_MATRIX_H
#define PLASK__MODULE_THERMAL_BAND_MATRIX_H

#include <cstddef>
#include <boost/concept_check.hpp>

#define UPLO 'L'

// LAPACK routines to solve set of linear equations
#define dpbtrf F77_GLOBAL(dpbtrf,DPBTRF)
F77SUB dpbtrf(const char& uplo, const int& n, const int& kd, double* ab, const int& ldab, int& info);

#define dpbtf2 F77_GLOBAL(dpbtf2,DPBTF2)
F77SUB dpbtf2(const char& uplo, const int& n, const int& kd, double* ab, const int& ldab, int& info);

#define dpbtrs F77_GLOBAL(dpbtrs,DPBTRS)
F77SUB dpbtrs(const char& uplo, const int& n, const int& kd, const int& nrhs, double* ab, const int& ldab, double* b, const int& ldb, int& info);

namespace plask { namespace solvers { namespace electrical {

/**
 * Oversimple symmetric band matrix structure. It only offers easy access to elements and nothing more.
 * Data is stored in LAPACK format.
 */
struct DpbMatrix {

    std::size_t size;   ///< order of the matrix, i.e. number of columns or rows
    std::size_t bands;  ///< size of the band reduced by one
    double* data;       ///< pointer to data

    DpbMatrix(std::size_t rank, std::size_t band): size(rank), bands(band-1), data(new double[rank*band]) {}
    DpbMatrix(const DpbMatrix&) = delete; // this object is non-copyable
    ~DpbMatrix() { delete[] data; }

    /**
     * Return index in data array
     * \param r index of the element row
     * \param c index of the element column
     */
    size_t index(std::size_t r, std::size_t c) {
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
    double& operator()(std::size_t r, std::size_t c) {
        return data[index(r,c)];
    }
};

}}} // namespace plask::solver::thermal

#endif // PLASK__MODULE_THERMAL_BAND_MATRIX_H
