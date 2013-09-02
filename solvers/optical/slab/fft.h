#ifndef PLASK__SOLVER_SLAB_FFT_H
#define PLASK__SOLVER_SLAB_FFT_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace slab {

struct FFT {

    /**
     * Symmetry of the transform.
     * Depending on the symmetry value, data must be sampled in different points:
     * 0 for arbitrary structure and ½ for even and odd symmetry
     */
    enum Symmetry {
        SYMMETRY_NONE,
        SYMMETRY_EVEN,
        SYMMETRY_ODD
    };

    /// General constructor
    FFT();

    // General destructor
    ~FFT();

    /**
     * Perform Fourier transform of \c howmany 1D arrays of size \c n
     * \param howmany number of arrays to transform
     * \param n size of a single array
     * \param data pointer to data to transform
     * \param symmetry symmetry of the transform
     */
    void forward(size_t howmany, size_t n, dcomplex* data, Symmetry symmetry);

    /**
     * Perform Fourier transform of \c howmany 2D arrays of size \c n
     * \param howmany number of arrays to transform
     * \param n1,n2 dimensions of a single array
     * \param data pointer to data to transform
     * \param symmetry1,symmetry2 symmetry of the transform
     */
    void forward(size_t howmany, size_t n1, size_t n2, dcomplex* data, Symmetry symmetry1, Symmetry symmetry2);


    /**
     * Perform inverse Fourier transform of \c howmany 1D arrays of size \c n
     * \param howmany number of arrays to transform
     * \param n size of a single array
     * \param data pointer to data to transform
     * \param symmetry symmetry of the transform
     */
    void backward(size_t howmany, size_t n, dcomplex* data, Symmetry symmetry);

    /**
     * Perform inverse Fourier transform of \c howmany 1D arrays of size \c n
     * \param howmany number of arrays to transform
     * \param n1,n2 dimensions of a single array
     * \param data pointer to data to transform
     * \param symmetry1,symmetry2 symmetry of the transform
     */
    void backward(size_t howmany, size_t n1, size_t n2, dcomplex* data, Symmetry symmetry1, Symmetry symmetry2);
};

}}} // namespace plask::solvers::slab

#endif // PLASK__SOLVER_SLAB_FFT_H
