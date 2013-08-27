#ifndef PLASK__FFT_H
#define PLASK__SOLVER_SLAB_FFT_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace slab {
    
/**
 * Perform Fourier transform of \c howmany 1D arrays of size \c n
 * \param howmany number of arrays to transform
 * \param n size of a single array
 * \param data pointer to data to transform
 */
void performFFT(size_t howmany, size_t n, dcomplex* data);
    
/**
 * Perform symmetric Fourier transform of \c howmany 1D arrays of size \c n
 * \param howmany number of arrays to transform
 * \param n size of a single array
 * \param data pointer to data to transform
 */
void performSymFFT(size_t howmany, size_t n, dcomplex* data);

/**
 * Perform Fourier transform of \c howmany 2D arrays of size \c n
 * \param howmany number of arrays to transform
 * \param n1,n2 dimensions of a single array
 * \param data pointer to data to transform
 */
void performFFT(size_t howmany, size_t n1, size_t n2, dcomplex* data);

/**
 * Perform symmetric Fourier transform of \c howmany 2D arrays of size \c n
 * \param howmany number of arrays to transform
 * \param n1,n2 dimensions of a single array
 * \param data pointer to data to transform
 */
void performSymFFT(size_t howmany, size_t n1, size_t n2, dcomplex* data);


/**
 * Perform inverse Fourier transform of \c howmany 1D arrays of size \c n
 * \param howmany number of arrays to transform
 * \param n size of a single array
 * \param data pointer to data to transform
 */
void performIFFT(size_t howmany, size_t n, dcomplex* data);
    
/**
 * Perform inverse symmetric Fourier transform of \c howmany 1D arrays of size \c n
 * \param howmany number of arrays to transform
 * \param n size of a single array
 * \param data pointer to data to transform
 */
void performEvenIFFT(size_t howmany, size_t n, dcomplex* data);

/**
 * Perform inverse symmetric Fourier transform of \c howmany 1D arrays of size \c n
 * \param howmany number of arrays to transform
 * \param n size of a single array
 * \param data pointer to data to transform
 */
void performOddIFFT(size_t howmany, size_t n, dcomplex* data);


/**
 * Perform inverse Fourier transform of \c howmany 1D arrays of size \c n
 * \param howmany number of arrays to transform
 * \param n1,n2 dimensions of a single array
 * \param data pointer to data to transform
 */
void performIFFT(size_t howmany, size_t n1, size_t n2, dcomplex* data);
    
/**
 * Perform inverse symmetric Fourier transform of \c howmany 1D arrays of size \c n
 * \param howmany number of arrays to transform
 * \param n1,n2 dimensions of a single array
 * \param data pointer to data to transform
 */
void performEvenIFFT(size_t howmany, size_t n1, size_t n2, dcomplex* data);

/**
 * Perform inverse symmetric Fourier transform of \c howmany 1D arrays of size \c n
 * \param howmany number of arrays to transform
 * \param n1,n2 dimensions of a single array
 * \param data pointer to data to transform
 */
void performOddIFFT(size_t howmany, size_t n1, size_t n2, dcomplex* data);


}}} // namespace plask::solvers::slab

#endif // PLASK__SOLVER_SLAB_FFT_H