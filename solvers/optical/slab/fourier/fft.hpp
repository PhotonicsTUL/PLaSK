#ifndef PLASK__SOLVER_SLAB_FFT_H
#define PLASK__SOLVER_SLAB_FFT_H

#include <plask/plask.hpp>

#include "plask/optical/slab/config.hpp"

#ifdef USE_FFTW
#   include <fftw3.h>
#endif

namespace plask { namespace optical { namespace slab { namespace FFT {

/**
 * Symmetry of the transform.
 * Depending on the symmetry value, data must be sampled in different points:
 * 0 for arbitrary structure and ½ for even and odd symmetry
 */
enum Symmetry {
    SYMMETRY_NONE = 0,
    SYMMETRY_EVEN_2 = 1,
    SYMMETRY_ODD_2 = 2,
    SYMMETRY_EVEN_1 = 5,
    SYMMETRY_ODD_1 = 6
};

/// Fourier transform of multiple 1D arrays
struct PLASK_SOLVER_API Forward1D {
    /// Create uninitialized transform
    Forward1D();
    /// Move constructor
    Forward1D(Forward1D&& old);
    /// Assignment operator
    Forward1D& operator=(Forward1D&& old);
    /** Init transfrom
     * \param strid data stride
     * \param n size of a single array
     * \param symmetry symmetry of the transform
     */
    Forward1D(std::size_t strid, std::size_t n, Symmetry symmetry);
    ~Forward1D();
    /** Execute transform
     * \param data data to execute FFT
     * \param lot number of arrays to transform, defaults to \c strid
     */
    void execute(dcomplex* data, int lot=0);
  private:
    int n;
    int strid;
    Symmetry symmetry;
#ifdef USE_FFTW
    fftw_plan plan;
#else
    double *wsave;
#endif
};

/// Fourier transform of multiple 2D arrays
struct PLASK_SOLVER_API Forward2D {
    /// Create uninitialized transform
    Forward2D();
    /// Move constructor
    Forward2D(Forward2D&& old);
    /// Assignment operator
    Forward2D& operator=(Forward2D&& old);
    /** Init transfrom
     * \param strid data stride
     * \param n1,n2 dimensions of a single array
     * \param symmetry1,symmetry2 symmetry of the transform
     * \param ld leading dimension (defaults to \c n1)
     */
    Forward2D(std::size_t strid, std::size_t n1, std::size_t n2, Symmetry symmetry1, Symmetry symmetry2, std::size_t ld=0);
    ~Forward2D();
    /** Execute transform
     * \param data data to execute FFT
     * \param lot number of arrays to transform, defaults to \c strid
     */
    void execute(dcomplex* data, int lot=0);
  private:
    int n1, n2;
    int strid1, strid2;
    Symmetry symmetry1, symmetry2;
#ifdef USE_FFTW
    fftw_plan plan;
#else
    double *wsave1, *wsave2;
#endif
};

/// Fourier transform of multiple 1D arrays
struct PLASK_SOLVER_API Backward1D {
    /// Create uninitialized transform
    Backward1D();
    /// Move constructor
    Backward1D(Backward1D&& old);
    /// Assignment operator
    Backward1D& operator=(Backward1D&& old);
    /** Init transfrom
     * \param strid data stride
     * \param n size of a single array
     * \param symmetry symmetry of the transform
     */
    Backward1D(std::size_t strid, std::size_t n, Symmetry symmetry);
    ~Backward1D();
    /** Execute transform
     * \param data data to execute FFT
     * \param lot number of arrays to transform, defaults to \c strid
     */
    void execute(dcomplex* data, int lot=0);
  private:
    int n;
    int strid;
    Symmetry symmetry;
#ifdef USE_FFTW
    fftw_plan plan;
#else
    double *wsave;
#endif
};

/// Fourier transform of multiple 2D arrays
struct PLASK_SOLVER_API Backward2D {
    /// Create uninitialized transform
    Backward2D();
    /// Move constructor
    Backward2D(Backward2D&& old);
    /// Assignment operator
    Backward2D& operator=(Backward2D&& old);
    /** Init transfrom
     * \param strid data stride
     * \param n1,n2 dimensions of a single array
     * \param symmetry1,symmetry2 symmetry of the transform
     * \param ld major row stride (defaults to \c n1)
     */
    Backward2D(std::size_t strid, std::size_t n1, std::size_t n2, Symmetry symmetry1, Symmetry symmetry2, std::size_t ld=0);
    ~Backward2D();
    /// Execute transform
    void execute();
    /** Execute transform
     *  \param data data to execute FFT
     * \param lot number of arrays to transform, defaults to \c strid
     */
    void execute(dcomplex* data, int lot=0);
  private:
    int n1, n2;
    int strid1, strid2;
    Symmetry symmetry1, symmetry2;
#ifdef USE_FFTW
    fftw_plan plan;
#else
    double *wsave1, *wsave2;
#endif
};

}}}} // namespace plask::optical::slab::FFT

#endif // PLASK__SOLVER_SLAB_FFT_H
