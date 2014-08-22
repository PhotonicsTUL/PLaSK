#ifndef PLASK__SOLVER_SLAB_FFT_H
#define PLASK__SOLVER_SLAB_FFT_H

#include <plask/plask.hpp>

#include <plask/optical/slab/config.h>

#ifdef USE_FFTW
#   include <fftw3.h>
#endif

namespace plask { namespace solvers { namespace slab { namespace FFT {

/**
 * Symmetry of the transform.
 * Depending on the symmetry value, data must be sampled in different points:
 * 0 for arbitrary structure and ½ for even and odd symmetry
 */
enum Symmetry {
    SYMMETRY_EVEN = 0,
    SYMMETRY_NONE = 1,
    SYMMETRY_ODD = 2
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
     * \param lot number of arrays to transform
     * \param n size of a single array
     * \param symmetry symmetry of the transform
     * \param strid data stride (defaults to \c lot)
     */
    Forward1D(int lot, int n, Symmetry symmetry, int strid=0);
    ~Forward1D();
    /** Execute transform
     * \param data data to execute FFT
     */
    void execute(dcomplex* data);
  private:
    int lot;
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
     * \param lot number of arrays to transform
     * \param n1,n2 dimensions of a single array
     * \param symmetry1,symmetry2 symmetry of the transform
     * \param strid data stride (defaults to \c lot)
     * \param ld leading dimension (defaults to \c n1)
     */
    Forward2D(int lot, int n1, int n2, Symmetry symmetry1, Symmetry symmetry2, int strid=0, int ld=0);
    ~Forward2D();
    /** Execute transform
     * \param data data to execute FFT
     */
    void execute(dcomplex* data);
  private:
    int lot;
    int n1, n2;
    int strid, strid2;
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
     * \param lot number of arrays to transform
     * \param n size of a single array
     * \param symmetry symmetry of the transform
     * \param strid data stride (defaults to \c lot)
     */
    Backward1D(int lot, int n, Symmetry symmetry, int strid=0);
    ~Backward1D();
    /** Execute transform
     * \param data data to execute FFT
     */
    void execute(dcomplex* data);
  private:
    int lot;
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
     * \param lot number of arrays to transform
     * \param n1,n2 dimensions of a single array
     * \param symmetry1,symmetry2 symmetry of the transform
     * \param strid data stride (defaults to \c lot)
     * \param ld major row stride (defaults to \c n1)
     */
    Backward2D(int lot, int n1, int n2, Symmetry symmetry1, Symmetry symmetry2, int strid=0, int ld=0);
    ~Backward2D();
    /// Execute transform
    void execute();
    /** Execute transform
     *  \param data data to execute FFT
     */
    void execute(dcomplex* data);
  private:
    int lot;
    int n1, n2;
    int strid, strid2;
    Symmetry symmetry1, symmetry2;
#ifdef USE_FFTW
    fftw_plan plan;
#else
    double *wsave1, *wsave2;
#endif
};

}}}} // namespace plask::solvers::slab::FFT

#endif // PLASK__SOLVER_SLAB_FFT_H
