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
    SYMMETRY_NONE,
    SYMMETRY_EVEN,
    SYMMETRY_ODD
};

/// Fourier transform of multiple 1D arrays
struct Forward1D {
    /// Create uninitialized transform
    Forward1D();
    /// Move constructor
    Forward1D(Forward1D&& old);
    /// Assignment operator
    Forward1D& operator=(Forward1D&& old);
    /** Init transfrom
     * \param lot number of arrays to transform
     * \param n size of a single array
     * \param data pointer to data to transform
     * \param symmetry symmetry of the transform
     */
    Forward1D(int lot, int n, Symmetry symmetry, dcomplex* data);
    ~Forward1D();
    /// Execute transform
    void execute();
    /** Execute transform
     * \param data tata to execute FFT
     */
    void execute(dcomplex* data);
  private:
    int lot;
    int n;
    Symmetry symmetry;
    dcomplex* data;
#ifdef USE_FFTW
    fftw_plan plan;
#else
    double *wsave;
#endif
};

/// Fourier transform of multiple 2D arrays
struct Forward2D {
    /// Create uninitialized transform
    Forward2D();
    /// Move constructor
    Forward2D(Forward2D&& old);
    /// Assignment operator
    Forward2D& operator=(Forward2D&& old);
    /** Init transfrom
     * \param lot number of arrays to transform
     * \param n1,n2 dimensions of a single array
     * \param data pointer to data to transform
     * \param symmetry1,symmetry2 symmetry of the transform
     */
    Forward2D(int lot, int n1, int n2, Symmetry symmetry1, Symmetry symmetry2, dcomplex* data);
    ~Forward2D();
    /// Execute transform
    void execute();
    /** Execute transform
     * \param data tata to execute FFT
     */
    void execute(dcomplex* data);
  private:
    int lot;
    int n1, n2;
    Symmetry symmetry1, symmetry2;
    dcomplex* data;
#ifdef USE_FFTW
    fftw_plan plan;
#else
    double *wsave;
#endif
};

/// Fourier transform of multiple 1D arrays
struct Backward1D {
    /// Create uninitialized transform
    Backward1D();
    /// Move constructor
    Backward1D(Backward1D&& old);
    /// Assignment operator
    Backward1D& operator=(Backward1D&& old);
    /** Init transfrom
     * \param lot number of arrays to transform
     * \param n size of a single array
     * \param data pointer to data to transform
     * \param symmetry symmetry of the transform
     */
    Backward1D(int lot, int n, Symmetry symmetry, dcomplex* data);
    ~Backward1D();
    /// Execute transform
    void execute();
    /** Execute transform
     * \param data tata to execute FFT
     */
    void execute(dcomplex* data);
  private:
    int lot;
    int n;
    Symmetry symmetry;
    dcomplex* data;
#ifdef USE_FFTW
    fftw_plan plan;
#else
    double *wsave;
#endif
};

/// Fourier transform of multiple 2D arrays
struct Backward2D {
    /// Create uninitialized transform
    Backward2D();
    /// Move constructor
    Backward2D(Backward2D&& old);
    /// Assignment operator
    Backward2D& operator=(Backward2D&& old);
    /** Init transfrom
     * \param lot number of arrays to transform
     * \param n1,n2 dimensions of a single array
     * \param data pointer to data to transform
     * \param symmetry1,symmetry2 symmetry of the transform
     */
    Backward2D(int lot, int n1, int n2, Symmetry symmetry1, Symmetry symmetry2, dcomplex* data);
    ~Backward2D();
    /// Execute transform
    void execute();
    /** Execute transform
     *  \param data tata to execute FFT
     */
    void execute(dcomplex* data);
  private:
    int lot;
    int n1, n2;
    Symmetry symmetry1, symmetry2;
    dcomplex* data;
#ifdef USE_FFTW
    fftw_plan plan;
#else
    double *wsave;
#endif
};

}}}} // namespace plask::solvers::slab::FFT

#endif // PLASK__SOLVER_SLAB_FFT_H
