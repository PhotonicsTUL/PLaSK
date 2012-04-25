#ifndef PLASK__NUMBERS_H
#define PLASK__NUMBERS_H

#include <plask/config.h>

#include <cmath>
#include <limits>

#ifdef PLASK_MATH_STD
#   include <complex>
#endif // PLASK_MATH_STD

namespace plask {

// size_t is preferred for array indexing
using std::size_t;

// Complex numbers library
#ifdef PLASK_MATH_STD
    using std::complex; using std::conj;
    using std::abs; using std::real; using std::imag;
    typedef complex<double> dcomplex;
    const dcomplex I(0.,1.);
#endif // PLASK_MATH_STD



// Limits for comparing approximate numbers with zero
const double SMALL = std::numeric_limits<double>::epsilon(); ///< The numeric precision limit
const double SMALL2 = SMALL*SMALL; ///< Squared numeric precision limit

/// Check if the real number is almost zero
/// \param v number to verify
inline bool is_zero(double v) {
    return abs(v) < SMALL;
}

/// Check if the complex number is almost zero
/// \param v number to verify
inline bool is_zero(dcomplex v) {
    return real(v)*real(v) + imag(v)*imag(v) < SMALL2;
}



// C++ is lacking some operators
inline plask::dcomplex operator*(int a, const plask::dcomplex& b) { return double(a) * b; }
inline plask::dcomplex operator*(const plask::dcomplex& a, int b) { return a * double(b); }
inline plask::dcomplex operator*(unsigned a, const plask::dcomplex& b) { return double(a) * b; }
inline plask::dcomplex operator*(const plask::dcomplex& a, unsigned b) { return a * double(b); }
inline plask::dcomplex operator/(int a, const plask::dcomplex& b) { return double(a) / b; }
inline plask::dcomplex operator/(const plask::dcomplex& a, int b) { return a / double(b); }
inline plask::dcomplex operator/(unsigned a, const plask::dcomplex& b) { return double(a) / b; }
inline plask::dcomplex operator/(const plask::dcomplex& a, unsigned b) { return a / double(b); }



} // namespace plask

#endif // PLASK__NUMBERS_H
