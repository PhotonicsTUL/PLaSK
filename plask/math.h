#ifndef PLASK__NUMBERS_H
#define PLASK__NUMBERS_H

#include <plask/config.h>

#include <cmath>
#include <limits>
#include <algorithm>

#ifdef PLASK_MATH_STD
#   include <complex>
#endif // PLASK_MATH_STD

//class T;  //Piotr: declaration like this hides some serious errors (like missing template <typename T>)!
namespace plask {

// size_t is preferred for array indexing
using std::size_t;
using std::ptrdiff_t;

// Complex numbers library
#ifdef PLASK_MATH_STD
    using std::complex; using std::conj;
    using std::sqrt;
    using std::abs; using std::real; using std::imag;
    using std::log; using std::exp;
    using std::sin; using std::cos; using std::tan;
    using std::sinh; using std::cosh; using std::tanh;
    using std::asin; using std::acos; using std::atan; using std::atan2;
    using std::asinh; using std::acosh; using std::atanh;
    using std::isnan;
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


// Useful functions
using std::max; using std::min;

inline double abs2(const dcomplex& x) {
    return real(x)*real(x) + imag(x)*imag(x);
}

/**
 * Clamp value to given range.
 * @param v value to clamp
 * @param min, max minimal and maximal value which can be returned
 * @return @p min if v < min, @p max if v > max, @p v in another cases
 */
template <typename T>
const T& clamp(const T& v, const T& min, const T& max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

/**
 * Check if value @p v is in given range [beg, end).
 * @param v value to check
 * @param beg, end ends of range [beg, end)
 * @return @c true only if beg <= v && v < end
 */
template <typename T>
const bool in_range(const T& v, const T& beg, const T& end) {
    return beg <= v && v < end;
}

} // namespace plask

#endif // PLASK__NUMBERS_H
