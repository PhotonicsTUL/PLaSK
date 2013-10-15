#ifndef PLASK__NUMBERS_H
#define PLASK__NUMBERS_H

#include <plask/config.h>

#include <cmath>
#include <limits>
#include <algorithm>

#ifdef PLASK_MATH_STD
#   include <complex>
#endif // PLASK_MATH_STD

#ifndef M_PI
#   define M_PI 3.14159265358979323846
#endif


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
    using std::isnan; using std::isinf;
    typedef complex<double> dcomplex;
    const dcomplex I(0.,1.);
#endif // PLASK_MATH_STD

const double PI = M_PI;
const double PI_DOUBLED = 6.28318530717958647692;

// Limits for comparing approximate numbers with zero
constexpr double SMALL = std::numeric_limits<double>::epsilon(); ///< The numeric precision limit
constexpr double SMALL2 = SMALL*SMALL; ///< Squared numeric precision limit

/// Check if the real number is almost zero.
/// \param v number to verify
/// \param abs_supremum
/// \return @c true only if abs(v) < abs_supremum
inline bool is_zero(double v, double abs_supremum = SMALL) {
    return abs(v) < abs_supremum;
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

/**
 * Wrapper over std::fma witch works for all types.
 * For float, double and long double it calls std::fma, and for rest it just calculates to_mult_1 * to_mult_2 + to_sum.
 * @param to_mult_1, to_mult_2, to_sum
 * @return result of to_mult_1 * to_mult_2 + to_sum, typically with better precision
 */
template <typename T1, typename T2, typename T3>
auto inline fma(T1 to_mult_1, T2 to_mult_2, T3 to_sum) -> decltype(to_mult_1*to_mult_2+to_sum) {
    return to_mult_1 * to_mult_2 + to_sum;
}

inline float fma(float to_mult_1, float to_mult_2, float to_sum) {
    return std::fma(to_mult_1, to_mult_2, to_sum);
}

inline double fma(double to_mult_1, double to_mult_2, double to_sum) {
    return std::fma(to_mult_1, to_mult_2, to_sum);
}

inline long double fma(long double to_mult_1, long double to_mult_2, long double to_sum) {
    return std::fma(to_mult_1, to_mult_2, to_sum);
}

/**
 * Array type for SIMD operations. It contains two 64-bit doubles.
 * TODO: not portable, make falback to struct{double,double}
 */
typedef double v2double __attribute__((vector_size(16)));
/// View allowing access to elements of v2double
typedef union { v2double simd; double v[2]; } v2double_view;

} // namespace plask

#endif // PLASK__NUMBERS_H
