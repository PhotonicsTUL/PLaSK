#ifndef PLASK__NUMBERS_H
#define PLASK__NUMBERS_H

#include <plask/config.h>

#include <cmath>
#include <limits>
#include <algorithm>

#include "exceptions.h"
#include <sstream>

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

#ifdef __GNUC__
/**
 * Array type for SIMD operations. It contains two 64-bit doubles.
 * TODO: not portable, make falback to struct{double,double}
 */
typedef double v2double __attribute__((vector_size(16)));
/// View allowing access to elements of v2double
typedef union { v2double simd; double v[2]; } v2double_view;
#endif


// Total order double comparision with NaN greater than all other numbers:

/**
 * Check if two doubles are equals.
 *
 * It differs from standard == operator in case of comparing NaN-s: dbl_compare_eq(NaN, NaN) is @c true.
 * @param x, y numbers to compare
 * @return @c true only if @p x equals to @p y
 */
inline bool dbl_compare_eq(double x, double y) {
    if (std::isnan(x)) return std::isnan(y);
    return x == y;
}

/**
 * Check if @p x is less than @p y.
 *
 * It differs from standard \< operator in case of comparing NaN-s: NaN is greater than all other numbers.
 * It is fine to use this to sort doubles or store doubles in standard containers (which is not true in case of standard \< operator).
 * @param x, y numbers to compare
 * @return @c true only if @p x is less than @p y
 */
inline bool dbl_compare_lt(double x, double y) {
      if (std::isnan(y)) return !std::isnan(x);
      return x < y; // if x is NaN it is grater than non-NaN y and std. < operator returns false
}

/**
 * Check if @p x is greater than @p y.
 *
 * It differs from standard \> operator in case of comparing NaN-s: NaN is greater than all other numbers.
 * It is fine to use this to sort doubles or store doubles in standard containers (which is not true in case of standard \> operator).
 * @param x, y numbers to compare
 * @return @c true only if @p x is greater than @p y
 */
inline bool dbl_compare_gt(double x, double y) { return dbl_compare_lt(y, x); }

/**
 * Check if @p x is less or equals to @p y.
 *
 * It differs from standard \>= operator in case of comparing NaN-s: NaN is greater than all other numbers.
 * @param x, y numbers to compare
 * @return @c true only if @p x is less or equals to @p y.
 */
inline bool dbl_compare_lteq(double x, double y) { return !dbl_compare_gt(x, y); }

/**
 * Check if @p x is greater or equals to @p y.
 *
 * It differs from standard \<= operator in case of comparing NaN-s: NaN is greater than all other numbers.
 * @param x, y numbers to compare
 * @return @c true only if @p x is greater or equals to @p y.
 */
inline bool dbl_compare_gteq(double x, double y) { return !dbl_compare_lt(x, y); }

/**
 * Exception thrown by complex parser when complex number is ill-formated.
 */
struct PLASK_API IllFormatedComplex: public Exception {

    /**
     * Constructor.
     * @param str_to_parse ill-formated complex number
     */
    IllFormatedComplex(const std::string& str_to_parse): Exception("Ill-formatted complex number \"{0}\". Allowed formats: 'R+Ij', 'R,Ij', '(R, I)', where R and I are floating point numbers.", str_to_parse)
    {}

};

/**
 * Parse complex number in format: R+Ij, R, Ij, or (R, I), where R and I are floating point numbers (last is standard C++ format).
 * @param str_to_parse string to parse
 * @return parsed complex number
 * @throw IllFormatedComplex when @p str_to_parse is in bad format
 */
template <typename T>
std::complex<T> parse_complex(std::string str_to_parse) {
    boost::trim(str_to_parse);
    if (str_to_parse.length() && str_to_parse[0] == '(' && str_to_parse[str_to_parse.length()-1] == ')' &&
        str_to_parse.find(',') == std::string::npos)
        str_to_parse = str_to_parse.substr(1, str_to_parse.length()-2);
    std::istringstream to_parse(str_to_parse);
    auto check_eof = [&] () {
        if (!to_parse.eof()) {  // we require end-of stream here
            char c;
            to_parse >> c;  // we check if there is non-white character, this operation should fail
            if (to_parse) throw IllFormatedComplex(str_to_parse);
        }
    };
    T real, imag;
    to_parse >> real;
    if (to_parse.fail()) {  // we will try standard >> operator
        to_parse.clear(); to_parse.str(str_to_parse);
        std::complex<T> res;
        to_parse >> res;
        if (to_parse.fail()) throw IllFormatedComplex(str_to_parse);
        check_eof();
        return res;
    }
    if (to_parse.eof()) return std::complex<T>(real);
    char c;
    to_parse >> c;
    if (to_parse.fail()) throw IllFormatedComplex(str_to_parse);
    if (to_parse.eof()) return std::complex<T>(real);
    if (c == 'i' || c == 'j') { // only imaginary part is given
        imag = real;
        real = 0.0;
    } else if (c == '+' || c == '-') {
        char c_ij;
        to_parse >> imag >> c_ij;
        if (to_parse.fail() || (c_ij != 'i' && c_ij != 'j')) throw IllFormatedComplex(str_to_parse);
        if (c == '-') imag = -imag;
    } else
         throw IllFormatedComplex(str_to_parse);
    check_eof();
    return std::complex<T>(real, imag);
}

extern template PLASK_API std::complex<double> parse_complex<double>(std::string str_to_parse);


/**
 * Allow to compute sum of doubles much more accurate than directly.
 *
 * It uses O(1) memory (stores 2 doubles) and all it's operation has O(1) time complexity.
 *
 * It use Kahan method
 * http://en.wikipedia.org/wiki/Kahan_summation_algorithm
 */
class AccurateSum {

    double s, c;    // c is a running compensation for lost low-order bits.

public:

    AccurateSum(double initial = 0.0): s(initial), c(0.0) {}

    AccurateSum& operator=(double v);

    AccurateSum(const AccurateSum& initial) = default;
    AccurateSum& operator=(const AccurateSum& v) = default;

    AccurateSum& operator+=(double v);
    AccurateSum& operator-=(double v) { return *this += -v; }

    operator double() const { return s; }

    AccurateSum& operator+=(const AccurateSum& other);

};


} // namespace plask

#endif // PLASK__NUMBERS_H
