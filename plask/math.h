#ifndef PLASK__NUMBERS_H
#define PLASK__NUMBERS_H

#include <plask/config.h>

// Math library
#include <cmath>

// Complex numbers library
#ifdef PLASK_MATH_STD
#   include <complex>
    namespace plask {
        using std::complex; using std::conj;
        using std::abs; using std::real; using std::imag;
        typedef complex<double> dcomplex;
        const dcomplex I(0.,1.);
    }
#endif // PLASK_MATH_STD


#include <limits>
namespace plask {
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
}


namespace plask {
    // C++ is lacking some operators
    inline plask::dcomplex operator*(int a, const plask::dcomplex& b) { return double(a) * b; }
    inline plask::dcomplex operator*(const plask::dcomplex& a, int b) { return a * double(b); }
    inline plask::dcomplex operator*(unsigned a, const plask::dcomplex& b) { return double(a) * b; }
    inline plask::dcomplex operator*(const plask::dcomplex& a, unsigned b) { return a * double(b); }
    inline plask::dcomplex operator/(int a, const plask::dcomplex& b) { return double(a) / b; }
    inline plask::dcomplex operator/(const plask::dcomplex& a, int b) { return a / double(b); }
    inline plask::dcomplex operator/(unsigned a, const plask::dcomplex& b) { return double(a) / b; }
    inline plask::dcomplex operator/(const plask::dcomplex& a, unsigned b) { return a / double(b); }
}

#endif // PLASK__NUMBERS_H
