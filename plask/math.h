#ifndef PLASK__NUMBERS_H
#define PLASK__NUMBERS_H

#include <plask/config.h>

#ifdef PLASK_MATH_STD

// Math library
#include <cmath>

// Complex numbers library
#ifdef PLASK_MATH_STD
#   include <complex>
    namespace plask {
        using std::complex; using std::conj;
        using std::abs; using std::real; using std::imag;
        typedef complex<double> dcomplex;
    }
#endif // PLASK_MATH_STD

// Limist for comparing approximate numbers with zero
#include <limits>
namespace plask {
    const double SMALL = std::numeric_limits<double>::epsilon();
    const double SMALL2 = SMALL*SMALL;

    /// Check if an approximate number is zero
    inline bool is_zero(double v) {
        return abs(v) < SMALL;
    }

    /// Check if an approximate number is zero
    inline bool is_zero(dcomplex v) {
        return real(v)*real(v) + imag(v)*imag(v) < SMALL2;
    }
}


#endif // PLASK_MATH_STD


#endif // PLASK__NUMBERS_H