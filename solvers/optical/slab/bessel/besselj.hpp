#ifndef PLASK__SOLVER__SLAB_BESSELJ_H
#define PLASK__SOLVER__SLAB_BESSELJ_H


#include "plask/optical/slab/config.hpp"


#ifdef USE_GSL
#   include <gsl/gsl_sf_bessel.h>

    inline double cyl_bessel_j(int m, double x) {
        return gsl_sf_bessel_Jn(m, x);
    }

    template <typename It>
    inline void cyl_bessel_j_zero(int m, size_t i, size_t n, It dst) {
        if (m == 0) {
            for (n = n+i; i < n; ++i, ++dst) *dst = gsl_sf_bessel_zero_J0(i);
        } else if (m == 1) {
            for (n = n+i; i < n; ++i, ++dst) *dst = gsl_sf_bessel_zero_J1(i);
        } else {
            double nu = m;
            for (n = n+i; i < n; ++i, ++dst) *dst = gsl_sf_bessel_zero_Jnu(nu, i);
        }
    }

#else
#   include <boost/math/special_functions/bessel.hpp>
    using boost::math::cyl_bessel_j;
    using boost::math::cyl_bessel_j_zero;
#endif


#endif // PLASK__SOLVER__SLAB_BESSELJ_H
