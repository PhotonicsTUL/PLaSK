#include "gauss_legendre.h"
#include "fortran.h"

#include <boost/math/tools/convert_from_string.hpp>
#include <boost/math/special_functions/legendre.hpp>
using boost::math::legendre_p;

namespace plask { namespace optical { namespace slab {


void gaussLegendre(size_t n, std::vector<double>& abscissae, DataVector<double>& weights)
{
    int info;

    abscissae.assign(n, 0.);
    weights.reset(n);

    for (size_t i = 1; i != n; ++i)
        weights[i-1] = 0.5 / std::sqrt(1. - 0.25/double(i*i));

    dsterf(int(n), &abscissae.front(), weights.data(), info);
    if (info < 0) throw CriticalException("Gauss-Legendre abscissae: Argument {:d} of DSTERF has bad value", -info);
    if (info > 0) throw ComputationError("Gauss-Legendre abscissae", "Could not converge in {:d}-th element", info);

    double nn = double(n*n);
    auto x = abscissae.begin();
    auto w = weights.begin();
    for (; x != abscissae.end(); ++x, ++w) {
        double P = legendre_p(int(n-1), *x);
        *w = 2. * (1. - (*x)*(*x)) / (nn * P*P);
    }
}

}}} // # namespace plask::optical::slab
