#include "gauss_legendre.h"
#include "fortran.h"

#include <boost/math/special_functions/legendre.hpp>
using boost::math::legendre_p;

namespace plask { namespace solvers { namespace slab {
    

void gaussData(size_t n, std::vector<double>& abscissae, DataVector<double>& weights)
{
    int info;

    abscissae.assign(n, 0.);
    weights.reset(n);
    
    for (size_t i = 1; i != n; ++i)
        weights[i-1] = 0.5 / std::sqrt(1. - 0.25/(i*i));

    dsterf(n, &abscissae.front(), weights.data(), info);
    if (info < 0) throw CriticalException("Gauss-Legendre abscissae: Argument %d of DSTERF has bad value", -info);
    if (info > 0) throw ComputationError("Gauss-Legendre abscissae", "Could not converge in %d-th element", info);

    double nn = n*n;
    auto x = abscissae.begin();
    auto w = weights.begin();
    for (; x != abscissae.end(); ++x, ++w) {
        double P = legendre_p(n-1, *x);
        *w = 2. * (1. - (*x)*(*x)) / (nn * P*P);
    }
}
     
}}} // # namespace plask::solvers::slab
