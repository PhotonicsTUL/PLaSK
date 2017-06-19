#include "gauss_legendre.h"
#include "fortran.h"

namespace plask { namespace solvers { namespace slab {
    

void gaussLaguerre(size_t n, std::vector<double>& abscissae, DataVector<double>& weights, double alpha)
{
    int info;

    abscissae.resize(n);
    weights.reset(n);
    
    for (size_t i = 0; i != n; ++i) {
        abscissae[i] = 2. * i + alpha + 1.;
        weights[i] = sqrt((alpha + i + 1.) * (i + 1.));
    }

    dsterf(n, &abscissae.front(), weights.data(), info);
    if (info < 0) throw CriticalException("Gauss-Laguerre abscissae: Argument {:d} of DSTERF has bad value", -info);
    if (info > 0) throw ComputationError("Gauss-Laguerre abscissae", "Could not converge in {:d}-th element", info);

//    double nn = n*n;
//    auto x = abscissae.begin();
//    auto w = weights.begin();
//    for (; x != abscissae.end(); ++x, ++w) {
//        double P = legendre_p(n-1, *x);
//        *w = 2. * (1. - (*x)*(*x)) / (nn * P*P);
//    }
}
     
}}} // # namespace plask::solvers::slab
