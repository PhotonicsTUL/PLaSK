#include "expansioncyl-infini.h"
#include "solvercyl.h"
#include "zeros-data.h"

#include "../gauss_legendre.h"

#include <boost/math/special_functions/bessel.hpp>
using boost::math::cyl_bessel_j;

#define SOLVER static_cast<BesselSolverCyl*>(solver)

namespace plask { namespace solvers { namespace slab {

ExpansionBesselInfini::ExpansionBesselInfini(BesselSolverCyl* solver): ExpansionBessel(solver)
{
}


void ExpansionBesselInfini::init2()
{
    SOLVER->writelog(LOG_DETAIL, "Preparing Bessel functions for m = {}", m);
    estimateIntegrals();
    eps0.resize(solver->lcount);
}


void ExpansionBesselInfini::reset()
{
    eps0.clear();
    ExpansionBessel::reset();
}


void ExpansionBesselInfini::layerIntegrals(size_t layer, double lam, double glam)
{
    eps0[layer] = integrateLayer(layer, lam, glam, true);
}


void ExpansionBesselInfini::getMatrices(size_t layer, cmatrix& RE, cmatrix& RH)
{
    size_t N = SOLVER->size;
    dcomplex ik0 = 1. / k0;
    double b = rbounds[rbounds.size()-1];

    const Integrals& eps = layers_integrals[layer];

    std::fill(RH.begin(), RH.end(), 0.);
    for (size_t i = 0; i != N; ++i) {
        size_t is = idxs(i); size_t ip = idxp(i);
        double i2eta = 1. / (cyl_bessel_j(m+1, factors[i]) * b); i2eta *= i2eta;
        dcomplex i2etak0 = i2eta * ik0;
        for (size_t j = 0; j != N; ++j) {
            size_t jp = idxp(j);
            double k = factors[j] / b;
            RH(is, jp) = - i2etak0 * k * (k * (eps.Vmm(i,j) - eps.Vpp(i,j)) + eps.Dm(i,j) + eps.Dp(i,j));
            RH(ip, jp) = - i2etak0 * k * (k * (eps.Vmm(i,j) + eps.Vpp(i,j)) + eps.Dm(i,j) - eps.Dp(i,j));
        }
        RH(is, is)  = k0;
        RH(ip, ip) += k0;
    }

    for (size_t i = 0; i != N; ++i) {
        size_t is = idxs(i); size_t ip = idxp(i);
        double i2eta = 1. / (cyl_bessel_j(m+1, factors[i]) * b); i2eta *= i2eta;
        for (size_t j = 0; j != N; ++j) {
            size_t js = idxs(j); size_t jp = idxp(j);
            RE(is, js) = i2eta * k0 * (eps.Tmm(i,j) + eps.Tmp(i,j) + eps.Tpp(i,j) + eps.Tpm(i,j));
            RE(ip, js) = i2eta * k0 * (eps.Tmm(i,j) + eps.Tmp(i,j) - eps.Tpp(i,j) - eps.Tpm(i,j));
            RE(is, jp) = i2eta * k0 * (eps.Tmm(i,j) - eps.Tmp(i,j) - eps.Tpp(i,j) + eps.Tpm(i,j));
            RE(ip, jp) = i2eta * k0 * (eps.Tmm(i,j) - eps.Tmp(i,j) + eps.Tpp(i,j) - eps.Tpm(i,j));
        }
        double g = factors[i] / b;
        RE(is, is) -= ik0 * g * g;
    }
}

}}} // # namespace plask::solvers::slab
