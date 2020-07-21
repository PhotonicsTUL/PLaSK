#include "expansioncyl-infini.h"
#include "solvercyl.h"
#include "../bessel/zeros-data.h"

#include "../gauss_legendre.h"
#include "../gauss_laguerre.h"

#include <boost/math/special_functions/bessel.hpp>
using boost::math::cyl_bessel_j;

#define SOLVER static_cast<OldBesselSolverCyl*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionOldBesselInfini::ExpansionOldBesselInfini(OldBesselSolverCyl* solver): ExpansionOldBessel(solver)
{
}


void ExpansionOldBesselInfini::init2()
{
    SOLVER->writelog(LOG_DETAIL, "Preparing Bessel functions for m = {}", m);

    if (SOLVER->geometry->getEdge(Geometry::DIRECTION_TRAN, true).type() != edge::Strategy::DEFAULT &&
        SOLVER->geometry->getEdge(Geometry::DIRECTION_TRAN, true).type() != edge::Strategy::SIMPLE &&
        SOLVER->geometry->getEdge(Geometry::DIRECTION_TRAN, true).type() != edge::Strategy::EXTEND)
        throw BadInput(solver->getId(), "Outer geometry edge must be 'extend' or a simple material");

    eps0.resize(solver->lcount);

    size_t N = SOLVER->size;
    double ib = 1. / rbounds[rbounds.size()-1];

    switch (SOLVER->kmethod) {
        case OldBesselSolverCyl::WAVEVECTORS_UNIFORM:
            kpts.resize(N);
            kdelts.reset(N, SOLVER->kscale * ib);
            for (size_t i = 0; i != N; ++i) kpts[i] = (0.5 + double(i)) * SOLVER->kscale;
            break;
        // case OldBesselSolverCyl::WAVEVECTORS_LEGENDRE:
        //     gaussLegendre(N, kpts, kdelts);
        //     for (double& k: kpts) k = 0.5 * N * SOLVER->kscale * (1. + k);
        //     kdelts *= 0.5 * N * SOLVER->kscale * ib;
        //     break;
        case OldBesselSolverCyl::WAVEVECTORS_LAGUERRE:
            gaussLaguerre(N, kpts, kdelts, 1. / (SOLVER->kscale));
            kdelts *= ib;
            break;
        case OldBesselSolverCyl::WAVEVECTORS_MANUAL:
            if (SOLVER->klist.size() != N+1)
                throw BadInput(SOLVER->getId(), "Number of manually specified wavevectors must be {}", N+1);
            kpts.resize(N);
            kdelts.reset(N);
            for (size_t i = 0; i != N; ++i) {
                kpts[i] = 0.5 * (SOLVER->klist[i] + SOLVER->klist[i+1]);
                kdelts[i] = ib * (SOLVER->klist[i+1] - SOLVER->klist[i]);
            }
            break;
    }

    init3();
}


void ExpansionOldBesselInfini::reset()
{
    eps0.clear();
    ExpansionOldBessel::reset();
}


void ExpansionOldBesselInfini::layerIntegrals(size_t layer, double lam, double glam)
{
    eps0[layer] = integrateLayer(layer, lam, glam, false);
}


void ExpansionOldBesselInfini::getMatrices(size_t layer, cmatrix& RE, cmatrix& RH)
{
    assert(initialized);
    if (isnan(k0)) throw BadInput(SOLVER->getId(), "Wavelength or k0 not set");
    if (isinf(k0.real())) throw BadInput(SOLVER->getId(), "Wavelength must not be 0");

    size_t N = SOLVER->size;
    dcomplex ik0 = 1. / k0;
    double ib = 1. / rbounds[rbounds.size()-1];

    const Integrals& eps = layers_integrals[layer];

    std::fill(RH.begin(), RH.end(), 0.);
    for (size_t i = 0; i != N; ++i) {
        size_t is = idxs(i); size_t ip = idxp(i);
        double g = kpts[i] * ib;
        double dg = kdelts[i];
        dcomplex f = 0.5 * g * dg * ik0;
        for (size_t j = 0; j != N; ++j) {
            size_t jp = idxp(j);
            double k = kpts[j] * ib;
            dcomplex fk = f * k;
            RH(is, jp) = - fk * (k * (eps.Vmm(i,j) - eps.Vpp(i,j)) + eps.Dm(i,j) + eps.Dp(i,j));
            RH(ip, jp) = - fk * (k * (eps.Vmm(i,j) + eps.Vpp(i,j)) + eps.Dm(i,j) - eps.Dp(i,j));
        }
        RH(is, is)  = k0;
        RH(ip, ip) += k0 - g*g * ik0 * eps0[layer].second;
    }

    for (size_t i = 0; i != N; ++i) {
        size_t is = idxs(i); size_t ip = idxp(i);
        double g = kpts[i] * ib;
        double dg = kdelts[i];
        dcomplex f = 0.5 * g * dg * k0;
        for (size_t j = 0; j != N; ++j) {
            size_t js = idxs(j); size_t jp = idxp(j);
            RE(ip, js) = f * (eps.Tmm(i,j) + eps.Tmp(i,j) - eps.Tpp(i,j) - eps.Tpm(i,j));
            RE(is, js) = f * (eps.Tmm(i,j) + eps.Tmp(i,j) + eps.Tpp(i,j) + eps.Tpm(i,j));
            RE(ip, jp) = f * (eps.Tmm(i,j) - eps.Tmp(i,j) + eps.Tpp(i,j) - eps.Tpm(i,j));
            RE(is, jp) = f * (eps.Tmm(i,j) - eps.Tmp(i,j) - eps.Tpp(i,j) + eps.Tpm(i,j));
        }
        dcomplex k0eps = k0 * eps0[layer].first;
        RE(ip, ip) += k0eps;
        RE(is, is) += k0eps - g*g * ik0;
    }
}


double ExpansionOldBesselInfini::integratePoyntingVert(const cvector& /*E*/, const cvector& /*H*/)
{
    return 1.;
}

double ExpansionOldBesselInfini::integrateField(WhichField /*field*/, size_t /*l*/,
                                             const cvector& /*E*/, const cvector& /*H*/)
{
    throw NotImplemented("BesselCyl", "field integration for infinite expansion");
}



}}} // # namespace plask::optical::slab
