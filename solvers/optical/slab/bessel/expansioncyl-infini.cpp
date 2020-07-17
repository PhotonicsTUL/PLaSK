#include "expansioncyl-infini.h"
#include "solvercyl.h"
#include "zeros-data.h"

#include "../gauss_legendre.h"
#include "../gauss_laguerre.h"

#include <boost/math/special_functions/bessel.hpp>
using boost::math::cyl_bessel_j;

#define SOLVER static_cast<BesselSolverCyl*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionBesselInfini::ExpansionBesselInfini(BesselSolverCyl* solver): ExpansionBessel(solver)
{
}


void ExpansionBesselInfini::init2()
{
    SOLVER->writelog(LOG_DETAIL, "Preparing Bessel functions for m = {}", m);

    if (SOLVER->geometry->getEdge(Geometry::DIRECTION_TRAN, true).type() != edge::Strategy::DEFAULT &&
        SOLVER->geometry->getEdge(Geometry::DIRECTION_TRAN, true).type() != edge::Strategy::SIMPLE &&
        SOLVER->geometry->getEdge(Geometry::DIRECTION_TRAN, true).type() != edge::Strategy::EXTEND)
        throw BadInput(solver->getId(), "Outer geometry edge must be 'extend' or a simple material");

    size_t N = SOLVER->size;
    double ib = 1. / rbounds[rbounds.size()-1];

    switch (SOLVER->kmethod) {
        case BesselSolverCyl::WAVEVECTORS_UNIFORM:
            kpts.resize(N);
            kdelts.reset(N, SOLVER->kscale * ib);
            for (size_t i = 0; i != N; ++i) kpts[i] = (0.5 + double(i)) * SOLVER->kscale;
            break;
        // case BesselSolverCyl::WAVEVECTORS_LEGENDRE:
        //     gaussLegendre(N, kpts, kdelts);
        //     for (double& k: kpts) k = 0.5 * N * SOLVER->kscale * (1. + k);
        //     kdelts *= 0.5 * N * SOLVER->kscale * ib;
        //     break;
        case BesselSolverCyl::WAVEVECTORS_LAGUERRE:
            gaussLaguerre(N, kpts, kdelts, 1. / (SOLVER->kscale));
            kdelts *= ib;
            break;
        case BesselSolverCyl::WAVEVECTORS_MANUAL:
            kpts.resize(N);
            kdelts.reset(N);
            if (!SOLVER->kweights) {
                if (SOLVER->klist.size() != N+1)
                    throw BadInput(SOLVER->getId(), "If no weights are given number of manually specified wavevectors must be {}",
                                   N+1);
                for (size_t i = 0; i != N; ++i) {
                    kpts[i] = 0.5 * (SOLVER->klist[i] + SOLVER->klist[i+1]);
                    kdelts[i] = ib * (SOLVER->klist[i+1] - SOLVER->klist[i]);
                }
            } else {
                if (SOLVER->klist.size() != N)
                    throw BadInput(SOLVER->getId(), "If weights are given number of manually specified wavevectors must be {}",
                                   N);
                if (SOLVER->kweights->size() != N)
                    throw BadInput(SOLVER->getId(), "Number of manually specified wavevector weights must be {}", N+1);
                kpts = SOLVER->klist;
                kdelts.reset(SOLVER->kweights->begin(), SOLVER->kweights->end());
                kdelts *= ib;
            }
            break;
    }

    init3();
}


void ExpansionBesselInfini::getMatrices(size_t layer, cmatrix& RE, cmatrix& RH)
{
    assert(initialized);
    if (isnan(k0)) throw BadInput(SOLVER->getId(), "Wavelength or k0 not set");
    if (isinf(k0.real())) throw BadInput(SOLVER->getId(), "Wavelength must not be 0");

    size_t N = SOLVER->size;
    dcomplex ik0 = 1. / k0;
    double ib = 1. / rbounds[rbounds.size()-1];

    const Integrals& eps = layers_integrals[layer];

    for (size_t j = 0; j != N; ++j) {
        size_t js = idxs(j), jp = idxp(j);
        // double k = kpts[j] * ib;
        for (size_t i = 0; i != N; ++i) {
            size_t is = idxs(i), ip = idxp(i);
            double g = kpts[i] * ib;
            dcomplex gVk = 0.5 * ik0 * g * eps.V_k(i,j);
            RH(is,jp) =  gVk;
            RH(is,js) =  gVk;
            RH(ip,jp) = -gVk;
            RH(ip,js) = -gVk;
        }
        RH(js,js) -= k0;
        RH(jp,jp) += k0;
    }

    for (size_t j = 0; j != N; ++j) {
        size_t js = idxs(j), jp = idxp(j);
        for (size_t i = 0; i != N; ++i) {
            size_t is = idxs(i), ip = idxp(i);
            // double g = kpts[i] * ib;
            RE(ip,js) =  0.5 * k0 * eps.Tps(i,j);
            RE(ip,jp) =  0.5 * k0 * eps.Tpp(i,j);
            RE(is,js) = -0.5 * k0 * eps.Tss(i,j);
            RE(is,jp) = -0.5 * k0 * eps.Tsp(i,j);
        }
        double k = kpts[j] * ib;
        dcomplex ik0k2 = 0.5 * ik0 * k*k;
        RE(jp,js) -= ik0k2;
        RE(jp,jp) -= ik0k2;
        RE(js,js) += ik0k2;
        RE(js,jp) += ik0k2;
    }
}

}}} // # namespace plask::optical::slab
