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

    double k0 = isnan(lam0)? this->k0.real() : 2e3*M_PI / lam0;
    double kmax = SOLVER->kmax * k0;

    size_t N = SOLVER->size;
    double R = rbounds[rbounds.size()-1];
    double ib = 1. / R;
    double kdlt;

    switch (SOLVER->kmethod) {
        case BesselSolverCyl::WAVEVECTORS_UNIFORM:
            SOLVER->writelog(LOG_DETAIL, "Using uniform wavevectors");
            if (isnan(k0)) throw BadInput(SOLVER->getId(), "No wavelength given: specify 'lam' or 'lam0'");
            kpts.resize(N);
            kdlt = SOLVER->kscale * kmax / N;
            kdelts.reset(N, kdlt);
            kdlt *= R;
            for (size_t i = 0; i != N; ++i) kpts[i] = (0.5 + double(i) * kdlt);
            break;
        case BesselSolverCyl::WAVEVECTORS_LAGUERRE:
            SOLVER->writelog(LOG_DETAIL, "Using Laguerre wavevectors");
            gaussLaguerre(N, kpts, kdelts, 1. / (SOLVER->kscale));
            kdelts *= ib;
            break;
        case BesselSolverCyl::WAVEVECTORS_MANUAL:
            SOLVER->writelog(LOG_DETAIL, "Using manual wavevectors");
            kpts.resize(N);
            kdelts.reset(N);
            if (!SOLVER->kweights) {
                if (SOLVER->klist.size() != N+1)
                    throw BadInput(SOLVER->getId(), "If no weights are given, number of manually specified wavevectors must be {}",
                                   N+1);
                for (size_t i = 0; i != N; ++i) {
                    kpts[i] = 0.5 * (SOLVER->klist[i] + SOLVER->klist[i+1]) * R;
                    kdelts[i] = (SOLVER->klist[i+1] - SOLVER->klist[i]);
                }
            } else {
                if (SOLVER->klist.size() != N)
                    throw BadInput(SOLVER->getId(), "If weights are given, number of manually specified wavevectors must be {}",
                                   N);
                if (SOLVER->kweights->size() != N)
                    throw BadInput(SOLVER->getId(), "Number of manually specified wavevector weights must be {}", N+1);
                kpts = SOLVER->klist;
                for (double& k: kpts) k *= R;
                kdelts.reset(SOLVER->kweights->begin(), SOLVER->kweights->end());
            }
            break;
        case BesselSolverCyl::WAVEVECTORS_NONUNIFORM:
            SOLVER->writelog(LOG_DETAIL, "Using non-uniform wavevectors");
            if (isnan(k0)) throw BadInput(SOLVER->getId(), "No wavelength given: specify 'lam' or 'lam0'");
            kpts.resize(N);
            kdelts.reset(N);
            // Häyrynen, T., de Lasson, J.R., Gregersen, N., 2016.
            // Open-geometry Fourier modal method: modeling nanophotonic structures in infinite domains.
            // J. Opt. Soc. Am. A 33, 1298. https://doi.org/10.1364/josaa.33.001298
            int M1, M2, M3;
            M1 = M2 = (N+1) / 3;
            M3 = N - M1 - M2;
            if (M3 < 0) throw BadInput(SOLVER->getId(), "Too small expansion size");
            int i = 0;
            double k1 = 0.;
            for(int m = 1; m <= M1; ++m, ++i) {
                double theta = 0.5*M_PI * double(m) / M1;
                double k2 = k0 * sin(theta) * SOLVER->kscale;
                kdelts[i] = k2 - k1;
                kpts[i] = 0.5 * (k1 + k2) * R;
                k1 = k2;
            }
            double dlt1;
            for(int m = 1; m <= M2; ++m, ++i) {
                double theta = 0.5*M_PI * (1. + double(m) / M2);
                double k2 = k0 * (2 - sin(theta)) * SOLVER->kscale;
                kdelts[i] = dlt1 = k2 - k1;
                kpts[i] = 0.5 * (k1 + k2) * R;
                k1 = k2;
            }
            double km = k1;
            double dlt2 = (kmax * SOLVER->kscale - km - M3 * dlt1) / (M3 * (M3 + 1));
            if (dlt2 < 0)
                throw BadInput(SOLVER->getId(), "For non-uniform wavevectors kmax must be at least {}",
                               (km + M3 * dlt1) / SOLVER->kscale / k0);
            for(int m = 1; m <= M3; ++m, ++i) {
                double k2 = km + dlt1 * m + dlt2 * m * (m+1);
                kdelts[i] = k2 - k1;
                kpts[i] = 0.5 * (k1 + k2) * R;
                k1 = k2;
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
