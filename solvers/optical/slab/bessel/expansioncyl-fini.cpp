#include "expansioncyl-fini.h"
#include "solvercyl.h"
#include "zeros-data.h"

#include "../gauss_legendre.h"

#include <boost/math/special_functions/bessel.hpp>
using boost::math::cyl_bessel_j;
using boost::math::cyl_bessel_j_zero;
using boost::math::legendre_p;

#define SOLVER static_cast<BesselSolverCyl*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionBesselFini::ExpansionBesselFini(BesselSolverCyl* solver): ExpansionBessel(solver)
{
}


void ExpansionBesselFini::computeBesselZeros()
{
    size_t N = SOLVER->size;
    size_t n = 0;
    kpts.resize(N);
    if (m < 5) {
        n = min(N, size_t(100));
        std::copy_n(bessel_zeros[m], n, kpts.begin());
    }
    if (n < N) {
        SOLVER->writelog(LOG_DEBUG, "Computing Bessel function J_({:d}) zeros {:d} to {:d}", m, n+1, N);
PLASK_NO_CONVERSION_WARNING_BEGIN
        cyl_bessel_j_zero(double(m), n+1, N-n, kpts.begin()+n);
PLASK_NO_WARNING_END
    }
    // #ifndef NDEBUG
    //     for (size_t i = 0; i != N; ++i) {
    //         auto report = [i,m,this]()->bool{
    //             std::cerr << "J(" << m << ", " << kpts[i] << ") = " << cyl_bessel_j(m, kpts[i]) << "\n";
    //             return false;
    //         };
    //         assert(is_zero(cyl_bessel_j(m, kpts[i]), 1e-9) || report());
    //     }
    // #endif
}


void ExpansionBesselFini::init2()
{
    SOLVER->writelog(LOG_DETAIL, "Preparing Bessel functions for m = {}", m);
    computeBesselZeros();

    init3();

    auto raxis = mesh->axis[0];

    // Compute integrals for permeability
    size_t N = SOLVER->size;
    mu_integrals.reset(N);

    if (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) {
        double ib = 1. / rbounds[rbounds.size()-1];
        size_t pmlseg = segments.size()-1;

        // Compute analytically for constant section using first and second Lommel's integrals
        double r0 = rbounds[pmlseg];
        double rr = r0*r0;
        for (std::size_t i = 0; i < N; ++i) {
            double g = kpts[i] * ib; double gr = g*r0; double gg = g*g;
            double Jmg = cyl_bessel_j(m-1, gr), Jpg = cyl_bessel_j(m+1, gr), Jg = cyl_bessel_j(m, gr),
                   Jm2g = cyl_bessel_j(m-2, gr), Jp2g = cyl_bessel_j(m+2, gr);
            mu_integrals.Vmm(i,i) = mu_integrals.Tmm(i,i) = 0.5 * rr * (Jmg*Jmg - Jg*Jm2g);
            mu_integrals.Vpp(i,i) = mu_integrals.Tpp(i,i) = 0.5 * rr * (Jpg*Jpg - Jg*Jp2g);
            mu_integrals.Tmp(i,i) = mu_integrals.Tpm(i,i) = mu_integrals.Dm(i,i) = mu_integrals.Dp(i,i) = 0.;
            for (std::size_t j = i+1; j < N; ++j) {
                double k = kpts[j] * ib; double kr = k*r0; double kk = k*k;
                double Jmk = cyl_bessel_j(m-1, kr), Jpk = cyl_bessel_j(m+1, kr), Jk = cyl_bessel_j(m, kr);
                    mu_integrals.Vmm(i,j) = mu_integrals.Tmm(i,j) = r0 / (gg - kk) * (g * Jg * Jmk - k * Jk * Jmg);
                    mu_integrals.Vpp(i,j) = mu_integrals.Tpp(i,j) = r0 / (gg - kk) * (k * Jk * Jpg - g * Jg * Jpk);
                    mu_integrals.Tmp(i,j) = mu_integrals.Tpm(i,j) = mu_integrals.Dm(i,j) = mu_integrals.Dp(i,j) = 0.;
            }
        }

        for (size_t ri = raxis->size()-segments[pmlseg].weights.size(), wi = 0, nr = raxis->size(); ri != nr; ++ri, ++wi) {
            double r = raxis->at(ri);
            double w = segments[pmlseg].weights[wi] * segments[pmlseg].D;

            dcomplex mu = 1. + (SOLVER->pml.factor - 1.) * pow((r-r0)/SOLVER->pml.size, SOLVER->pml.order);
            dcomplex imu = 1. / mu;
            dcomplex mua = 0.5 * (imu + mu), dmu = 0.5 * (imu - mu);
            dcomplex imu1 = imu - 1.;

            imu *= w; mua *= w; dmu *= w; imu1 *= w;

            for (std::size_t i = 0; i < N; ++i) {
                double g = kpts[i] * ib; double gr = g*r;
                double Jmg = cyl_bessel_j(m-1, gr), Jpg = cyl_bessel_j(m+1, gr), Jg = cyl_bessel_j(m, gr),
                       Jm2g = cyl_bessel_j(m-2, gr), Jp2g = cyl_bessel_j(m+2, gr);
                for (std::size_t j = i; j < N; ++j) {
                    double k = kpts[j] * ib; double kr = k*r;
                    double Jmk = cyl_bessel_j(m-1, kr), Jpk = cyl_bessel_j(m+1, kr), Jk = cyl_bessel_j(m, kr);
                    mu_integrals.Vmm(i,j) += r * Jmg * imu * Jmk;
                    mu_integrals.Vpp(i,j) += r * Jpg * imu * Jpk;
                    mu_integrals.Tmm(i,j) += r * Jmg * mua * Jmk;
                    mu_integrals.Tpp(i,j) += r * Jpg * mua * Jpk;
                    mu_integrals.Tmp(i,j) += r * Jmg * dmu * Jpk;
                    mu_integrals.Tpm(i,j) += r * Jpg * dmu * Jmk;
                    mu_integrals.Dm(i,j) -= imu1 * (0.5*r*(g*(Jm2g-Jg)*Jk + k*Jmg*(Jmk-Jpk)) + Jmg*Jk);
                    mu_integrals.Dp(i,j) -= imu1 * (0.5*r*(g*(Jg-Jp2g)*Jk + k*Jpg*(Jmk-Jpk)) + Jpg*Jk);
                    if (j != i) {
                        double Jm2k = cyl_bessel_j(m-2, kr), Jp2k = cyl_bessel_j(m+2, kr);
                        mu_integrals.Dm(j,i) -= imu1 * (0.5*r*(k*(Jm2k-Jk)*Jg + g*Jmk*(Jmg-Jpg)) + Jmk*Jg);
                        mu_integrals.Dp(j,i) -= imu1 * (0.5*r*(k*(Jk-Jp2k)*Jg + g*Jpk*(Jmg-Jpg)) + Jpk*Jg);
                    }
                }
            }
        }
    } else {
        mu_integrals.zero();
        for (std::size_t i = 0; i < N; ++i) {
            double eta = cyl_bessel_j(m+1, kpts[i]) * rbounds[rbounds.size()-1]; eta = 0.5 * eta*eta;;
            mu_integrals.Vmm(i,i) = mu_integrals.Vpp(i,i) = mu_integrals.Tmm(i,i) = mu_integrals.Tpp(i,i) = eta;
        }
    }
}


void ExpansionBesselFini::reset()
{
    mu_integrals.reset();
    ExpansionBessel::reset();
}


void ExpansionBesselFini::layerIntegrals(size_t layer, double lam, double glam)
{
    integrateLayer(layer, lam, glam, true);
}


void ExpansionBesselFini::getMatrices(size_t layer, cmatrix& RE, cmatrix& RH)
{
    size_t N = SOLVER->size;
    dcomplex ik0 = 1. / k0;
    double b = rbounds[rbounds.size()-1];

    const Integrals& eps = layers_integrals[layer];
    #define mu mu_integrals

    for (size_t i = 0; i != N; ++i) {
        size_t is = idxs(i); size_t ip = idxp(i);
        double i2eta = 1. / (cyl_bessel_j(m+1, kpts[i]) * b); i2eta *= i2eta;
        for (size_t j = 0; j != N; ++j) {
            size_t js = idxs(j); size_t jp = idxp(j);
            double k = kpts[j] / b;
            RH(is, js) = i2eta *  k0 * (mu.Tmm(i,j) - mu.Tmp(i,j) + mu.Tpp(i,j) - mu.Tpm(i,j));
            RH(ip, js) = i2eta *  k0 * (mu.Tmm(i,j) - mu.Tmp(i,j) - mu.Tpp(i,j) + mu.Tpm(i,j));
            RH(is, jp) = i2eta * (k0 * (mu.Tmm(i,j) + mu.Tmp(i,j) - mu.Tpp(i,j) - mu.Tpm(i,j))
                                - ik0 * k * (k * (eps.Vmm(i,j) - eps.Vpp(i,j)) + eps.Dm(i,j) + eps.Dp(i,j)));
            RH(ip, jp) = i2eta * (k0 * (mu.Tmm(i,j) + mu.Tmp(i,j) + mu.Tpp(i,j) + mu.Tpm(i,j))
                                  - ik0 * k * (k * (eps.Vmm(i,j) + eps.Vpp(i,j)) + eps.Dm(i,j) - eps.Dp(i,j)));
        }
    }

    for (size_t i = 0; i != N; ++i) {
        size_t is = idxs(i); size_t ip = idxp(i);
        double i2eta = 1. / (cyl_bessel_j(m+1, kpts[i]) * b); i2eta *= i2eta;
        for (size_t j = 0; j != N; ++j) {
            size_t js = idxs(j); size_t jp = idxp(j);
            double k = kpts[j] / b;
            RE(is, js) = i2eta * (k0 * (eps.Tmm(i,j) + eps.Tmp(i,j) + eps.Tpp(i,j) + eps.Tpm(i,j))
                                - ik0 * k * (k * (mu.Vmm(i,j) + mu.Vpp(i,j)) + mu.Dm(i,j) - mu.Dp(i,j)));
            RE(ip, js) = i2eta * (k0 * (eps.Tmm(i,j) + eps.Tmp(i,j) - eps.Tpp(i,j) - eps.Tpm(i,j))
                                - ik0 * k * (k * (mu.Vmm(i,j) - mu.Vpp(i,j)) + mu.Dm(i,j) + mu.Dp(i,j)));
            RE(is, jp) = i2eta *  k0 * (eps.Tmm(i,j) - eps.Tmp(i,j) - eps.Tpp(i,j) + eps.Tpm(i,j));
            RE(ip, jp) = i2eta *  k0 * (eps.Tmm(i,j) - eps.Tmp(i,j) + eps.Tpp(i,j) - eps.Tpm(i,j));
        }
    }
    #undef mu
}


#ifndef NDEBUG
cmatrix ExpansionBesselFini::muVmm() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Vmm(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muVpp() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Vpp(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muTmm() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Tmm(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muTpp() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Tpp(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muTmp() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Tmp(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muTpm() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Tpm(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muDm() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Dm(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muDp() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Dp(i,j);
    return result;
}
#endif

}}} // # namespace plask::optical::slab
