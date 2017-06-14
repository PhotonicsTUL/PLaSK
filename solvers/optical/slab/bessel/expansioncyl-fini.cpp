#include "expansioncyl-fini.h"
#include "solvercyl.h"
#include "zeros-data.h"

#include "../gauss_legendre.h"

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/legendre.hpp>
using boost::math::cyl_bessel_j;
using boost::math::cyl_bessel_j_zero;
using boost::math::legendre_p;

#define SOLVER static_cast<BesselSolverCyl*>(solver)

namespace plask { namespace solvers { namespace slab {

ExpansionBesselFini::ExpansionBesselFini(BesselSolverCyl* solver): ExpansionBessel(solver)
{
}


void ExpansionBesselFini::computeBesselZeros()
{
    size_t N = SOLVER->size;
    size_t n = 0;
    factors.resize(N);
    if (m < 5) {
        n = min(N, size_t(100));
        std::copy_n(bessel_zeros[m], n, factors.begin());
    }
    if (n < N) {
        SOLVER->writelog(LOG_DEBUG, "Computing Bessel function J_({:d}) zeros {:d} to {:d}", m, n+1, N);
        cyl_bessel_j_zero(double(m), n+1, N-n, factors.begin()+n);
    }
    // #ifndef NDEBUG
    //     for (size_t i = 0; i != N; ++i) {
    //         auto report = [i,m,this]()->bool{
    //             std::cerr << "J(" << m << ", " << factors[i] << ") = " << cyl_bessel_j(m, factors[i]) << "\n";
    //             return false;
    //         };
    //         assert(is_zero(cyl_bessel_j(m, factors[i]), 1e-9) || report());
    //     }
    // #endif
}


void ExpansionBesselFini::init2()
{
    SOLVER->writelog(LOG_DETAIL, "Preparing Bessel functions for m = {}", m);
    computeBesselZeros();

    size_t nseg = rbounds.size() - 1;

    // Estimate necessary number of integration points
    double k = factors[factors.size()-1];

    double expected = cyl_bessel_j(m+1, k) * rbounds[rbounds.size()-1];
    expected = 0.5 * expected*expected;

    k /= rbounds[rbounds.size()-1];

    double max_error = SOLVER->integral_error * expected / nseg;
    double error = 0.;

    std::deque<std::vector<double>> abscissae_cache;
    std::deque<DataVector<double>> weights_cache;

    auto raxis = plask::make_shared<OrderedAxis>();
    OrderedAxis::WarningOff nowarn_raxis(raxis);

    double expcts = 0.;
    for (size_t i = 0; i < nseg; ++i) {
        double b = rbounds[i+1];

        // excpected value is the second Lommel's integral
        double expct = expcts;
        expcts = cyl_bessel_j(m, k*b); expcts = 0.5 * b*b * (expcts*expcts - cyl_bessel_j(m-1, k*b) * cyl_bessel_j(m+1, k*b));
        expct = expcts - expct;

        double err = 2 * max_error;
        std::vector<double> points;
        size_t j, n = 0;
        double sum;
        for (j = 0; err > max_error && n <= SOLVER->max_itegration_points; ++j) {
            n = 4 * (j+1) - 1;
            if (j == abscissae_cache.size()) {
                abscissae_cache.push_back(std::vector<double>());
                weights_cache.push_back(DataVector<double>());
                gaussData(n, abscissae_cache.back(), weights_cache.back());
            }
            assert(j < abscissae_cache.size());
            assert(j < weights_cache.size());
            const std::vector<double>& abscissae = abscissae_cache[j];
            points.clear(); points.reserve(abscissae.size());
            sum = 0.;
            for (size_t a = 0; a != abscissae.size(); ++a) {
                double r = segments[i].Z + segments[i].D * abscissae[a];
                double Jm = cyl_bessel_j(m, k*r);
                sum += weights_cache[j][a] * Jm*Jm*r;
                points.push_back(r);
            }
            sum *= segments[i].D;
            err = abs(sum - expct);
        }
        error += err;
        raxis->addOrderedPoints(points.begin(), points.end());
        segments[i].weights = weights_cache[j-1];
    }

    SOLVER->writelog(LOG_DETAIL, "Sampling structure in {:d} points (error: {:g}/{:g})", raxis->size(), error/expected, SOLVER->integral_error);

    // Compute integrals for permeability
    size_t N = SOLVER->size;
    mu_integrals.reset(N);

    if (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) {
        double ib = 1. / rbounds[rbounds.size()-1];
        size_t pmlseg = segments.size()-1;

        // Compute analytically for constant section using first and second Lommel's integrals
        double r0 = rbounds[pmlseg];
        double rr = r0*r0;
        for (int i = 0; i < N; ++i) {
            double g = factors[i] * ib; double gr = g*r0; double gg = g*g;
            double Jmg = cyl_bessel_j(m-1, gr), Jpg = cyl_bessel_j(m+1, gr), Jg = cyl_bessel_j(m, gr),
                   Jm2g = cyl_bessel_j(m-2, gr), Jp2g = cyl_bessel_j(m+2, gr);
            mu_integrals.Vmm(i,i) = mu_integrals.Tmm(i,i) = 0.5 * rr * (Jmg*Jmg - Jg*Jm2g);
            mu_integrals.Vpp(i,i) = mu_integrals.Tpp(i,i) = 0.5 * rr * (Jpg*Jpg - Jg*Jp2g);
            mu_integrals.Tmp(i,i) = mu_integrals.Tpm(i,i) = mu_integrals.Dm(i,i) = mu_integrals.Dp(i,i) = 0.;
            for (int j = i+1; j < N; ++j) {
                double k = factors[j] * ib; double kr = k*r0; double kk = k*k;
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

            for (int i = 0; i < N; ++i) {
                double g = factors[i] * ib; double gr = g*r;
                double Jmg = cyl_bessel_j(m-1, gr), Jpg = cyl_bessel_j(m+1, gr), Jg = cyl_bessel_j(m, gr),
                       Jm2g = cyl_bessel_j(m-2, gr), Jp2g = cyl_bessel_j(m+2, gr);
                for (int j = i; j < N; ++j) {
                    double k = factors[j] * ib; double kr = k*r;
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
        for (int i = 0; i < N; ++i) {
            double eta = cyl_bessel_j(m+1, factors[i]) * rbounds[rbounds.size()-1]; eta = 0.5 * eta*eta;;
            mu_integrals.Vmm(i,i) = mu_integrals.Vpp(i,i) = mu_integrals.Tmm(i,i) = mu_integrals.Tpp(i,i) = eta;
        }
    }

    // Allocate memory for integrals
    size_t nlayers = solver->lcount;
    layers_integrals.resize(nlayers);
    iepsilons.resize(nlayers);
    for (size_t l = 0, nr = raxis->size(); l != nlayers; ++l)
        iepsilons[l].reset(nr);

    mesh = plask::make_shared<RectangularMesh<2>>(raxis, solver->verts, RectangularMesh<2>::ORDER_01);

    m_changed = false;
}


void ExpansionBesselFini::reset()
{
    layers_integrals.clear();
    ExpansionBessel::reset();
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
        double i2eta = 1. / (cyl_bessel_j(m+1, factors[i]) * b); i2eta *= i2eta;
        for (size_t j = 0; j != N; ++j) {
            size_t js = idxs(j); size_t jp = idxp(j);
            double k = factors[j] / b;
            RH(is, js) = i2eta *  k0 * (mu.Tmm(i,j) - mu.Tmp(i,j) + mu.Tpp(i,j) - mu.Tpm(i,j));
            RH(ip, js) = i2eta *  k0 * (mu.Tmm(i,j) - mu.Tmp(i,j) - mu.Tpp(i,j) + mu.Tpm(i,j));
            RH(is, jp) = i2eta * (k0 * (mu.Tmm(i,j) + mu.Tmp(i,j) - mu.Tpp(i,j) - mu.Tpm(i,j))
                                - ik0 * k* (k * (eps.Vmm(i,j) - eps.Vpp(i,j)) + eps.Dm(i,j) + eps.Dp(i,j)));
            RH(ip, jp) = i2eta * (k0 * (mu.Tmm(i,j) + mu.Tmp(i,j) + mu.Tpp(i,j) + mu.Tpm(i,j))
                                  - ik0 * k* (k * (eps.Vmm(i,j) + eps.Vpp(i,j)) + eps.Dm(i,j) - eps.Dp(i,j)));
        }
    }

    for (size_t i = 0; i != N; ++i) {
        size_t is = idxs(i); size_t ip = idxp(i);
        double i2eta = 1. / (cyl_bessel_j(m+1, factors[i]) * b); i2eta *= i2eta;
        for (size_t j = 0; j != N; ++j) {
            size_t js = idxs(j); size_t jp = idxp(j);
            double k = factors[j] / b;
            RE(is, js) = i2eta * (k0 * (eps.Tmm(i,j) + eps.Tmp(i,j) + eps.Tpp(i,j) + eps.Tpm(i,j))
                                - ik0 * k* (k * (mu.Vmm(i,j) + mu.Vpp(i,j)) + mu.Dm(i,j) - mu.Dp(i,j)));
            RE(ip, js) = i2eta * (k0 * (eps.Tmm(i,j) + eps.Tmp(i,j) - eps.Tpp(i,j) - eps.Tpm(i,j))
                                - ik0 * k* (k * (mu.Vmm(i,j) - mu.Vpp(i,j)) + mu.Dm(i,j) + mu.Dp(i,j)));
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

}}} // # namespace plask::solvers::slab