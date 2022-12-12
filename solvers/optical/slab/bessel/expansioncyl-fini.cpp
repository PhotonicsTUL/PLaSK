/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "expansioncyl-fini.hpp"
#include "solvercyl.hpp"
#include "zeros-data.hpp"

#include "../gauss_legendre.hpp"

#include "besselj.hpp"

#define SOLVER static_cast<BesselSolverCyl*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionBesselFini::ExpansionBesselFini(BesselSolverCyl* solver) : ExpansionBessel(solver) {}

void ExpansionBesselFini::computeBesselZeros() {
    size_t N = SOLVER->size;
    size_t n = 0;
    kpts.resize(N);
    if (m < 5) {
        n = min(N, size_t(100));
        std::copy_n(bessel_zeros[m], n, kpts.begin());
    }
    if (n < N) {
        SOLVER->writelog(LOG_DEBUG, "Computing Bessel function J_({:d}) zeros {:d} to {:d}", m, n + 1, N);
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

void ExpansionBesselFini::init2() {
    SOLVER->writelog(LOG_DETAIL, "Preparing Bessel functions for m = {}", m);
    computeBesselZeros();

    init3();

    // Compute integrals for permeability
    auto raxis = mesh->tran();

    size_t nr = raxis->size(), N = SOLVER->size;

    double ib = 1. / rbounds[rbounds.size() - 1];

    // Compute mu integrals
    if (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) {

        SOLVER->writelog(LOG_DETAIL, "Computing permeability integrals with {} rule", SOLVER->ruleName());

        size_t pmlseg = segments.size() - 1;

        size_t pmli = raxis->size() - segments[pmlseg].weights.size();
        double pmlr = rbounds[pmlseg];

        aligned_unique_ptr<dcomplex> mu_data(aligned_malloc<dcomplex>(nr));
        aligned_unique_ptr<dcomplex> imu_data(aligned_malloc<dcomplex>(nr));

        for (size_t ri = 0, wi = 0, seg = 0, nw = segments[0].weights.size(); ri != nr; ++ri, ++wi) {
            if (wi == nw) {
                nw = segments[++seg].weights.size();
                wi = 0;
            }
            double r = raxis->at(ri);
            double w = segments[seg].weights[wi] * segments[seg].D;

            dcomplex mu = 1., imu = 1.;
            if (ri >= pmli) {
                mu = 1. + (SOLVER->pml.factor - 1.) * pow((r - pmlr) / SOLVER->pml.size, SOLVER->pml.order);
                imu = 1. / mu;
            }

            mu_data.get()[ri] = mu * w;
            imu_data.get()[ri] = imu * w;
        }

        switch (SOLVER->rule) {
          case BesselSolverCyl::RULE_COMBINED_1:
          case BesselSolverCyl::RULE_COMBINED_2:
            integrateParams(mu_integrals, mu_data.get(), mu_data.get(), mu_data.get(), 1., 1., 1.); break;
          case BesselSolverCyl::RULE_DIRECT:
            integrateParams(mu_integrals, mu_data.get(), imu_data.get(), mu_data.get(), 1., 1., 1.); break;
          case BesselSolverCyl::RULE_OLD:
            integrateParams(mu_integrals, mu_data.get(), imu_data.get(), imu_data.get(), 1., 1., 1.); break;
        }

    } else {

        mu_integrals.reset(N);
        zero_matrix(mu_integrals.V_k);
        zero_matrix(mu_integrals.Tss);
        zero_matrix(mu_integrals.Tsp);
        zero_matrix(mu_integrals.Tps);
        zero_matrix(mu_integrals.Tpp);
        for (size_t i = 0; i < N; ++i) {
            double k = kpts[i] * ib;
            mu_integrals.V_k(i,i) = k;
            mu_integrals.Tss(i,i) = mu_integrals.Tpp(i,i) = 2.;
        }
    }
}

void ExpansionBesselFini::reset() {
    mu_integrals.reset();
    ExpansionBessel::reset();
}

void ExpansionBesselFini::getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) {
    assert(initialized);
    if (isnan(k0)) throw BadInput(SOLVER->getId(), "Wavelength or k0 not set");
    if (isinf(k0.real())) throw BadInput(SOLVER->getId(), "Wavelength must not be 0");

    size_t N = SOLVER->size;
    dcomplex ik0 = 1. / k0;
    double ib = 1. / rbounds[rbounds.size() - 1];

    const Integrals& eps = layers_integrals[layer];
    #define mu mu_integrals

    for (size_t j = 0; j != N; ++j) {
        size_t js = idxs(j), jp = idxp(j);
        // double k = kpts[j] * ib;
        for (size_t i = 0; i != N; ++i) {
            size_t is = idxs(i), ip = idxp(i);
            double g = kpts[i] * ib;
            dcomplex gVk = ik0 * g * eps.V_k(i,j);
            RH(is,jp) = 0.5 * (  gVk - k0 * mu.Tsp(i,j) );
            RH(is,js) = 0.5 * (  gVk - k0 * mu.Tss(i,j) );
            RH(ip,jp) = 0.5 * ( -gVk + k0 * mu.Tpp(i,j) );
            RH(ip,js) = 0.5 * ( -gVk + k0 * mu.Tps(i,j) );
        }
    }

    for (size_t j = 0; j != N; ++j) {
        size_t js = idxs(j), jp = idxp(j);
        // double k = kpts[j] * ib;
        for (size_t i = 0; i != N; ++i) {
            size_t is = idxs(i), ip = idxp(i);
            double g = kpts[i] * ib;
            dcomplex gVk = ik0 * g * mu.V_k(i,j);
            RE(ip,js) = 0.5 * ( -gVk + k0 * eps.Tps(i,j) );
            RE(ip,jp) = 0.5 * ( -gVk + k0 * eps.Tpp(i,j) );
            RE(is,js) = 0.5 * (  gVk - k0 * eps.Tss(i,j) );
            RE(is,jp) = 0.5 * (  gVk - k0 * eps.Tsp(i,j) );
        }
    }
    #undef mu
}

double ExpansionBesselFini::fieldFactor(size_t i) {
    double eta = cyl_bessel_j(m + 1, kpts[i]) * rbounds[rbounds.size() - 1];
    return 0.5 * eta * eta;
}


#ifndef NDEBUG
cmatrix ExpansionBesselFini::muV_k() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.V_k(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muTss() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Tss(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muTsp() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Tsp(i,j);
    return result;
}
cmatrix ExpansionBesselFini::muTps() {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = mu_integrals.Tps(i,j);
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
#endif

void ExpansionBesselFini::integrateParams(Integrals& integrals,
                                          const dcomplex* datap, const dcomplex* datar, const dcomplex* dataz,
                                          dcomplex datap0, dcomplex datar0, dcomplex dataz0) {
    auto raxis = mesh->tran();

    size_t nr = raxis->size(), N = SOLVER->size;
    double R = rbounds[rbounds.size()-1];
    double ib = 1. / R;
    double* factors; // scale factors for making matrices orthonormal

    integrals.reset(N);

    TempMatrix temp = getTempMatrix();
    aligned_unique_ptr<double> _tmp;

    cmatrix JmJp, JpJm;
    if (N < 2) {
        _tmp.reset(aligned_malloc<double>(4*N));
        factors = _tmp.get();
    } else if (SOLVER->rule == BesselSolverCyl::RULE_COMBINED_1) {
        factors = reinterpret_cast<double*>(integrals.V_k.data());
    } else if (SOLVER->rule == BesselSolverCyl::RULE_COMBINED_2) {
        _tmp.reset(aligned_malloc<double>(3*N + 4*N*N));
        JmJp.reset(N, N, integrals.V_k.data());
        JpJm.reset(N, N, reinterpret_cast<dcomplex*>(_tmp.get() + 3*N));
        factors = _tmp.get();
    } else {
        factors = reinterpret_cast<double*>(temp.data());
    }
    for (size_t i = 0; i < N; ++i) {
        double fact = R * cyl_bessel_j(m+1, kpts[i]);
        factors[i] = 2. / (fact * fact);
    }
    double* Jm = factors + N;
    double* Jp = factors + 2*N;

    if (SOLVER->rule == BesselSolverCyl::RULE_OLD) {

        double* J  = factors + 3*N;

        zero_matrix(integrals.V_k);
        zero_matrix(integrals.Tss);
        zero_matrix(integrals.Tsp);
        zero_matrix(integrals.Tps);
        zero_matrix(integrals.Tpp);

        for (size_t ri = 0; ri != nr; ++ri) {
            double r = raxis->at(ri);
            dcomplex repst = r * (datar[ri] + datap[ri]), repsd = r * (datar[ri] - datap[ri]);
            const dcomplex riepsz = r * dataz[ri];

            for (size_t i = 0; i < N; ++i) {
                double kr = kpts[i] * ib * r;
                Jm[i] = cyl_bessel_j(m-1, kr);
                J[i]  = cyl_bessel_j(m, kr);
                Jp[i] = cyl_bessel_j(m+1, kr);
            }
            for (size_t j = 0; j < N; ++j) {
                double k = kpts[j] * ib;
                double Jk = J[j], Jmk = Jm[j], Jpk = Jp[j];
                for (size_t i = 0; i < N; ++i) {
                    double Jg = J[i], Jmg = Jm[i], Jpg = Jp[i];
                    integrals.V_k(i,j) += Jg * riepsz * Jk * k * factors[i];
                    integrals.Tss(i,j) += Jmg * repst * Jmk * factors[i];
                    integrals.Tsp(i,j) += Jmg * repsd * Jpk * factors[i];
                    integrals.Tps(i,j) += Jpg * repsd * Jmk * factors[i];
                    integrals.Tpp(i,j) += Jpg * repst * Jpk * factors[i];
                }
            }
        }

    } else {

        if (SOLVER->rule == BesselSolverCyl::RULE_DIRECT) {

            zero_matrix(integrals.Tss);
            zero_matrix(integrals.Tsp);
            zero_matrix(integrals.Tps);
            zero_matrix(integrals.Tpp);

            for (size_t ri = 0; ri != nr; ++ri) {
                double r = raxis->at(ri);
                dcomplex repst = r * (datar[ri] + datap[ri]), repsd = r * (datar[ri] - datap[ri]);
                for (size_t i = 0; i < N; ++i) {
                    double kr = kpts[i] * ib * r;
                    Jm[i] = cyl_bessel_j(m-1, kr);
                    Jp[i] = cyl_bessel_j(m+1, kr);
                }
                for (size_t i = 0; i < N; ++i) {
                    dcomplex cs = factors[i] * repst, cd = factors[i] * repsd;
                    for (size_t j = 0; j < N; ++j) {
                        integrals.Tss(i,j) += cs * Jm[i] * Jm[j];
                        integrals.Tsp(i,j) += cd * Jm[i] * Jp[j];
                        integrals.Tps(i,j) += cd * Jp[i] * Jm[j];
                        integrals.Tpp(i,j) += cs * Jp[i] * Jp[j];
                    }
                }
            }

        } else {

            if (SOLVER->rule == BesselSolverCyl::RULE_COMBINED_1) {

                cmatrix workess(N, N, temp.data()), workepp(N, N, temp.data()+N*N),
                        worksp(N, N, temp.data()+2*N*N), workps(N, N, temp.data()+3*N*N);

                zero_matrix(workess);
                zero_matrix(workepp);
                zero_matrix(worksp);
                zero_matrix(workps);

                for (size_t ri = 0, wi = 0, seg = 0, nw = segments[0].weights.size(); ri != nr; ++ri, ++wi) {
                    if (wi == nw) {
                        nw = segments[++seg].weights.size();
                        wi = 0;
                    }
                    double r = raxis->at(ri);
                    double rw = r * segments[seg].weights[wi] * segments[seg].D;
                    dcomplex riepsr = r * datar[ri];
                    for (size_t i = 0; i < N; ++i) {
                        double kr = kpts[i] * ib * r;
                        Jm[i] = cyl_bessel_j(m-1, kr);
                        Jp[i] = cyl_bessel_j(m+1, kr);
                    }
                    for (size_t i = 0; i < N; ++i) {
                        double cw = factors[i] * rw;
                        dcomplex ce = factors[i] * riepsr;
                        for (size_t j = 0; j < N; ++j) {
                            workess(i,j) += ce * Jm[i] * Jm[j];
                            workepp(i,j) += ce * Jp[i] * Jp[j];
                            worksp(i,j) += cw * Jm[i] * Jp[j];
                            workps(i,j) += cw * Jp[i] * Jm[j];
                        }
                    }
                }
                make_unit_matrix(integrals.Tss);
                make_unit_matrix(integrals.Tpp);

                invmult(workess, integrals.Tss);
                invmult(workepp, integrals.Tpp);

                mult_matrix_by_matrix(integrals.Tss, worksp, integrals.Tsp);
                mult_matrix_by_matrix(integrals.Tpp, workps, integrals.Tps);

                if (N > 2) {
                    std::copy_n(factors, N, reinterpret_cast<double*>(temp.data()));
                    factors = reinterpret_cast<double*>(temp.data());
                    Jm = factors + N; Jp = factors + 2*N;
                }

            } else { // if (SOLVER->rule == BesselSolverCyl::RULE_COMBINED_2)

                cmatrix work(temp);

                zero_matrix(integrals.TT);
                zero_matrix(work);
                zero_matrix(JmJp);
                zero_matrix(JpJm);

                for (size_t ri = 0, wi = 0, seg = 0, nw = segments[0].weights.size(); ri != nr; ++ri, ++wi) {
                    if (wi == nw) {
                        nw = segments[++seg].weights.size();
                        wi = 0;
                    }
                    double r = raxis->at(ri);
                    double rw = r * segments[seg].weights[wi] * segments[seg].D;
                    dcomplex riepsr = r * datar[ri];
                    for (size_t i = 0; i < N; ++i) {
                        double kr = kpts[i] * ib * r;
                        Jm[i] = cyl_bessel_j(m-1, kr);
                        Jp[i] = cyl_bessel_j(m+1, kr);
                    }
                    for (size_t j = 0; j < N; ++j) {
                        for (size_t i = 0; i < N; ++i) {
                            dcomplex ce = riepsr * factors[i];
                            double cw = rw * factors[i];
                            integrals.TT(i,j) += ce * Jm[i] * Jm[j];
                            integrals.TT(i,j+N) += ce * Jm[i] * Jp[j];
                            integrals.TT(i+N,j) += ce * Jp[i] * Jm[j];
                            integrals.TT(i+N,j+N) += ce * Jp[i] * Jp[j];
                            double mp = cw * Jm[i] * Jp[j];
                            double pm = cw * Jp[i] * Jm[j];
                            work(i,j+N) += mp;
                            work(i+N,j) += pm;
                            JmJp(i,j) += mp;
                            JpJm(i,j) += pm;
                        }
                    }
                }
                for (size_t i = 0; i < N; ++i) work(i,i) = 1.;
                for (size_t i = 0; i < N; ++i) work(i+N,i+N) = 1.;

                invmult(integrals.TT, work);

                for (size_t j = 0; j < N; ++j) {
                    for (size_t i = 0; i < N; ++i) integrals.Tss(i,j) = work(i,j);
                    for (size_t i = 0; i < N; ++i) integrals.Tps(i,j) = work(i+N,j);
                }
                for (size_t j = 0; j < N; ++j) {
                    for (size_t i = 0; i < N; ++i) integrals.Tsp(i,j) = work(i,j+N);
                    for (size_t i = 0; i < N; ++i) integrals.Tpp(i,j) = work(i+N,j+N);
                }

                zgemm('N', 'N', int(N), int(N), int(N), 1., JpJm.data(), int(N), work.data()+2*N*N, int(2*N), 1.,
                    integrals.Tpp.data(), int(N));
                zgemm('N', 'N', int(N), int(N), int(N), 1., JmJp.data(), int(N), work.data()+N, int(2*N), 1.,
                    integrals.Tss.data(), int(N));
                zgemm('N', 'N', int(N), int(N), int(N), 1., JpJm.data(), int(N), work.data(), int(2*N), 1.,
                    integrals.Tps.data(), int(N));
                zgemm('N', 'N', int(N), int(N), int(N), 1., JmJp.data(), int(N), work.data()+2*N*N+N, int(2*N), 1.,
                    integrals.Tsp.data(), int(N));
            }

            for (size_t ri = 0; ri != nr; ++ri) {
                double r = raxis->at(ri);
                dcomplex repsp = r * datap[ri];
                for (size_t i = 0; i < N; ++i) {
                    double kr = kpts[i] * ib * r;
                    Jm[i] = cyl_bessel_j(m-1, kr);
                    Jp[i] = cyl_bessel_j(m+1, kr);
                }
                for (size_t i = 0; i < N; ++i) {
                    dcomplex c = repsp * factors[i];
                    for (size_t j = 0; j < N; ++j) {
                        integrals.Tss(i,j) += c * Jm[i] * Jm[j];
                        integrals.Tsp(i,j) -= c * Jm[i] * Jp[j];
                        integrals.Tps(i,j) -= c * Jp[i] * Jm[j];
                        integrals.Tpp(i,j) += c * Jp[i] * Jp[j];
                    }
                }
            }
        }

        cmatrix work(N, N, temp.data()+N*N);

        zero_matrix(work);
        double* J = Jm;

        for (size_t ri = 0; ri != nr; ++ri) {
            double r = raxis->at(ri);
            dcomplex repsz = r * dataz[ri];
            for (size_t i = 0; i < N; ++i) {
                double kr = kpts[i] * ib * r;
                J[i] = cyl_bessel_j(m, kr);
            }
            for (size_t j = 0; j < N; ++j) {
                for (size_t i = 0; i < N; ++i) {
                    work(i,j) += repsz * factors[i] * J[i] * J[j];
                }
            }
        }

        // make_unit_matrix(integrals.V_k);
        zero_matrix(integrals.V_k);
        for (int i = 0; i < N; ++i) {
            double k = kpts[i] * ib;
            integrals.V_k(i,i) = k;
        }
        invmult(work, integrals.V_k);
        // for (size_t i = 0; i < N; ++i) {
        //     double g = kpts[i] * ib;
        //     for (size_t j = 0; j < N; ++j) integrals.V_k(i,j) *= g;
        // }

    }
}

}}}  // namespace plask::optical::slab
