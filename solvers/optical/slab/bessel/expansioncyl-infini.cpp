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
#include "expansioncyl-infini.hpp"
#include "solvercyl.hpp"
#include "zeros-data.hpp"
#include "../plask/math.hpp"

#include "../gauss_legendre.hpp"
#include "../gauss_laguerre.hpp"

#include "besselj.hpp"

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
        throw BadInput(solver->getId(), "outer geometry edge must be 'extend' or a simple material");

    double k0 = isnan(lam0)? this->k0.real() : 2e3*M_PI / lam0;
    double kmax = SOLVER->kmax * k0;

    size_t N = SOLVER->size;
    double R = rbounds[rbounds.size()-1];
    double ib = 1. / R;
    double kdlt;

    switch (SOLVER->kmethod) {
        case BesselSolverCyl::WAVEVECTORS_UNIFORM:
            SOLVER->writelog(LOG_DETAIL, "Using uniform wavevectors");
            if (isnan(k0)) throw BadInput(SOLVER->getId(), "no wavelength given: specify 'lam' or 'lam0'");
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
                    throw BadInput(SOLVER->getId(), "if no weights are given, number of manually specified wavevectors must be {}",
                                   N+1);
                for (size_t i = 0; i != N; ++i) {
                    kpts[i] = 0.5 * (SOLVER->klist[i] + SOLVER->klist[i+1]) * R;
                    kdelts[i] = (SOLVER->klist[i+1] - SOLVER->klist[i]);
                }
            } else {
                if (SOLVER->klist.size() != N)
                    throw BadInput(SOLVER->getId(), "if weights are given, number of manually specified wavevectors must be {}",
                                   N);
                if (SOLVER->kweights->size() != N)
                    throw BadInput(SOLVER->getId(), "number of manually specified wavevector weights must be {}", N+1);
                kpts = SOLVER->klist;
                for (double& k: kpts) k *= R;
                kdelts.reset(SOLVER->kweights->begin(), SOLVER->kweights->end());
            }
            break;
        case BesselSolverCyl::WAVEVECTORS_NONUNIFORM:
            SOLVER->writelog(LOG_DETAIL, "Using non-uniform wavevectors");
            if (isnan(k0)) throw BadInput(SOLVER->getId(), "no wavelength given: specify 'lam' or 'lam0'");
            kpts.resize(N);
            kdelts.reset(N);
            // Häyrynen, T., de Lasson, J.R., Gregersen, N., 2016.
            // Open-geometry Fourier modal method: modeling nanophotonic structures in infinite domains.
            // J. Opt. Soc. Am. A 33, 1298. https://doi.org/10.1364/josaa.33.001298
            int M1, M2, M3;
            M1 = M2 = (N+1) / 3;
            M3 = N - M1 - M2;
            if (M3 < 0) throw BadInput(SOLVER->getId(), "too small expansion size");
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
                throw BadInput(SOLVER->getId(), "for non-uniform wavevectors kmax must be at least {}",
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
    if (isnan(k0)) throw BadInput(SOLVER->getId(), "wavelength or k0 not set");
    if (isinf(k0.real())) throw BadInput(SOLVER->getId(), "wavelength must not be 0");

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

void ExpansionBesselInfini::integrateParams(Integrals& integrals,
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
        factors[i] = kpts[i] * ib * kdelts[i];
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
        for (size_t i = 0; i < N; ++i) {
            integrals.V_k(i,i) += dataz0 * kpts[i] * ib;
            dcomplex epst = datar0 + datap0, epsd = datar0 - datap0;
            integrals.Tss(i,i) += epst;
            integrals.Tsp(i,i) += epsd;
            integrals.Tps(i,i) += epsd;
            integrals.Tpp(i,i) += epst;
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
            for (size_t i = 0; i < N; ++i) {
                dcomplex epst = datar0 + datap0, epsd = datar0 - datap0;
                integrals.Tss(i,i) += epst;
                integrals.Tsp(i,i) += epsd;
                integrals.Tps(i,i) += epsd;
                integrals.Tpp(i,i) += epst;
            }

        } else {

            if (SOLVER->rule == BesselSolverCyl::RULE_COMBINED_1) {

                cmatrix workess(N, N, temp.data()), workepp(N, N, temp.data()+N*N),
                        worksp(N, N, temp.data()+2*N*N), workps(N, N, temp.data()+3*N*N);

                zero_matrix(workess);
                zero_matrix(workepp);
                zero_matrix(worksp);
                zero_matrix(workps);

                for (size_t ri = 0; ri != nr; ++ri) {
                    double r = raxis->at(ri);
                    dcomplex riepsr = r * datar[ri];
                    for (size_t i = 0; i < N; ++i) {
                        double kr = kpts[i] * ib * r;
                        Jm[i] = cyl_bessel_j(m-1, kr);
                        Jp[i] = cyl_bessel_j(m+1, kr);
                    }
                    for (size_t i = 0; i < N; ++i) {
                        dcomplex ce = factors[i] * riepsr;
                        for (size_t j = 0; j < N; ++j) {
                            workess(i,j) += ce * Jm[i] * Jm[j];
                            workepp(i,j) += ce * Jp[i] * Jp[j];
                        }
                    }
                }
                for (size_t i = 0; i < N; ++i) {
                    workess(i,i) += datar0;
                    workepp(i,i) += datar0;
                }

                make_unit_matrix(integrals.Tss);
                make_unit_matrix(integrals.Tpp);

                invmult(workess, integrals.Tss);
                invmult(workepp, integrals.Tpp);

                zero_matrix(integrals.Tsp);
                zero_matrix(integrals.Tps);

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

                for (size_t ri = 0; ri != nr; ++ri) {
                    double r = raxis->at(ri);
                    dcomplex riepsr = r * datar[ri];
                    for (size_t i = 0; i < N; ++i) {
                        double kr = kpts[i] * ib * r;
                        Jm[i] = cyl_bessel_j(m-1, kr);
                        Jp[i] = cyl_bessel_j(m+1, kr);
                    }
                    for (size_t j = 0; j < N; ++j) {
                        for (size_t i = 0; i < N; ++i) {
                            dcomplex ce = riepsr * factors[i];
                            integrals.TT(i,j) += ce * Jm[i] * Jm[j];
                            integrals.TT(i,j+N) += ce * Jm[i] * Jp[j];
                            integrals.TT(i+N,j) += ce * Jp[i] * Jm[j];
                            integrals.TT(i+N,j+N) += ce * Jp[i] * Jp[j];
                        }
                    }
                }
                for (size_t i = 0; i < 2*N; ++i) {
                    integrals.TT(i,i) += datar0;
                }

                // Compute  Jp(kr) Jm(gr) r dr  and  Jp(gr) Jm(kr) r dr  using analytical formula
                for (size_t j = 0; j < N; ++j) {
                    double k = kpts[j] * ib;
                    for (size_t i = 0; i < j; ++i) {
                        double g = kpts[i] * ib;
                        double val = factors[i] * 2*m / (k*g) * pow(g/k, m);
                        JmJp(i,j) = work(i,j+N) = val;   // g<k s=g p=k
                        integrals.TT(i,j+N) += datar0 * val;
                    }
                    double val = factors[j] * m / (k*k) - 1.;
                    JmJp(j,j) = JpJm(j,j) = work(j,j+N) = work(j+N,j) = val;
                    dcomplex iepsr0 = val * datar0;
                    integrals.TT(j,j+N) += iepsr0;
                    integrals.TT(j+N,j) += iepsr0;
                    for (size_t i = j+1; i < N; ++i) {
                        double g = kpts[i] * ib;
                        double val = factors[i] * 2*m / (k*g) * pow(k/g, m);
                        JpJm(i,j) = work(i+N,j) = val;   // k<g s=k p=g
                        integrals.TT(i+N,j) += datar0 * val;
                    }
                }
                for (size_t i = 0; i < N; ++i) work(i,i) = 1.;
                for (size_t i = 0; i < N; ++i) work(i+N,i+N) = 1.;

                invmult(integrals.TT, work);

                zero_matrix(integrals.Tsp);
                zero_matrix(integrals.Tps);

                for (size_t j = 0; j < N; ++j) {
                    for (size_t i = 0; i < N; ++i) integrals.Tss(i,j) = work(i,j);
                    for (size_t i = 0; i < N; ++i) integrals.Tps(i,j) = work(i+N,j);
                }

                zgemm('N', 'N', int(N), int(N), int(N), 1., JpJm.data(), int(N), work.data()+2*N*N, int(2*N), 1.,
                    integrals.Tpp.data(), int(N));
                zgemm('N', 'N', int(N), int(N), int(N), 1., JmJp.data(), int(N), work.data()+N, int(2*N), 1.,
                    integrals.Tss.data(), int(N));
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
                        integrals.Tpp(i,j) += c * Jp[i] * Jp[j];
                    }
                }
            }
            for (size_t i = 0; i < N; ++i) {
                integrals.Tss(i,i) += datap0;
                integrals.Tpp(i,i) += datap0;
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
        for (size_t i = 0; i < N; ++i) work(i,i) += dataz0;

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

}}} // # namespace plask::optical::slab
