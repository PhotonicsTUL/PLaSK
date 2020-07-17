#include "expansioncyl.h"
#include "solvercyl.h"
#include "zeros-data.h"

#include "expansioncyl-fini.h"
#include "expansioncyl-infini.h"

#include "../gauss_legendre.h"

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/legendre.hpp>
using boost::math::cyl_bessel_j;
using boost::math::cyl_bessel_j_zero;
using boost::math::legendre_p;

#define SOLVER static_cast<BesselSolverCyl*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionBessel::ExpansionBessel(BesselSolverCyl* solver) : Expansion(solver), m(1), initialized(false), m_changed(true) {}

size_t ExpansionBessel::matrixSize() const { return 2 * SOLVER->size; }

void ExpansionBessel::init1() {
    // Initialize segments
    if (SOLVER->mesh)
        rbounds = OrderedAxis(*SOLVER->getMesh());
    else
        rbounds = std::move(*makeGeometryGrid1D(SOLVER->getGeometry()));
    rbounds.addPoint(0.);
    OrderedAxis::WarningOff nowarn_rbounds(rbounds);
    size_t nseg = rbounds.size() - 1;
    if (dynamic_cast<ExpansionBesselFini*>(this)) {
        if (SOLVER->pml.dist > 0.) rbounds.addPoint(rbounds[nseg++] + SOLVER->pml.dist);
        if (SOLVER->pml.size > 0.) rbounds.addPoint(rbounds[nseg++] + SOLVER->pml.size);
    }
    segments.resize(nseg);
    double a, b = 0.;
    for (size_t i = 0; i < nseg; ++i) {
        a = b;
        b = rbounds[i + 1];
        segments[i].Z = 0.5 * (a + b);
        segments[i].D = 0.5 * (b - a);
    }

    diagonals.assign(solver->lcount, false);
    initialized = true;
    m_changed = true;
}

void ExpansionBessel::reset() {
    layers_integrals.clear();
    segments.clear();
    kpts.clear();
    initialized = false;
    mesh.reset();
    temporary.reset();
}

void ExpansionBessel::init3() {
    size_t nseg = rbounds.size() - 1;

    // Estimate necessary number of integration points
    double k = kpts[kpts.size() - 1];

    double expected = rbounds[rbounds.size()-1] * cyl_bessel_j(m+1, k);
    expected = 0.5 * expected * expected;

    k /= rbounds[rbounds.size() - 1];

    double max_error = SOLVER->integral_error * expected / double(nseg);
    double error = 0.;

    std::deque<std::vector<double>> abscissae_cache;
    std::deque<DataVector<double>> weights_cache;

    auto raxis = plask::make_shared<OrderedAxis>();
    OrderedAxis::WarningOff nowarn_raxis(raxis);

    double expcts = 0.;
    for (size_t i = 0; i < nseg; ++i) {
        double b = rbounds[i + 1];

        // excpected value is the second Lommel's integral
        double expct = expcts;
        expcts = cyl_bessel_j(m, k*b);
        expcts = 0.5 * b * b * (expcts * expcts - cyl_bessel_j(m-1, k*b) * cyl_bessel_j(m+1, k*b));
        expct = expcts - expct;

        double err = 2 * max_error;
        std::vector<double> points;
        size_t j, n = 0;
        double sum;
        for (j = 0; err > max_error && n <= SOLVER->max_integration_points; ++j) {
            n = 4 * (j+1) - 1;
            if (j == abscissae_cache.size()) {
                abscissae_cache.push_back(std::vector<double>());
                weights_cache.push_back(DataVector<double>());
                gaussLegendre(n, abscissae_cache.back(), weights_cache.back());
            }
            assert(j < abscissae_cache.size());
            assert(j < weights_cache.size());
            const std::vector<double>& abscissae = abscissae_cache[j];
            points.clear();
            points.reserve(abscissae.size());
            sum = 0.;
            for (size_t a = 0; a != abscissae.size(); ++a) {
                double r = segments[i].Z + segments[i].D * abscissae[a];
                double Jm = cyl_bessel_j(m, k*r);
                sum += weights_cache[j][a] * r * Jm * Jm;
                points.push_back(r);
            }
            sum *= segments[i].D;
            err = abs(sum - expct);
        }
        error += err;
        raxis->addOrderedPoints(points.begin(), points.end());
        segments[i].weights = weights_cache[j-1];
    }

    SOLVER->writelog(LOG_DETAIL, "Sampling structure in {:d} points (error: {:g}/{:g})", raxis->size(), error / expected,
                     SOLVER->integral_error);

    // Allocate memory for integrals
    size_t nlayers = solver->lcount;
    layers_integrals.resize(nlayers);

    mesh = plask::make_shared<RectangularMesh<2>>(raxis, solver->verts, RectangularMesh<2>::ORDER_01);

    m_changed = false;
}

void ExpansionBessel::prepareIntegrals(double lam, double glam) {
    if (m_changed) init2();
    temperature = SOLVER->inTemperature(mesh);
    gain_connected = SOLVER->inGain.hasProvider();
    if (gain_connected) {
        if (isnan(glam)) glam = lam;
        gain = SOLVER->inGain(mesh, glam);
    }
}

Tensor3<dcomplex> ExpansionBessel::getEps(size_t layer, size_t ri, double r, double matz, double lam, double glam) {
    Tensor3<dcomplex> eps;
    {
        OmpLockGuard<OmpNestLock> lock;  // this must be declared before `material` to guard its destruction
        auto material = SOLVER->getGeometry()->getMaterial(vec(r, matz));
        lock = material->lock();
        eps = material->NR(lam, getT(layer, ri));
        if (isnan(eps.c00) || isnan(eps.c11) || isnan(eps.c22) || isnan(eps.c01))
            throw BadInput(solver->getId(), "Complex refractive index (NR) for {} is NaN at lam={}nm and T={}K", material->name(),
                           lam, getT(layer, ri));
    }
    if (!is_zero(eps.c00 - eps.c11) || eps.c01 != 0.)
        throw BadInput(solver->getId(), "Lateral anisotropy not allowed for this solver");
    if (gain_connected && solver->lgained[layer]) {
        auto roles = SOLVER->getGeometry()->getRolesAt(vec(r, matz));
        if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
            Tensor2<double> g = 0.;
            double W = 0.;
            for (size_t k = 0, v = ri * solver->verts->size(); k != mesh->vert()->size(); ++v, ++k) {
                if (solver->stack[k] == layer) {
                    double w =
                        (k == 0 || k == mesh->vert()->size() - 1) ? 1e-6 : solver->vbounds->at(k) - solver->vbounds->at(k - 1);
                    g += w * gain[v];
                    W += w;
                }
            }
            Tensor2<double> ni = glam * g / W * (0.25e-7 / PI);
            eps.c00.imag(ni.c00);
            eps.c11.imag(ni.c00);
            eps.c22.imag(ni.c11);
        }
    }
    eps.sqr_inplace();
    return eps;
}

void ExpansionBessel::cleanupIntegrals(double, double) {
    temperature.reset();
    gain.reset();
}

void ExpansionBessel::integrateParams(Integrals& integrals,
                                      const dcomplex* datap, const dcomplex* datar, const dcomplex* dataz,
                                      dcomplex datap0, dcomplex datar0, dcomplex dataz0) {
    auto raxis = mesh->tran();

    size_t nr = raxis->size(), N = SOLVER->size;
    double R = rbounds[rbounds.size()-1];
    double ib = 1. / R;
    double* factors; // scale factors for making matrices orthonormal

    ExpansionBesselInfini* this_infini = dynamic_cast<ExpansionBesselInfini*>(this);
    bool finite = !this_infini;

    integrals.reset(N);

    TempMatrix temp = getTempMatrix();
    aligned_unique_ptr<double> _tmp;

    if (N < 2) {
        _tmp.reset(aligned_malloc<double>(4*N));
        factors = _tmp.get();
    } else if (SOLVER->rule == BesselSolverCyl::RULE_INVERSE_3) {
        _tmp.reset(aligned_malloc<double>(3*N));
        factors = _tmp.get();
    } else if (SOLVER->rule == BesselSolverCyl::RULE_INVERSE_1 || SOLVER->rule == BesselSolverCyl::RULE_INVERSE_2 ) {
        factors = reinterpret_cast<double*>(integrals.V_k.data());
    } else {
        factors = reinterpret_cast<double*>(temp.data());
    }
    if (finite) {
        for (size_t i = 0; i < N; ++i) {
            double fact = R * cyl_bessel_j(m+1, kpts[i]);
            factors[i] = 2. / (fact * fact);
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            factors[i] = kpts[i] * ib * this_infini->kdelts[i];
        }
    }
    double* Jm = factors + N;
    double* Jp = factors + 2*N;

    if (SOLVER->rule == BesselSolverCyl::RULE_DIRECT) {

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
        if (!finite) {
            for (size_t i = 0; i < N; ++i) {
                integrals.V_k(i,i) += dataz0 * kpts[i] * ib;
                dcomplex epst = datar0 + datap0, epsd = datar0 - datap0;
                integrals.Tss(i,i) += epst;
                integrals.Tsp(i,i) += epsd;
                integrals.Tps(i,i) += epsd;
                integrals.Tpp(i,i) += epst;
            }
        }

    } else {

        if (SOLVER->rule == BesselSolverCyl::RULE_SEMI_INVERSE) {

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
            if (!finite) {
                for (size_t i = 0; i < N; ++i) {
                    dcomplex epst = datar0 + datap0, epsd = datar0 - datap0;
                    integrals.Tss(i,i) += epst;
                    integrals.Tsp(i,i) += epsd;
                    integrals.Tps(i,i) += epsd;
                    integrals.Tpp(i,i) += epst;
                }
            }

        } else {

            if (SOLVER->rule == BesselSolverCyl::RULE_INVERSE_1) {

                cmatrix workess(N, N, temp.data()), workepp(N, N, temp.data()+N*N),
                        worksp(N, N, temp.data()+2*N*N), workps(N, N, temp.data()+3*N*N);

                zero_matrix(workess);
                zero_matrix(workepp);
                zero_matrix(worksp);
                zero_matrix(workps);

                if (finite) {
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
                } else {
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

                    // Compute  Jp(kr) Jm(gr) r dr  and  Jp(gr) Jm(kr) r dr  using analytical formula
                    for (size_t j = 0; j < N; ++j) {
                        double k = kpts[j] * ib;
                        for (size_t i = 0; i < j; ++i) {
                            double g = kpts[i] * ib;
                            worksp(i,j) = factors[i] * 2*m / (k*g) * pow(g/k, m);   // g<k s=g p=k
                        }
                        worksp(j,j) = workps(j,j) = factors[j] * m / (k*k) - 1.;
                        for (size_t i = j+1; i < N; ++i) {
                            double g = kpts[i] * ib;
                            workps(i,j) = factors[i] * 2*m / (k*g) * pow(k/g, m);   // k<g s=k p=g
                        }
                    }
                }

                make_unit_matrix(integrals.Tss);
                make_unit_matrix(integrals.Tpp);

                invmult(workess, integrals.Tss);
                invmult(workepp, integrals.Tpp);

                mult_matrix_by_matrix(integrals.Tss, worksp, integrals.Tsp);
                mult_matrix_by_matrix(integrals.Tpp, workps, integrals.Tps);

                std::copy_n(factors, N, reinterpret_cast<double*>(temp.data()));
                factors = reinterpret_cast<double*>(temp.data());
                Jm = factors + N; Jp = factors + 2*N;

            } else if (SOLVER->rule == BesselSolverCyl::RULE_INVERSE_2) {
                if (!finite) throw NotImplemented("inverse rule (variant 2) for infinite expansion");

                double* J  = factors + 3*N;

                cmatrix workqs(N, N, temp.data()), workqp(N, N, temp.data()+N*N),
                        worksh(N, N, temp.data()+2*N*N), workph(N, N, temp.data()+3*N*N);

                zero_matrix(workqs);
                zero_matrix(workqp);
                zero_matrix(worksh);
                zero_matrix(workph);
                zero_matrix(integrals.Tss);

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
                        J[i]  = cyl_bessel_j(m, kr);
                        Jp[i] = cyl_bessel_j(m+1, kr);
                    }
                    for (size_t i = 0; i < N; ++i) {
                        double cw = factors[i] * rw;
                        for (size_t j = 0; j < N; ++j) {
                            integrals.Tss(i,j) += riepsr * J[i] * J[j];
                            workqs(i,j) += rw * J[i] * Jm[j];
                            workqp(i,j) += rw * J[i] * Jp[j];
                            worksh(i,j) += cw * Jm[i] * J[j];
                            workph(i,j) += cw * Jp[i] * J[j];
                        }
                    }
                }

                cmatrix workqsp(N, 2*N, temp.data());
                invmult(integrals.Tss, workqsp);    // column storage makes it possible

                mult_matrix_by_matrix(worksh, workqs, integrals.Tss);
                mult_matrix_by_matrix(worksh, workqp, integrals.Tsp);
                mult_matrix_by_matrix(workph, workqs, integrals.Tps);
                mult_matrix_by_matrix(workph, workqp, integrals.Tpp);

                std::copy_n(factors, N, reinterpret_cast<double*>(temp.data()));
                factors = reinterpret_cast<double*>(temp.data());
                Jm = factors + N; Jp = factors + 2*N;

            } else { // if (SOLVER->rule == BesselSolverCyl::RULE_INVERSE_3)
                if (!finite) throw NotImplemented("inverse rule (variant 3) for infinite expansion");

                cmatrix work(temp);

                zero_matrix(integrals.V_k);
                zero_matrix(integrals.TT);
                zero_matrix(work);

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
                            integrals.TT(i,j) += riepsr * Jm[i] * Jm[j];
                            integrals.TT(i+N,j) += riepsr * Jp[i] * Jm[j];
                            integrals.TT(i,j+N) += riepsr * Jm[i] * Jp[j];
                            integrals.TT(i+N,j+N) += riepsr * Jp[i] * Jp[j];
                            work(i+N,j) += rw * Jp[i] * Jm[j];
                            dcomplex mp = rw * Jm[i] * Jp[j];
                            work(i,j+N) += mp;
                            integrals.V_k(i,j) += mp;
                        }
                    }
                }
                for (size_t i = 0; i < N; ++i) work(i,i) = 1. / factors[i];
                for (size_t i = 0; i < N; ++i) work(i+N,i+N) = 1. / factors[i];

                invmult(integrals.TT, work);

                for (size_t j = 0; j < N; ++j) {
                    for (size_t i = 0; i < N; ++i) integrals.Tss(i,j) = work(i,j) / factors[i];
                    for (size_t i = 0; i < N; ++i) integrals.Tps(i,j) = work(i+N,j) / factors[i];
                }
                for (size_t j = 0; j < N; ++j) {
                    for (size_t i = 0; i < N; ++i) integrals.Tsp(i,j) = work(i,j+N) / factors[i];
                    for (size_t i = 0; i < N; ++i) integrals.Tpp(i,j) = work(i+N,j+N) / factors[i];
                }

                zgemm('T', 'N', int(N), int(N), int(N), 1., integrals.V_k.data(), int(N), work.data(), int(2*N), 1.,
                    integrals.Tps.data(), int(N));
                zgemm('T', 'N', int(N), int(N), int(N), 1., integrals.V_k.data(), int(N), work.data()+2*N*N, int(2*N), 1.,
                    integrals.Tpp.data(), int(N));
                zgemm('N', 'N', int(N), int(N), int(N), 1., integrals.V_k.data(), int(N), work.data()+N, int(2*N), 1.,
                    integrals.Tss.data(), int(N));
                zgemm('N', 'N', int(N), int(N), int(N), 1., integrals.V_k.data(), int(N), work.data()+2*N*N+N, int(2*N), 1.,
                    integrals.Tsp.data(), int(N));

                for (size_t j = 0; j < N; ++j) {
                    for (size_t i = 0; i < N; ++i) {
                        integrals.Tss(i,j) *= factors[i];
                        integrals.Tsp(i,j) *= factors[i];
                        integrals.Tps(i,j) *= factors[i];
                        integrals.Tpp(i,j) *= factors[i];
                    }
                }

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
            if (!finite) {
                for (size_t i = 0; i < N; ++i) {
                    integrals.Tss(i,i) += datap0;
                    integrals.Tsp(i,i) -= datap0;
                    integrals.Tps(i,i) -= datap0;
                    integrals.Tpp(i,i) += datap0;
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
        if (!finite) for (size_t i = 0; i < N; ++i) work(i,i) += dataz0;

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

void ExpansionBessel::layerIntegrals(size_t layer, double lam, double glam) {
    if (isnan(real(k0)) || isnan(imag(k0))) throw BadInput(SOLVER->getId(), "No wavelength specified");

    auto geometry = SOLVER->getGeometry();

    auto raxis = mesh->tran();

    #if defined(OPENMP_FOUND)  // && !defined(NDEBUG)
        SOLVER->writelog(LOG_DETAIL, "Computing integrals for layer {:d}/{:d} with {} rule in thread {:d}",
                         layer, solver->lcount, SOLVER->ruleName(), omp_get_thread_num());
    #else
        SOLVER->writelog(LOG_DETAIL, "Computing integrals for layer {:d}/{:d} with {} rule",
                         layer, solver->lcount, SOLVER->ruleName());
    #endif

    if (isnan(lam)) throw BadInput(SOLVER->getId(), "No wavelength given: specify 'lam' or 'lam0'");

    size_t nr = raxis->size(), N = SOLVER->size;

    if (gain_connected && solver->lgained[layer]) {
        SOLVER->writelog(LOG_DEBUG, "Layer {:d} has gain", layer);
        if (isnan(glam)) glam = lam;
    }

    double matz;
    for (size_t i = 0; i != solver->stack.size(); ++i) {
        if (solver->stack[i] == layer) {
            matz = solver->verts->at(i);
            break;
        }
    }

    // For checking if the layer is uniform
    diagonals[layer] = true;

    bool finite = !dynamic_cast<ExpansionBesselInfini*>(this);

    size_t pmli = raxis->size();
    double pmlr;
    dcomplex epsp0, epsr0, epsz0;
    if (finite) {
        if (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) {
            size_t pmlseg = segments.size() - 1;
            pmli -= segments[pmlseg].weights.size();
            pmlr = rbounds[pmlseg];
        }
    } else {
        Tensor3<dcomplex> eps0 = getEps(layer, mesh->tran()->size()-1,
                                        rbounds[rbounds.size()-1] + 0.001, matz,
                                        lam, glam);
        eps0.sqr_inplace();
        epsp0 = eps0.c00;
        if (SOLVER->rule != BesselSolverCyl::RULE_DIRECT) {
            epsr0 = (SOLVER->rule != BesselSolverCyl::RULE_SEMI_INVERSE)? 1. / eps0.c11 : eps0.c11;
            epsz0 = eps0.c22;
        } else {
            epsr0 = eps0.c11;
            epsz0 = 1. / eps0.c22;
        }
        if (abs(epsp0.imag()) < SMALL) epsp0.imag(0.);
        if (abs(epsr0.imag()) < SMALL) epsr0.imag(0.);
        if (abs(epsz0.imag()) < SMALL) epsz0.imag(0.);
        writelog(LOG_DEBUG, "Reference refractive index for layer {} is {} / {}", layer, str(sqrt(epsp0)),
                 str(sqrt((SOLVER->rule != BesselSolverCyl::RULE_DIRECT)? epsz0 : (1. / epsz0))));
    }

    aligned_unique_ptr<dcomplex> epsp_data(aligned_malloc<dcomplex>(nr));
    aligned_unique_ptr<dcomplex> epsr_data(aligned_malloc<dcomplex>(nr));
    aligned_unique_ptr<dcomplex> epsz_data(aligned_malloc<dcomplex>(nr));

    // Compute integrals
    for (size_t ri = 0, wi = 0, seg = 0, nw = segments[0].weights.size(); ri != nr; ++ri, ++wi) {
        if (wi == nw) {
            nw = segments[++seg].weights.size();
            wi = 0;
        }
        double r = raxis->at(ri);
        double w = segments[seg].weights[wi] * segments[seg].D;

        Tensor3<dcomplex> eps = getEps(layer, ri, r, matz, lam, glam);
        if (ri >= pmli) {
            dcomplex f = 1. + (SOLVER->pml.factor - 1.) * pow((r - pmlr) / SOLVER->pml.size, SOLVER->pml.order);
            eps.c00 *= f;
            eps.c11 /= f;
            eps.c22 *= f;
        }
        dcomplex epsp = eps.c00, epsr, epsz;
        if (SOLVER->rule != BesselSolverCyl::RULE_DIRECT) {
            epsr = (SOLVER->rule != BesselSolverCyl::RULE_SEMI_INVERSE)? 1. / eps.c11 : eps.c11;
            epsz = eps.c22;
        } else {
            epsr = eps.c11;
            epsz = 1. / eps.c22;
        }

        if (finite) {
            if (ri == 0) {
                epsp0 = epsp;
                epsr0 = epsr;
                epsz0 = epsz;
            } else {
                if (!is_zero(epsp - epsp0) || !is_zero(epsr - epsr0) || !is_zero(epsz - epsz0))
                    diagonals[layer] = false;
            }
        } else {
            epsp -= epsp0;
            epsr -= epsr0;
            epsz -= epsz0;
            if (!is_zero(epsp) || !is_zero(epsr) || !is_zero(epsz)) diagonals[layer] = false;
        }

        epsp_data.get()[ri] = epsp * w;
        epsr_data.get()[ri] = epsr * w;
        epsz_data.get()[ri] = epsz * w;
    }

    if (diagonals[layer]) {
        Integrals& integrals = layers_integrals[layer];
        SOLVER->writelog(LOG_DETAIL, "Layer {0} is uniform", layer);
        integrals.reset(N);
        zero_matrix(integrals.V_k);
        zero_matrix(integrals.Tss);
        zero_matrix(integrals.Tsp);
        zero_matrix(integrals.Tps);
        zero_matrix(integrals.Tpp);
        double ib = 1. / rbounds[rbounds.size() - 1];
        for (size_t i = 0; i < N; ++i) {
            double k = kpts[i] * ib;
            if (SOLVER->rule != BesselSolverCyl::RULE_DIRECT)
                integrals.V_k(i,i) = k / epsz0;
            else
                integrals.V_k(i,i) = k * epsz0;
            // integrals.Tss(i,i) = integrals.Tpp(i,i) = 1. / iepsr0 + epsp0;
            // integrals.Tsp(i,i) = integrals.Tps(i,i) = 1. / iepsr0 - epsp0;
            integrals.Tss(i,i) = integrals.Tpp(i,i) = 2. * epsp0;
        }
    } else {
        integrateParams(layers_integrals[layer], epsp_data.get(), epsr_data.get(), epsz_data.get(), epsp0, epsr0, epsz0);
    }
}

#ifndef NDEBUG
cmatrix ExpansionBessel::epsV_k(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].V_k(i,j);
    return result;
}
cmatrix ExpansionBessel::epsTss(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].Tss(i,j);
    return result;
}
cmatrix ExpansionBessel::epsTsp(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].Tsp(i,j);
    return result;
}
cmatrix ExpansionBessel::epsTps(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].Tps(i,j);
    return result;
}
cmatrix ExpansionBessel::epsTpp(size_t layer) {
    size_t N = SOLVER->size;
    cmatrix result(N, N, 0.);
    for (size_t i = 0; i != N; ++i)
        for (size_t j = 0; j != N; ++j)
            result(i,j) = layers_integrals[layer].Tpp(i,j);
    return result;
}
#endif

void ExpansionBessel::prepareField() {
    if (field_interpolation == INTERPOLATION_DEFAULT) field_interpolation = INTERPOLATION_NEAREST;
}

void ExpansionBessel::cleanupField() {}

LazyData<Vec<3, dcomplex>> ExpansionBessel::getField(size_t layer,
                                                     const shared_ptr<const typename LevelsAdapter::Level>& level,
                                                     const cvector& E,
                                                     const cvector& H) {
    size_t N = SOLVER->size;

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());
    double ib = 1. / rbounds[rbounds.size() - 1];
    const dcomplex fz = - I / k0;

    auto src_mesh =
        plask::make_shared<RectangularMesh<2>>(mesh->tran(), plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));


    if (which_field == FIELD_E) {
        cvector Ez(N);
        {
            cvector Dz(N);
            for (size_t j = 0; j != N; ++j) {
                size_t js = idxs(j), jp = idxp(j);
                Dz[j] = fz * (H[js] + H[jp]);
            }
            mult_matrix_by_vector(layers_integrals[layer].V_k, Dz, Ez);
        }
        return LazyData<Vec<3, dcomplex>>(dest_mesh->size(), [=](size_t i) -> Vec<3, dcomplex> {
            double r = dest_mesh->at(i)[0];
            Vec<3, dcomplex> result{0., 0., 0.};
            for (size_t j = 0; j != N; ++j) {
                double k = kpts[j] * ib;
                double kr = k * r;
                double Jm = cyl_bessel_j(m-1, kr), Jp = cyl_bessel_j(m+1, kr), J = cyl_bessel_j(m, kr);
                size_t js = idxs(j), jp = idxp(j);
                result.c0 += Jm * E[js] - Jp * E[jp];       // E_p
                result.c1 += Jm * E[js] + Jp * E[jp];       // E_r
                result.c2 += J * Ez[j];                     // E_z
            }
            return result;
        });
    } else {  // which_field == FIELD_H
        double r0 = (SOLVER->pml.size > 0. && SOLVER->pml.factor != 1.) ? rbounds[rbounds.size() - 1] : INFINITY;
        return LazyData<Vec<3, dcomplex>>(dest_mesh->size(), [=](size_t i) -> Vec<3, dcomplex> {
            double r = dest_mesh->at(i)[0];
            dcomplex imu = 1.;
            if (r > r0) imu = 1. / (1. + (SOLVER->pml.factor - 1.) * pow((r - r0) / SOLVER->pml.size, SOLVER->pml.order));
            Vec<3, dcomplex> result{0., 0., 0.};
            for (size_t j = 0; j != N; ++j) {
                double k = kpts[j] * ib;
                double kr = k * r;
                double Jm = cyl_bessel_j(m-1, kr), Jp = cyl_bessel_j(m+1, kr), J = cyl_bessel_j(m, kr);
                size_t js = idxs(j), jp = idxp(j);
                result.c0 -= Jm * H[js] - Jp * H[jp];               // H_p
                result.c1 += Jm * H[js] + Jp * H[jp];               // H_r
                result.c2 += fz * k * imu * J * (E[js] + E[jp]);    // H_z
            }
            return result;
        });
    }
}

LazyData<Tensor3<dcomplex>> ExpansionBessel::getMaterialNR(size_t layer,
                                                           const shared_ptr<const typename LevelsAdapter::Level>& level,
                                                           InterpolationMethod interp) {
    if (interp == INTERPOLATION_DEFAULT) interp = INTERPOLATION_NEAREST;

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());

    double lam, glam;
    if (!isnan(lam0)) {
        lam = lam0;
        glam = (solver->always_recompute_gain) ? real(02e3 * PI / k0) : lam;
    } else {
        lam = glam = real(02e3 * PI / k0);
    }

    auto raxis = mesh->tran();

    DataVector<Tensor3<dcomplex>> nrs(raxis->size());
    for (size_t i = 0; i != nrs.size(); ++i) {
        Tensor3<dcomplex> eps = getEps(layer, i, raxis->at(i), level->vpos(), lam, glam);
        nrs[i] = eps.sqrt();
    }

    auto src_mesh =
        plask::make_shared<RectangularMesh<2>>(mesh->tran(), plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));
    return interpolate(
        src_mesh, nrs, dest_mesh, interp,
        InterpolationFlags(SOLVER->getGeometry(), InterpolationFlags::Symmetry::POSITIVE, InterpolationFlags::Symmetry::NO));
}


double ExpansionBessel::integratePoyntingVert(const cvector& E, const cvector& H) {
    double result = 0.;
    for (size_t i = 0, N = SOLVER->size; i < N; ++i) {
        size_t is = idxs(i);
        size_t ip = idxp(i);
        result += real(- E[is] * conj(H[is]) + E[ip] * conj(H[ip])) * fieldFactor(i);
    }
    return 4e-12 * PI * result;  // µm² -> m²
}

double ExpansionBessel::integrateField(WhichField field, size_t layer, const cvector& E, const cvector& H) {
    size_t N = SOLVER->size;
    double resxy = 0.;
    double resz = 0.;
    double R = rbounds[rbounds.size()-1];
    if (which_field == FIELD_E) {
        cvector Ez(N), Dz(N);
        for (size_t j = 0; j != N; ++j) {
            size_t js = idxs(j), jp = idxp(j);
            Dz[j] = H[js] + H[jp];
        }
        mult_matrix_by_vector(layers_integrals[layer].V_k, Dz, Ez);
        for (size_t i = 0, N = SOLVER->size; i < N; ++i) {
            double eta = fieldFactor(i);
            size_t is = idxs(i);
            size_t ip = idxp(i);
            resxy += real(E[is]*conj(E[is]) + E[ip]*conj(E[ip])) * eta;
            resz += real(Ez[i]*conj(Ez[i])) * eta;
        }
    } else {
        cvector Bz(N);
        for (size_t j = 0; j != N; ++j) {
            size_t js = idxs(j), jp = idxp(j);
            Bz[j] = E[js] + E[jp];
        }
        cvector Hz = getHz(Bz);
        for (size_t i = 0, N = SOLVER->size; i < N; ++i) {
            double eta = fieldFactor(i);
            size_t is = idxs(i);
            size_t ip = idxp(i);
            resxy += real(H[is]*conj(H[is]) + H[ip]*conj(H[ip])) * eta;
            resz += real(Hz[i]*conj(Hz[i])) * eta;
        }
    }
    return 2*PI * (resxy + resz / real(k0*conj(k0)));
}

}}}  // namespace plask::optical::slab
