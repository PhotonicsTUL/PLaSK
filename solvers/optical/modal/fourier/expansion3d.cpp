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
#include <boost/algorithm/clamp.hpp>
using boost::algorithm::clamp;

#include "../meshadapter.hpp"
#include "expansion3d.hpp"
#include "solver3d.hpp"

#define SOLVER static_cast<FourierSolver3D*>(solver)

namespace plask { namespace optical { namespace modal {

constexpr const char* GradientFunctions::NAME;
constexpr const char* GradientFunctions::UNIT;

ExpansionPW3D::ExpansionPW3D(FourierSolver3D* solver)
    : Expansion(solver), initialized(false), symmetry_long(E_UNSPECIFIED), symmetry_tran(E_UNSPECIFIED) {}

void ExpansionPW3D::init() {
    auto geometry = SOLVER->getGeometry();

    RegularAxis long_mesh, tran_mesh;

    periodic_long = geometry->isPeriodic(Geometry3D::DIRECTION_LONG);
    periodic_tran = geometry->isPeriodic(Geometry3D::DIRECTION_TRAN);

    back = geometry->getChild()->getBoundingBox().lower[0];
    front = geometry->getChild()->getBoundingBox().upper[0];
    left = geometry->getChild()->getBoundingBox().lower[1];
    right = geometry->getChild()->getBoundingBox().upper[1];

    size_t refl = SOLVER->refine_long, reft = SOLVER->refine_tran, nMl, nMt;
    if (refl == 0) refl = 1;
    if (reft == 0) reft = 1;

    if (symmetry_long != E_UNSPECIFIED && !geometry->isSymmetric(Geometry3D::DIRECTION_LONG))
        throw BadInput(solver->getId(), "longitudinal symmetry not allowed for asymmetric structure");
    if (symmetry_tran != E_UNSPECIFIED && !geometry->isSymmetric(Geometry3D::DIRECTION_TRAN))
        throw BadInput(solver->getId(), "transverse symmetry not allowed for asymmetric structure");

    if (geometry->isSymmetric(Geometry3D::DIRECTION_LONG)) {
        if (front <= 0.) {
            back = -back;
            front = -front;
            std::swap(back, front);
        }
        if (back != 0.)
            throw BadInput(SOLVER->getId(), "longitudinally symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric_long()) back = -front;
    }
    if (geometry->isSymmetric(Geometry3D::DIRECTION_TRAN)) {
        if (right <= 0.) {
            left = -left;
            right = -right;
            std::swap(left, right);
        }
        if (left != 0.)
            throw BadInput(SOLVER->getId(), "transversely symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric_tran()) left = -right;
    }

    if (!periodic_long) {
        if (SOLVER->getLongSize() == 0)
            throw BadInput(solver->getId(),
                           "Flat structure in longitudinal direction (size_long = 0) allowed only for periodic geometry");
        // Add PMLs
        if (!symmetric_long()) back -= SOLVER->pml_long.size + SOLVER->pml_long.dist;
        front += SOLVER->pml_long.size + SOLVER->pml_long.dist;
    }
    if (!periodic_tran) {
        if (SOLVER->getTranSize() == 0)
            throw BadInput(solver->getId(),
                           "Flat structure in transverse direction (size_tran = 0) allowed only for periodic geometry");
        // Add PMLs
        if (!symmetric_tran()) left -= SOLVER->pml_tran.size + SOLVER->pml_tran.dist;
        right += SOLVER->pml_tran.size + SOLVER->pml_tran.dist;
    }

    double Ll, Lt;

    eNl = 2 * SOLVER->getLongSize() + 1;
    eNt = 2 * SOLVER->getTranSize() + 1;

    if (!symmetric_long()) {
        Ll = front - back;
        Nl = 2 * SOLVER->getLongSize() + 1;
        nNl = 4 * SOLVER->getLongSize() + 1;
        nMl = refl * nNl;
        double dx = 0.5 * Ll * double(refl - 1) / double(nMl);
        long_mesh = RegularAxis(back - dx, front - dx - Ll / double(nMl), nMl);
    } else {
        Ll = 2 * front;
        Nl = SOLVER->getLongSize() + 1;
        nNl = 2 * SOLVER->getLongSize() + 1;
        nMl = refl * nNl;
        if (SOLVER->dct2()) {
            double dx = 0.25 * Ll / double(nMl);
            long_mesh = RegularAxis(dx, front - dx, nMl);
        } else {
            size_t nNa = 4 * SOLVER->getLongSize() + 1;
            double dx = 0.5 * Ll * double(refl - 1) / double(refl * nNa);
            long_mesh = RegularAxis(-dx, front + dx, nMl);
        }
    }  // N = 3  nN = 5  refine = 5  M = 25
    if (!symmetric_tran()) {                                    //  . . 0 . . . . 1 . . . . 2 . . . . 3 . . . . 4 . .
        Lt = right - left;                                      //  ^ ^ ^ ^ ^
        Nt = 2 * SOLVER->getTranSize() + 1;                     // |0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|5 6 7 8 9|0 1 2 3 4|
        nNt = 4 * SOLVER->getTranSize() + 1;                    // N = 3  nN = 5  refine = 4  M = 20
        nMt = reft * nNt;                                       // . . 0 . . . 1 . . . 2 . . . 3 . . . 4 . . . 0
        double dx = 0.5 * Lt * double(reft - 1) / double(nMt);  //  ^ ^ ^ ^
        tran_mesh = RegularAxis(left - dx, right - dx - Lt / double(nMt), nMt);  // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
    } else {
        Lt = 2 * right;                       // N = 3  nN = 5  refine = 4  M = 20
        Nt = SOLVER->getTranSize() + 1;       // # . 0 . # . 1 . # . 2 . # . 3 . # . 4 . # . 4 .
        nNt = 2 * SOLVER->getTranSize() + 1;  //  ^ ^ ^ ^
        nMt = reft * nNt;                     // |0 1 2 3|4 5 6 7|8 9 0 1|2 3 4 5|6 7 8 9|
        if (SOLVER->dct2()) {
            double dx = 0.25 * Lt / double(nMt);
            tran_mesh = RegularAxis(dx, right - dx, nMt);
        } else {
            size_t nNa = 4 * SOLVER->getTranSize() + 1;
            double dx = 0.5 * Lt * double(reft - 1) / double(reft * nNa);
            tran_mesh = RegularAxis(-dx, right + dx, nMt);
        }
    }

    SOLVER->writelog(LOG_DETAIL, "Creating expansion{3} with {0}x{1} plane-waves (matrix size: {2})", Nl, Nt, matrixSize(),
                     (!symmetric_long() && !symmetric_tran())  ? ""
                     : (symmetric_long() && symmetric_tran())  ? " symmetric in longitudinal and transverse directions"
                     : (!symmetric_long() && symmetric_tran()) ? " symmetric in transverse direction"
                                                               : " symmetric in longitudinal direction");

    if (symmetric_long())
        SOLVER->writelog(LOG_DETAIL, "Longitudinal symmetry is {0}", (symmetry_long == E_TRAN) ? "Etran" : "Elong");
    if (symmetric_tran()) SOLVER->writelog(LOG_DETAIL, "Transverse symmetry is {0}", (symmetry_tran == E_TRAN) ? "Etran" : "Elong");

    auto dct_symmetry = SOLVER->dct2() ? FFT::SYMMETRY_EVEN_2 : FFT::SYMMETRY_EVEN_1;
    auto dct_asymmetry = SOLVER->dct2() ? FFT::SYMMETRY_ODD_2 : FFT::SYMMETRY_ODD_1;

    matFFT = FFT::Forward2D(6, nNl, nNt, symmetric_long() ? dct_symmetry : FFT::SYMMETRY_NONE,
                            symmetric_tran() ? dct_symmetry : FFT::SYMMETRY_NONE);

    if (SOLVER->expansion_rule == FourierSolver3D::RULE_COMBINED) {
        cos2FFT = FFT::Forward2D(2, nNl, nNt, symmetric_long() ? dct_symmetry : FFT::SYMMETRY_NONE,
                                 symmetric_tran() ? dct_symmetry : FFT::SYMMETRY_NONE);
        cssnFFT = FFT::Forward2D(2, nNl, nNt, symmetric_long() ? dct_asymmetry : FFT::SYMMETRY_NONE,
                                 symmetric_tran() ? dct_asymmetry : FFT::SYMMETRY_NONE);
    }

    // Compute permeability coefficients
    if (!periodic_long || !periodic_tran) {
        SOLVER->writelog(LOG_DETAIL, "Adding side PMLs (total structure dimensions: {0}um x {1}um)", Ll, Lt);
    }
    if (periodic_long) {
        mag_long.reset(nNl, Tensor2<dcomplex>(0.));
        mag_long[0].c00 = 1.;
        mag_long[0].c11 = 1.;  // constant 1
    } else {
        mag_long.reset(nNl, Tensor2<dcomplex>(0.));
        double pb = back + SOLVER->pml_long.size, pf = front - SOLVER->pml_long.size;
        if (symmetric_long())
            pib = 0;
        else
            pib = std::lower_bound(long_mesh.begin(), long_mesh.end(), pb) - long_mesh.begin();
        pif = std::lower_bound(long_mesh.begin(), long_mesh.end(), pf) - long_mesh.begin();
        for (size_t i = 0; i != nNl; ++i) {
            for (size_t j = refl * i, end = refl * (i + 1); j != end; ++j) {
                dcomplex s = 1.;
                if (j < pib) {
                    double h = (pb - long_mesh[j]) / SOLVER->pml_long.size;
                    s = 1. + (SOLVER->pml_long.factor - 1.) * pow(h, SOLVER->pml_long.order);
                } else if (j > pif) {
                    double h = (long_mesh[j] - pf) / SOLVER->pml_long.size;
                    s = 1. + (SOLVER->pml_long.factor - 1.) * pow(h, SOLVER->pml_long.order);
                }
                mag_long[i] += Tensor2<dcomplex>(s, 1. / s);
            }
            mag_long[i] /= double(refl);
        }
        // Compute FFT
        FFT::Forward1D(2, nNl, symmetric_long() ? dct_symmetry : FFT::SYMMETRY_NONE)
            .execute(reinterpret_cast<dcomplex*>(mag_long.data()));
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = PI / Ll;
            bb4 *= bb4;  // (2π/L)² / 4
            for (std::size_t i = 0; i != nNl; ++i) {
                int k = int(i);
                if (!symmetric_long() && k > int(nNl / 2)) k -= int(nNl);
                mag_long[i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }
    if (periodic_tran) {
        mag_tran.reset(nNt, Tensor2<dcomplex>(0.));
        mag_tran[0].c00 = 1.;
        mag_tran[0].c11 = 1.;  // constant 1
    } else {
        mag_tran.reset(nNt, Tensor2<dcomplex>(0.));
        double pl = left + SOLVER->pml_tran.size, pr = right - SOLVER->pml_tran.size;
        if (symmetric_tran())
            pil = 0;
        else
            pil = std::lower_bound(tran_mesh.begin(), tran_mesh.end(), pl) - tran_mesh.begin();
        pir = std::lower_bound(tran_mesh.begin(), tran_mesh.end(), pr) - tran_mesh.begin();
        for (size_t i = 0; i != nNt; ++i) {
            for (size_t j = reft * i, end = reft * (i + 1); j != end; ++j) {
                dcomplex s = 1.;
                if (j < pil) {
                    double h = (pl - tran_mesh[j]) / SOLVER->pml_tran.size;
                    s = 1. + (SOLVER->pml_tran.factor - 1.) * pow(h, SOLVER->pml_tran.order);
                } else if (j > pir) {
                    double h = (tran_mesh[j] - pr) / SOLVER->pml_tran.size;
                    s = 1. + (SOLVER->pml_tran.factor - 1.) * pow(h, SOLVER->pml_tran.order);
                }
                mag_tran[i] += Tensor2<dcomplex>(s, 1. / s);
            }
            mag_tran[i] /= double(reft);
        }
        // Compute FFT
        FFT::Forward1D(2, nNt, symmetric_tran() ? dct_symmetry : FFT::SYMMETRY_NONE)
            .execute(reinterpret_cast<dcomplex*>(mag_tran.data()));
        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4 = PI / Lt;
            bb4 *= bb4;  // (2π/L)² / 4
            for (std::size_t i = 0; i != nNt; ++i) {
                int k = int(i);
                if (!symmetric_tran() && k > int(nNt / 2)) k -= int(nNt);
                mag_tran[i] *= exp(-SOLVER->smooth * bb4 * k * k);
            }
        }
    }

    // Allocate memory for expansion coefficients
    size_t nlayers = solver->lcount;
    coeffs.resize(nlayers);
    gradients.resize(nlayers);
    coeffs_ezz.resize(nlayers);
    coeffs_dexx.assign(nlayers, cmatrix());
    coeffs_deyy.assign(nlayers, cmatrix());
    diagonals.assign(nlayers, false);

    mesh = plask::make_shared<RectangularMesh<3>>(plask::make_shared<RegularAxis>(long_mesh),
                                                  plask::make_shared<RegularAxis>(tran_mesh), solver->verts,
                                                  RectangularMesh<3>::ORDER_102);

    initialized = true;
}

void ExpansionPW3D::reset() {
    coeffs.clear();
    coeffs_ezz.clear();
    coeffs_dexx.clear();
    coeffs_deyy.clear();
    gradients.clear();
    initialized = false;
    k0 = klong = ktran = lam0 = NAN;
    mesh.reset();
    temporary.reset();
}

void ExpansionPW3D::beforeLayersIntegrals(dcomplex lam, dcomplex glam) { SOLVER->prepareExpansionIntegrals(this, mesh, lam, glam); }

template <typename T1, typename T2>
inline static Tensor3<decltype(T1() * T2())> commutator(const Tensor3<T1>& A, const Tensor3<T2>& B) {
    return Tensor3<decltype(T1() * T2())>(A.c00 * B.c00 + A.c01 * B.c01, A.c01 * B.c01 + A.c11 * B.c11, A.c22 * B.c22,
                                          0.5 * ((A.c00 + A.c11) * B.c01 + A.c01 * (B.c00 + B.c11)));
}

inline void add_vertex(int l, int t, ExpansionPW3D::Gradient& val, long double& W, const ExpansionPW3D::Gradient::Vertex& vertex) {
    if (l == vertex.l && t == vertex.t) return;
    int dl = l - vertex.l, dt = t - vertex.t;
    long double w = 1. / (dl * dl + dt * dt);
    w *= w;
    val += vertex.val * w;
    W += w;
}

inline double cf(const ExpansionPW3D::Coeff& c) { return c.c00.real() + c.c11.real(); }

void ExpansionPW3D::layerIntegrals(size_t layer, double lam, double glam) {
    auto geometry = SOLVER->getGeometry();

    auto long_mesh = static_pointer_cast<RegularAxis>(mesh->lon()), tran_mesh = static_pointer_cast<RegularAxis>(mesh->tran());

    const double Lt = right - left, Ll = front - back;
    const size_t refl = (SOLVER->refine_long) ? SOLVER->refine_long : 1, reft = (SOLVER->refine_tran) ? SOLVER->refine_tran : 1;
    const size_t nN = nNl * nNt, NN = Nl * Nt, nNl1 = nNl - 1, nNt1 = nNt - 1;
    double normlim = min(Ll / double(nNl), Lt / double(nNt)) * 1e-9;

#if defined(OPENMP_FOUND)  // && !defined(NDEBUG)
    SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {}/{} (sampled at {}x{} points) in thread {}", layer,
                     solver->lcount, refl * nNl, reft * nNt, omp_get_thread_num());
#else
    SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {}/{} (sampled at {}x{} points)", layer, solver->lcount,
                     refl * nNl, reft * nNt);
#endif

    if (isnan(lam)) throw BadInput(SOLVER->getId(), "no wavelength given: specify 'lam' or 'lam0'");

    double matv;
    for (size_t i = 0; i != solver->stack.size(); ++i) {
        if (solver->stack[i] == layer) {
            matv = solver->verts->at(i);
            break;
        }
    }

    if (epsilon_connected && solver->lcomputed[layer]) {
        SOLVER->writelog(LOG_DEBUG, "Layer {:d} takes some materials parameters from inEpsilon", layer);
        if (isnan(glam)) glam = lam;
    }
    if (gain_connected && solver->lgained[layer]) {
        SOLVER->writelog(LOG_DEBUG, "Layer {:d} has gain", layer);
        if (isnan(glam)) glam = lam;
    }

    DataVector<Coeff>& coeffs = this->coeffs[layer];

    // Make space for the result
    coeffs.reset(nN);
    std::fill_n(reinterpret_cast<char*>(coeffs.data()), nN * sizeof(Coeff), 0);

    // Normal cos² ans cos·sin for proper expansion rule
    if (SOLVER->expansion_rule == FourierSolver3D::RULE_COMBINED) {
        gradients[layer].reset(nN, Gradient(NAN, 0.));
    }
    size_t normnans = nN;

    // Average material parameters
    DataVector<Tensor3<dcomplex>> cell(refl * reft);
    double nfact = 1. / double(cell.size());

    double pb = back + SOLVER->pml_long.size, pf = front - SOLVER->pml_long.size;
    double pl = left + SOLVER->pml_tran.size, pr = right - SOLVER->pml_tran.size;

    bool anisotropic = SOLVER->expansion_rule != FourierSolver3D::RULE_INVERSE && !(symmetric_long() || symmetric_tran()),
         nondiagonal = false;

    // We store data for computing gradients
    // TODO: it is possible to progressively store only neighbour cells and thus reduce the memory usage,
    // but it is too hard for now
    std::unique_ptr<double[]> vals_lock;
    std::unique_ptr<TempMatrix> vals_temp_matrix;
    double* vals;
    size_t nMl = refl * nNl, nMt = reft * nNt;
    if (SOLVER->expansion_rule == FourierSolver3D::RULE_COMBINED) {
        if (nMl * nMt > 2 * matrixSize() * matrixSize()) {
            vals_lock.reset(new double[nMl * nMt]);
            vals = vals_lock.get();
        } else {
            size_t N = matrixSize();
            vals_temp_matrix.reset(new TempMatrix(&temporary, N, N));
            vals = reinterpret_cast<double*>(vals_temp_matrix->data());
        }
    }

    for (size_t it = 0; it != nNt; ++it) {
        size_t tbegin = reft * it;
        size_t tend = tbegin + reft;
        double tran0 = 0.5 * (tran_mesh->at(tbegin) + tran_mesh->at(tend - 1));

        size_t cto = nNl * it;

        for (size_t il = 0; il != nNl; ++il) {
            size_t lbegin = refl * il;
            size_t lend = lbegin + refl;
            double long0 = 0.5 * (long_mesh->at(lbegin) + long_mesh->at(lend - 1));

            // Store epsilons for a single cell and compute surface normal (only for old rules)
            Vec<2> norm(0., 0.);
            for (size_t t = tbegin, j = 0; t != tend; ++t) {
                for (size_t l = lbegin; l != lend; ++l, ++j) {
                    std::set<std::string> roles;
                    if (epsilon_connected && solver->lcomputed[layer] || gain_connected && solver->lgained[layer])
                        roles = geometry->getRolesAt(vec(long_mesh->at(l), tran_mesh->at(t), matv));
                    bool computed = epsilon_connected && solver->lcomputed[layer] && roles.find("inEpsilon") != roles.end();
                    if (computed) {
                        double W = 0.;
                        Tensor3<dcomplex> val(0.);
                        for (size_t k = 0, v = mesh->index(l, t, 0); k != mesh->vert()->size(); ++v, ++k) {
                            if (solver->stack[k] == layer) {
                                if (isnan(epsilons[v]))
                                    throw BadInput(solver->getId(), "complex permittivity tensor got from inEpsilon is NaN at {}",
                                                   mesh->at(v));
                                double w = (k == 0 || k == mesh->vert()->size() - 1)
                                               ? 1e-6
                                               : solver->vbounds->at(k) - solver->vbounds->at(k - 1);
                                val += w * epsilons[v];
                                W += w;
                            }
                        }
                        cell[j] = val / W;
                    } else {
                        double T = 0., W = 0., C = 0.;
                        for (size_t k = 0, v = mesh->index(l, t, 0); k != mesh->vert()->size(); ++v, ++k) {
                            if (solver->stack[k] == layer) {
                                double w = (k == 0 || k == mesh->vert()->size() - 1)
                                               ? 1e-6
                                               : solver->vbounds->at(k) - solver->vbounds->at(k - 1);
                                T += w * temperature[v];
                                C += w * carriers[v];
                                W += w;
                            }
                        }
                        T /= W;
                        C /= W;
                        {
                            OmpLockGuard lock;  // this must be declared before `material` to guard its destruction
                            auto material = geometry->getMaterial(vec(long_mesh->at(l), tran_mesh->at(t), matv));
                            lock = material->lock();
                            cell[j] = material->Eps(lam, T, C);
                            if (isnan(cell[j]))
                                throw BadInput(solver->getId(),
                                               "complex permittivity tensor (Eps) for {} is NaN at lam={}nm, T={}K n={}/cm3",
                                               material->name(), lam, T, C);
                        }
                        if (gain_connected && solver->lgained[layer]) {
                            if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() ||
                                roles.find("gain") != roles.end()) {
                                Tensor2<double> g = 0.;
                                W = 0.;
                                for (size_t k = 0, v = mesh->index(l, t, 0); k != mesh->vert()->size(); ++v, ++k) {
                                    if (solver->stack[k] == layer) {
                                        double w = (k == 0 || k == mesh->vert()->size() - 1)
                                                       ? 1e-6
                                                       : solver->vbounds->at(k) - solver->vbounds->at(k - 1);
                                        g += w * gain[v];
                                        W += w;
                                    }
                                }
                                Tensor2<double> ni = glam * g / W * (0.25e-7 / PI);
                                double n00 = sqrt(cell[j].c00).real(), n11 = sqrt(cell[j].c11).real(),
                                       n22 = sqrt(cell[j].c22).real();
                                cell[j].c00 = dcomplex(n00 * n00 - ni.c00 * ni.c00, 2 * n00 * ni.c00);
                                cell[j].c11 = dcomplex(n11 * n11 - ni.c00 * ni.c00, 2 * n11 * ni.c00);
                                cell[j].c22 = dcomplex(n22 * n22 - ni.c11 * ni.c11, 2 * n22 * ni.c11);
                                cell[j].c01.imag(0.);
                                cell[j].c02.imag(0.);
                                cell[j].c10.imag(0.);
                                cell[j].c12.imag(0.);
                                cell[j].c20.imag(0.);
                                cell[j].c21.imag(0.);
                            }
                        }
                    }

                    // Add PMLs
                    if (!periodic_long) {
                        dcomplex s = 1.;
                        if (l < pib) {
                            double h = (pb - long_mesh->at(l)) / SOLVER->pml_long.size;
                            s = 1. + (SOLVER->pml_long.factor - 1.) * pow(h, SOLVER->pml_long.order);
                        } else if (l > pif) {
                            double h = (long_mesh->at(l) - pf) / SOLVER->pml_long.size;
                            s = 1. + (SOLVER->pml_long.factor - 1.) * pow(h, SOLVER->pml_long.order);
                        }
                        cell[j].c00 *= 1. / s;
                        cell[j].c11 *= s;
                        cell[j].c22 *= s;
                    }
                    if (!periodic_tran) {
                        dcomplex s = 1.;
                        if (t < pil) {
                            double h = (pl - tran_mesh->at(t)) / SOLVER->pml_tran.size;
                            s = 1. + (SOLVER->pml_tran.factor - 1.) * pow(h, SOLVER->pml_tran.order);
                        } else if (t > pir) {
                            double h = (tran_mesh->at(t) - pr) / SOLVER->pml_tran.size;
                            s = 1. + (SOLVER->pml_tran.factor - 1.) * pow(h, SOLVER->pml_tran.order);
                        }
                        cell[j].c00 *= s;
                        cell[j].c11 *= 1. / s;
                        cell[j].c22 *= s;
                    }

                    if (SOLVER->expansion_rule == FourierSolver3D::RULE_COMBINED)
                        vals[nMl * t + l] = abs(real(cell[j].c00)) + abs(real(cell[j].c11));
                    else
                        norm += (real(cell[j].c00) + real(cell[j].c11)) * vec(long_mesh->at(l) - long0, tran_mesh->at(t) - tran0);
                }
            }

            Tensor3<dcomplex> eps(0.);
            Coeff& coeff = coeffs[cto + il];

            // Compute avg(eps)
            for (size_t t = tbegin, j = 0; t != tend; ++t) {
                for (size_t l = lbegin; l != lend; ++l, ++j) {
                    eps += cell[j];
                }
            }
            eps *= nfact;
            if (SOLVER->expansion_rule == FourierSolver3D::RULE_OLD && eps.c22 != 0.) eps.c22 = 1. / eps.c22;

            double a = abs(norm);
            if (a >= normlim && SOLVER->expansion_rule != FourierSolver3D::RULE_COMBINED) {
                norm /= a;

                // Compute avg(eps**(-1))
                Tensor3<dcomplex> ieps(0.);
                for (size_t t = tbegin, j = 0; t != tend; ++t) {
                    for (size_t l = lbegin; l != lend; ++l, ++j) {
                        ieps += cell[j].inv();
                    }
                }
                ieps *= nfact;
                // Average permittivity tensor according to:
                // [ S. G. Johnson and J. D. Joannopoulos, Opt. Express, vol. 8, pp. 173-190 (2001) ]
                Tensor3<double> P(norm.c0 * norm.c0, norm.c1 * norm.c1, 0., norm.c0 * norm.c1);
                Tensor3<double> P1(1. - P.c00, 1. - P.c11, 1., -P.c01);
                eps = commutator(P, ieps.inv()) + commutator(P1, eps);
            }

            if (!is_zero(eps.c00 - eps.c11)) anisotropic = true;
            if (!is_zero(eps.c01)) {
                if (!is_zero(eps.c01 - conj(eps.c10)))
                    throw BadInput(solver->getId(), "complex permittivity tensor (Eps) must be Hermitian for this solver");
                nondiagonal = true;
            }

            coeff = eps;
        }
    }

    // Compute surface normals using larger cell sizes
    if (SOLVER->expansion_rule == FourierSolver3D::RULE_COMBINED) {
        // First increase tolerance to avoid artifacts
        double vmax = 0., vmin = std::numeric_limits<double>::max();
        double* v = vals;
        for (size_t i = 0, nM = nMl * nMt; i < nM; ++i, ++v) {
            if (*v > vmax) vmax = *v;
            if (*v < vmin) vmin = *v;
        }
        if (vmax != vmin) normlim *= 0.1 * (vmax - vmin);

        bool nowrapl = !periodic_long || symmetric_long(), nowrapt = !periodic_tran || symmetric_tran();

        for (size_t it = 0; it != nNt; ++it) {
            std::ptrdiff_t tbegin, tend;
            if (it == 0 && nowrapt) {
                tbegin = 0;
                tend = tbegin + reft;
            } else if (it == nNt - 1 && nowrapt) {
                tbegin = reft * it - reft / 2;
                tend = nMt;
            } else {
                tbegin = reft * it - reft / 2;
                tend = tbegin + 2 * reft;
            }
            double tran0 = tran_mesh->first() + 0.5 * (tbegin + tend - 1) * tran_mesh->step();

            for (size_t il = 0; il != nNl; ++il) {
                std::ptrdiff_t lbegin, lend;
                if (il == 0 && nowrapl) {
                    lbegin = 0;
                    lend = lbegin + refl + refl / 2;
                } else if (il == nNl - 1 && nowrapl) {
                    lbegin = refl * il - refl / 2;
                    lend = nMl;
                } else {
                    lbegin = refl * il - refl / 2;
                    lend = lbegin + 2 * refl;
                }
                double long0 = long_mesh->first() + 0.5 * (lbegin + lend - 1) * long_mesh->step();

                Vec<2> norm(0., 0.);
                for (std::ptrdiff_t jt = tbegin; jt != tend; ++jt) {
                    size_t t = (jt < 0) ? jt + nMt : (jt >= nMt) ? jt - nMt : jt;
                    for (std::ptrdiff_t jl = lbegin; jl != lend; ++jl) {
                        size_t l = (jl < 0) ? jl + nMl : (jl >= nMl) ? jl - nMl : jl;
                        norm += vals[nMl * t + l] * vec(long_mesh->at(l) - long0, tran_mesh->at(t) - tran0);
                    }
                }

                double a = abs(norm);
                if (a >= normlim) {
                    gradients[layer][nNl * it + il] = norm / a;
                    --normnans;
                }
            }
        }
        vals_temp_matrix.reset();  // probably not really necessary
    }

    // int nn = 0;
    // for (size_t il = 0; il < nNl; ++il) {
    //     for (size_t it = 0; it < nNl; ++it) {
    //         double val = gradients[layer][nNl * it + il].cs.real();
    //         if (isnan(val)) nn++;
    //         std::cerr << str(val, "{:4.1f} ");
    //     }
    //     std::cerr << "\n";
    // }
    // std::cerr << ":" << normnans << "/" << nn << "\n";

    // Check if the layer is uniform
    if (periodic_tran && periodic_long && !nondiagonal) {
        diagonals[layer] = true;
        for (size_t i = 1; i != nN; ++i) {
            if (coeffs[i].c00 != coeffs[i].c22 || coeffs[i].c11 != coeffs[i].c22 || coeffs[i].differs(coeffs[0])) {
                diagonals[layer] = false;
                break;
            }
        }
    } else {
        diagonals[layer] = false;
    }

    if (diagonals[layer]) {
        SOLVER->writelog(LOG_DETAIL, "Layer {0} is uniform", layer);
        std::fill_n(reinterpret_cast<dcomplex*>(coeffs.data() + 1), 6 * (coeffs.size() - 1), 0.);
    } else {
        // Perform FFT
        if (SOLVER->expansion_rule == FourierSolver3D::RULE_INVERSE) {
            matFFT.execute(reinterpret_cast<dcomplex*>(coeffs.data()), 1);
            matFFT.execute(reinterpret_cast<dcomplex*>(coeffs.data()) + 2, 1);
            if (anisotropic) {
                matFFT.execute(reinterpret_cast<dcomplex*>(coeffs.data()) + 4, 1);
            } else {
                for (size_t i = 0; i != nN; ++i) coeffs[i].ic11 = coeffs[i].ic00;
            }
        } else if (SOLVER->expansion_rule == FourierSolver3D::RULE_COMBINED) {
            if (anisotropic) {
                matFFT.execute(reinterpret_cast<dcomplex*>(coeffs.data()), 5);
            } else {
                matFFT.execute(reinterpret_cast<dcomplex*>(coeffs.data()), 3);
                for (size_t i = 0; i != nN; ++i) {
                    coeffs[i].c11 = coeffs[i].c00;
                    coeffs[i].ic11 = coeffs[i].ic00;
                }
            }
        } else {
            matFFT.execute(reinterpret_cast<dcomplex*>(coeffs.data()), 2);
            if (anisotropic) {
                matFFT.execute(reinterpret_cast<dcomplex*>(coeffs.data()) + 3, 1);
            } else {
                for (size_t i = 0; i != nN; ++i) coeffs[i].c11 = coeffs[i].c00;
            }
        }
        if (nondiagonal) matFFT.execute(reinterpret_cast<dcomplex*>(coeffs.data()) + 5, 1);

        // Smooth coefficients
        if (SOLVER->smooth) {
            double bb4l = PI / ((front - back) * (symmetric_long() ? 2 : 1));
            bb4l *= bb4l;  // (2π/Ll)² / 4
            double bb4t = PI / ((right - left) * (symmetric_tran() ? 2 : 1));
            bb4t *= bb4t;  // (2π/Lt)² / 4
            for (std::size_t it = 0; it != nNt; ++it) {
                int kt = int(it);
                if (!symmetric_tran() && kt > int(nNt / 2)) kt -= int(nNt);
                for (std::size_t il = 0; il != nNl; ++il) {
                    int kl = int(il);
                    if (!symmetric_long() && kl > int(nNl / 2)) kl -= int(nNl);
                    coeffs[nNl * it + il] *= exp(-SOLVER->smooth * (bb4l * kl * kl + bb4t * kt * kt));
                }
            }
        }

        if (SOLVER->expansion_rule != FourierSolver3D::RULE_OLD) {
            TempMatrix tempMatrix = getTempMatrix();
            cmatrix work1(NN, NN, tempMatrix.data());
            cmatrix work2(NN, NN, tempMatrix.data() + NN * NN);

            int ordl = int(SOLVER->getLongSize()), ordt = int(SOLVER->getTranSize());

            char symx = char(symmetric_long() ? 2 * int(symmetry_long) - 3 : 0),
                 symy = char(symmetric_tran() ? 2 * int(symmetry_tran) - 3 : 0);
            // +1: Ex+, Ey-, Hx-, Hy+
            //  0: no symmetry
            // -1: Ex-, Ey+, Hx+, Hy-

            makeToeplitzMatrix(work1, ordl, ordt, layer, 0, -symx, symy);
            coeffs_ezz[layer].reset(NN, NN);
            make_unit_matrix(coeffs_ezz[layer]);
            invmult(work1, coeffs_ezz[layer]);

            if (SOLVER->expansion_rule == FourierSolver3D::RULE_INVERSE) {
                makeToeplitzMatrix(work1, ordl, ordt, layer, 2, symx, symy);
                coeffs_dexx[layer].reset(NN, NN);
                make_unit_matrix(coeffs_dexx[layer]);
                invmult(work1, coeffs_dexx[layer]);
                if (anisotropic) {
                    makeToeplitzMatrix(work1, ordl, ordt, layer, 4, -symx, -symy);
                    coeffs_deyy[layer].reset(NN, NN);
                    make_unit_matrix(coeffs_deyy[layer]);
                    invmult(work1, coeffs_deyy[layer]);
                } else {
                    coeffs_deyy[layer] = coeffs_dexx[layer];
                }

            } else if (SOLVER->expansion_rule == FourierSolver3D::RULE_COMBINED) {
                // Fill gaps in cos² and cos·sin matrices
                if (nN == normnans)
                    throw ComputationError(SOLVER->getId(), "cannot compute normals - consider changing expansion size");
                std::vector<Gradient::Vertex> vertices;
                vertices.reserve(nN - normnans);
                size_t i = 0;
                for (int t = 0; t < nNt; ++t) {
                    for (int l = 0; l < nNl; ++l, ++i) {
                        if (!gradients[layer][i].isnan()) vertices.emplace_back(l, t, gradients[layer][i]);
                    }
                }
                i = 0;
                for (int t = 0; t < nNt; ++t) {
                    for (int l = 0; l < nNl; ++l, ++i) {
                        if (gradients[layer][i].isnan()) {
                            Gradient val(0., 0.);
                            long double W = 0.;
                            for (const Gradient::Vertex& v : vertices) {
                                add_vertex(l, t, val, W, v);
                                if (symmetric_long()) {
                                    add_vertex(l, t, val, W, Gradient::Vertex(-v.l, v.t, v.val));
                                    if (periodic_long) add_vertex(l, t, val, W, Gradient::Vertex(2 * nNl - v.l, v.t, v.val));
                                } else if (periodic_long) {
                                    add_vertex(l, t, val, W, Gradient::Vertex(v.l + nNl, v.t, v.val));
                                    add_vertex(l, t, val, W, Gradient::Vertex(v.l - nNl, v.t, v.val));
                                }
                                if (symmetric_tran()) {
                                    add_vertex(l, t, val, W, Gradient::Vertex(v.l, -v.t, v.val));
                                    if (periodic_tran) add_vertex(l, t, val, W, Gradient::Vertex(v.l, 2 * nNt - v.t, v.val));
                                } else if (periodic_tran) {
                                    add_vertex(l, t, val, W, Gradient::Vertex(v.l, v.t + nNt, v.val));
                                    add_vertex(l, t, val, W, Gradient::Vertex(v.l, v.t - nNt, v.val));
                                }
                            }
                            gradients[layer][i] = val / W;
                        }
                    }
                }
                for (auto& grad : gradients[layer]) {
                    double c2 = grad.c2.real();
                    if (c2 > 1.) {
                        grad.c2 = c2 = 1.;
                    }  // just for safety
                    double cs = sqrt(c2 - c2 * c2);
                    if (grad.cs.real() >= 0)
                        grad.cs = cs;
                    else
                        grad.cs = -cs;
                }

                // for (size_t il = 0; il < nNl; ++il) {
                //     for (size_t it = 0; it < nNt; ++it) {
                //         auto val = gradients[layer][nNl * it + il];
                //         std::cerr << str(val.c2, "{:.6f}{:+.6f}j/") << str(val.cs, "{:.6f}{:+.6f}j ");
                //     }
                // }
                // std::cerr << "\n\n";

                cos2FFT.execute(reinterpret_cast<dcomplex*>(gradients[layer].data()), 1);
                cssnFFT.execute(reinterpret_cast<dcomplex*>(gradients[layer].data()) + 1, 1);

                // makeToeplitzMatrix(work1, work2, gradients[layer], ordl, ordt, symx, symy);
                // for (size_t r = 0; r != work1.rows(); ++r) {
                //     for (size_t c = 0; c != work1.cols(); ++c)
                //         std::cerr << str(work1(r,c), "{:7.4f}{:+7.4f}j ");
                //     std::cerr << "\n";
                // }
                // std::cerr << "\n";
                // for (size_t r = 0; r != work2.rows(); ++r) {
                //     for (size_t c = 0; c != work2.cols(); ++c)
                //         std::cerr << str(work2(r,c), "{:7.4f}{:+7.4f}j ");
                //     std::cerr << "\n";
                // }

                // Smooth coefficients
                if (SOLVER->grad_smooth) {
                    double bb4l = PI / ((front - back) * (symmetric_long() ? 2 : 1));
                    bb4l *= bb4l;  // (2π/Ll)² / 4
                    double bb4t = PI / ((right - left) * (symmetric_tran() ? 2 : 1));
                    bb4t *= bb4t;  // (2π/Lt)² / 4
                    for (std::size_t it = 0; it != nNt; ++it) {
                        int kt = int(it);
                        if (!symmetric_tran() && kt > int(nNt / 2)) kt -= int(nNt);
                        for (std::size_t il = 0; il != nNl; ++il) {
                            int kl = int(il);
                            if (!symmetric_long() && kl > int(nNl / 2)) kl -= int(nNl);
                            gradients[layer][nNl * it + il] *= exp(-SOLVER->grad_smooth * (bb4l * kl * kl + bb4t * kt * kt));
                        }
                    }
                }

                makeToeplitzMatrix(work1, ordl, ordt, layer, 2, symx, symy);
                // for (size_t r = 0; r < NN; ++r) {
                //     for (size_t c = 0; c < NN; ++c) {
                //         std::cerr << str(work1(r,c), "{:9.6f}{:+9.6f}j") << "  ";
                //     }
                //     std::cerr << "\n";
                // }
                // std::cerr << "\n";
                coeffs_dexx[layer].reset(NN, NN);
                make_unit_matrix(coeffs_dexx[layer]);
                invmult(work1, coeffs_dexx[layer]);
                addToeplitzMatrix(coeffs_dexx[layer], ordl, ordt, layer, 1, symx, symy, -1.);

                if (anisotropic || !(symmetric_long() && symmetric_tran())) {
                    makeToeplitzMatrix(work1, ordl, ordt, layer, 4, -symx, -symy);
                    coeffs_deyy[layer].reset(NN, NN);
                    make_unit_matrix(coeffs_deyy[layer]);
                    invmult(work1, coeffs_deyy[layer]);
                    addToeplitzMatrix(coeffs_deyy[layer], ordl, ordt, layer, 3, -symx, -symy, -1.);
                } else {
                    coeffs_deyy[layer] = coeffs_dexx[layer];
                }
            } else {
                coeffs_dexx[layer].reset();
                coeffs_deyy[layer].reset();
            }
        }
    }
}

LazyData<Tensor3<dcomplex>> ExpansionPW3D::getMaterialEps(size_t lay,
                                                          const shared_ptr<const typename LevelsAdapter::Level>& level,
                                                          InterpolationMethod interp) {
    assert(dynamic_pointer_cast<const MeshD<3>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<3>>(level->mesh());

    if (interp == INTERPOLATION_DEFAULT || interp == INTERPOLATION_FOURIER) {
        return LazyData<Tensor3<dcomplex>>(dest_mesh->size(), [this, lay, dest_mesh](size_t i) -> Tensor3<dcomplex> {
            Tensor3<dcomplex> eps(0.);
            const int nt = symmetric_tran() ? int(nNt) - 1 : int(nNt / 2), nl = symmetric_long() ? int(nNl) - 1 : int(nNl / 2);
            double Lt = right - left;
            if (symmetric_tran()) Lt *= 2;
            double Ll = front - back;
            if (symmetric_long()) Ll *= 2;
            for (int kt = -nt; kt <= nt; ++kt) {
                size_t t = (kt >= 0) ? kt : (symmetric_tran()) ? -kt : kt + nNt;
                const double phast = kt * (dest_mesh->at(i).c1 - left) / Lt;
                for (int kl = -nl; kl <= nl; ++kl) {
                    size_t l = (kl >= 0) ? kl : (symmetric_long()) ? -kl : kl + nNl;
                    const double phasl = kl * (dest_mesh->at(i).c0 - back) / Ll;
                    if (SOLVER->expansion_rule != FourierSolver3D::RULE_INVERSE)
                        eps += Tensor3<dcomplex>(coeffs[lay][nNl * t + l]) * exp(2 * PI * I * (phasl + phast));
                    else
                        eps += coeffs[lay][nNl * t + l].toInverseTensor() * exp(2 * PI * I * (phasl + phast));
                }
            }
            if (SOLVER->expansion_rule == FourierSolver3D::RULE_OLD) {
                eps.c22 = 1. / eps.c22;
            } else if (SOLVER->expansion_rule == FourierSolver3D::RULE_INVERSE) {
                eps.c00 = 1. / eps.c00;
                eps.c11 = 1. / eps.c11;
            }
            return eps;
        });
    } else {
        size_t nl = symmetric_long() ? nNl : nNl + 1, nt = symmetric_tran() ? nNt : nNt + 1;
        DataVector<Tensor3<dcomplex>> params(nl * nt);
        for (size_t t = 0; t != nNt; ++t) {
            size_t op = nl * t, oc = nNl * t;
            if (SOLVER->expansion_rule != FourierSolver3D::RULE_INVERSE)
                for (size_t l = 0; l != nNl; ++l) params[op + l] = coeffs[lay][oc + l];
            else
                for (size_t l = 0; l != nNl; ++l) params[op + l] = coeffs[lay][oc + l].toInverseTensor();
        }
        auto dct_symmetry = SOLVER->dct2() ? FFT::SYMMETRY_EVEN_2 : FFT::SYMMETRY_EVEN_1;
        FFT::Backward2D(9, nNl, nNt, symmetric_long() ? dct_symmetry : FFT::SYMMETRY_NONE,
                        symmetric_tran() ? dct_symmetry : FFT::SYMMETRY_NONE, nl)
            .execute(reinterpret_cast<dcomplex*>(params.data()));
        shared_ptr<RegularAxis> lcmesh = plask::make_shared<RegularAxis>(), tcmesh = plask::make_shared<RegularAxis>();
        if (symmetric_long()) {
            if (SOLVER->dct2()) {
                double dx = 0.5 * (front - back) / double(nl);
                lcmesh->reset(back + dx, front - dx, nl);
            } else {
                lcmesh->reset(0., front, nl);
            }
        } else {
            lcmesh->reset(back, front, nl);
            for (size_t t = 0, end = nl * nt, dist = nl - 1; t != end; t += nl) params[dist + t] = params[t];
        }
        if (symmetric_tran()) {
            if (SOLVER->dct2()) {
                double dy = 0.5 * right / double(nt);
                tcmesh->reset(dy, right - dy, nt);
            } else {
                tcmesh->reset(0., right, nt);
            }
        } else {
            tcmesh->reset(left, right, nt);
            for (size_t l = 0, last = nl * (nt - 1); l != nl; ++l) params[last + l] = params[l];
        }
        for (Tensor3<dcomplex>& eps : params) {
            if (SOLVER->expansion_rule == FourierSolver3D::RULE_OLD) {
                eps.c22 = 1. / eps.c22;
            } else if (SOLVER->expansion_rule == FourierSolver3D::RULE_INVERSE) {
                eps.c00 = 1. / eps.c00;
                eps.c11 = 1. / eps.c11;
            }
        }
        auto src_mesh = plask::make_shared<RectangularMesh<3>>(
            lcmesh, tcmesh, plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1), RectangularMesh<3>::ORDER_210);
        return interpolate(
            src_mesh, params, dest_mesh, interp,
            InterpolationFlags(SOLVER->getGeometry(),
                               symmetric_long() ? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                               symmetric_tran() ? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                               InterpolationFlags::Symmetry::POSITIVE));
    }
}

void ExpansionPW3D::getMatrices(size_t lay, cmatrix& RE, cmatrix& RH) {
    assert(initialized);
    if (isnan(k0)) throw BadInput(SOLVER->getId(), "wavelength or k0 not set");
    if (isinf(k0.real())) throw BadInput(SOLVER->getId(), "wavelength must not be 0");

    bool diagonal = diagonals[lay];

    int ordl = int(SOLVER->getLongSize()), ordt = int(SOLVER->getTranSize());

    char symx = char(symmetric_long() ? 2 * int(symmetry_long) - 3 : 0),
         symy = char(symmetric_tran() ? 2 * int(symmetry_tran) - 3 : 0);
    // +1: Ex+, Ey-, Hx-, Hy+
    //  0: no symmetry
    // -1: Ex-, Ey+, Hx+, Hy-

    assert(!(symx && klong != 0.));
    assert(!(symy && ktran != 0.));

    assert(!isnan(k0.real()) && !isnan(k0.imag()));

    double Gx = 2. * PI / (front - back) * (symx ? 0.5 : 1.), Gy = 2. * PI / (right - left) * (symy ? 0.5 : 1.);

    dcomplex ik0 = 1. / k0;

    const size_t N = Nl * Nt;
    assert((symx ? ordl + 1 : 2 * ordl + 1) == Nl);
    assert((symy ? ordt + 1 : 2 * ordt + 1) == Nt);

    if (SOLVER->expansion_rule == FourierSolver3D::RULE_COMBINED && !coeffs_dexx[lay].empty() && !diagonal) {
        TempMatrix tempMatrix = getTempMatrix();
        cmatrix workxx(N, N, tempMatrix.data());
        cmatrix workxy(N, N, tempMatrix.data() + N * N);
        cmatrix workyx(N, N, tempMatrix.data() + 2 * N * N);
        cmatrix workyy(N, N, tempMatrix.data() + 3 * N * N);
        cmatrix workc2(N, N, RE.data());
        cmatrix workcs(N, N, RE.data() + N * N);

        makeToeplitzMatrix(workc2, workcs, gradients[lay], ordl, ordt, symx, symy);

        mult_matrix_by_matrix(coeffs_dexx[lay], workc2, workxx);
        mult_matrix_by_matrix(coeffs_dexx[lay], workcs, workxy);
        if (!(symmetric_long() || symmetric_tran())) {
            if (coeffs_deyy[lay].data() != coeffs_dexx[lay].data()) {
                mult_matrix_by_matrix(coeffs_deyy[lay], workc2, workyy);
                mult_matrix_by_matrix(coeffs_deyy[lay], workcs, workyx);
            } else {
                workyy = workxx;
                workyx = workxy;
            }
        } else {
            makeToeplitzMatrix(workc2, workcs, gradients[lay], ordl, ordt, -symx, -symy);
            mult_matrix_by_matrix(coeffs_deyy[lay], workc2, workyy);
            mult_matrix_by_matrix(coeffs_deyy[lay], workcs, workyx);
        }

        zero_matrix(RE);
        zero_matrix(RH);

        for (int iy = (symy ? 0 : -ordt); iy <= ordt; ++iy) {
            dcomplex gy = iy * Gy - ktran;
            for (int ix = (symx ? 0 : -ordl); ix <= ordl; ++ix) {
                dcomplex gx = ix * Gx - klong;
                size_t iex = iEx(ix, iy), iey = iEy(ix, iy);
                size_t ihx = iHx(ix, iy), ihy = iHy(ix, iy);
                size_t i = idx(ix, iy);

                for (int jy = -ordt; jy <= ordt; ++jy) {
                    dcomplex py = jy * Gy - ktran;
                    int ijy = iy - jy;
                    if (symy && ijy < 0) ijy = -ijy;
                    for (int jx = -ordl; jx <= ordl; ++jx) {
                        bool toeplitz = true;
                        dcomplex px = jx * Gx - klong;
                        int ijx = ix - jx;
                        if (symx && ijx < 0) ijx = -ijx;
                        size_t jex = iEx(jx, jy), jey = iEy(jx, jy);
                        size_t jhx = iHx(jx, jy), jhy = iHy(jx, jy);
                        double fx = 1., fy = 1.;
                        if (jx < 0 && symx) {
                            fx *= symx;
                            fy *= -symx;
                            toeplitz = false;
                        }
                        if (jy < 0 && symy) {
                            fx *= symy;
                            fy *= -symy;
                            toeplitz = false;
                        }
                        RH(iex, jhy) += fx * k0 * muyy(lay, ijx, ijy);
                        RH(iey, jhx) += fy * k0 * muxx(lay, ijx, ijy);
                        dcomplex imu = imuzz(lay, ijx, ijy) * ik0;
                        RE(ihy, jex) += fx * (-gy * py * imu + k0 * epsxx(lay, ijx, ijy));
                        RE(ihx, jex) += fx * (gx * py * imu + k0 * epsyx(lay, ijx, ijy));
                        RE(ihy, jey) += fy * (gy * px * imu + k0 * epsxy(lay, ijx, ijy));
                        RE(ihx, jey) += fy * (-gx * px * imu + k0 * epsyy(lay, ijx, ijy));
                        if (toeplitz) {
                            size_t j = idx(jx, jy);
                            dcomplex iepszz = coeffs_ezz[lay](i, j) * ik0;
                            RH(iex, jhy) += fx * (-gx * px * iepszz);
                            RH(iey, jhy) += fx * (-gy * px * iepszz);
                            RH(iex, jhx) += fy * (-gx * py * iepszz);
                            RH(iey, jhx) += fy * (-gy * py * iepszz);
                            RE(ihy, jex) += fx * (k0 * workxx(i, j));
                            RE(ihx, jex) += fx * (k0 * conj(workyx(i, j)));  // Should this conjugate be here?
                            RE(ihy, jey) += fy * (k0 * workxy(i, j));
                            RE(ihx, jey) += fy * (k0 * (coeffs_deyy[lay](i, j) - workyy(i, j)));  // ([[1]]-[[c²]]) [[Δε]]
                        }
                    }
                }

                // Ugly hack to avoid singularity
                if (RE(iex, iex) == 0.) RE(iex, iex) = 1e-32;
                if (RE(iey, iey) == 0.) RE(iey, iey) = 1e-32;
                if (RH(ihx, ihx) == 0.) RH(ihx, ihx) = 1e-32;
                if (RH(ihy, ihy) == 0.) RH(ihy, ihy) = 1e-32;
            }
        }

    } else if (SOLVER->expansion_rule == FourierSolver3D::RULE_INVERSE && !coeffs_dexx[lay].empty() && !diagonal) {
        zero_matrix(RE);
        zero_matrix(RH);

        for (int iy = (symy ? 0 : -ordt); iy <= ordt; ++iy) {
            dcomplex gy = iy * Gy - ktran;
            for (int ix = (symx ? 0 : -ordl); ix <= ordl; ++ix) {
                dcomplex gx = ix * Gx - klong;
                size_t iex = iEx(ix, iy), iey = iEy(ix, iy);
                size_t ihx = iHx(ix, iy), ihy = iHy(ix, iy);
                size_t i = idx(ix, iy);

                for (int jy = -ordt; jy <= ordt; ++jy) {
                    dcomplex py = jy * Gy - ktran;
                    int ijy = iy - jy;
                    if (symy && ijy < 0) ijy = -ijy;
                    for (int jx = -ordl; jx <= ordl; ++jx) {
                        bool toeplitz = true;
                        dcomplex px = jx * Gx - klong;
                        int ijx = ix - jx;
                        if (symx && ijx < 0) ijx = -ijx;
                        size_t jex = iEx(jx, jy), jey = iEy(jx, jy);
                        size_t jhx = iHx(jx, jy), jhy = iHy(jx, jy);
                        double fx = 1., fy = 1.;
                        if (symx && jx < 0) {
                            fx *= symx;
                            fy *= -symx;
                            toeplitz = false;
                        }
                        if (symy && jy < 0) {
                            fx *= symy;
                            fy *= -symy;
                            toeplitz = false;
                        }
                        RH(iex, jhy) += fx * k0 * muyy(lay, ijx, ijy);
                        RH(iey, jhx) += fy * k0 * muxx(lay, ijx, ijy);
                        dcomplex imu = imuzz(lay, ijx, ijy) * ik0;
                        RE(ihy, jex) += fx * (-gy * py * imu);
                        RE(ihx, jex) += fx * (gx * py * imu + k0 * epsyx(lay, ijx, ijy));
                        RE(ihy, jey) += fy * (gy * px * imu + k0 * epsxy(lay, ijx, ijy));
                        RE(ihx, jey) += fy * (-gx * px * imu);
                        if (toeplitz) {
                            size_t j = idx(jx, jy);
                            dcomplex iepszz = coeffs_ezz[lay](i, j) * ik0;
                            RH(iex, jhy) += fx * (-gx * px * iepszz);
                            RH(iey, jhy) += fx * (-gy * px * iepszz);
                            RH(iex, jhx) += fy * (-gx * py * iepszz);
                            RH(iey, jhx) += fy * (-gy * py * iepszz);
                            RE(ihy, jex) += fx * (k0 * coeffs_dexx[lay](i, j));
                            RE(ihx, jey) += fy * (k0 * coeffs_deyy[lay](i, j));
                        }
                    }
                }

                // Ugly hack to avoid singularity
                if (RE(iex, iex) == 0.) RE(iex, iex) = 1e-32;
                if (RE(iey, iey) == 0.) RE(iey, iey) = 1e-32;
                if (RH(ihx, ihx) == 0.) RH(ihx, ihx) = 1e-32;
                if (RH(ihy, ihy) == 0.) RH(ihy, ihy) = 1e-32;
            }
        }

    } else {
        zero_matrix(RE);
        zero_matrix(RH);

        for (int iy = (symy ? 0 : -ordt); iy <= ordt; ++iy) {
            dcomplex gy = iy * Gy - ktran;
            for (int ix = (symx ? 0 : -ordl); ix <= ordl; ++ix) {
                dcomplex gx = ix * Gx - klong;
                size_t iex = iEx(ix, iy), iey = iEy(ix, iy);
                size_t ihx = iHx(ix, iy), ihy = iHy(ix, iy);

                for (int jy = -ordt; jy <= ordt; ++jy) {
                    dcomplex py = jy * Gy - ktran;
                    int ijy = iy - jy;
                    if (symy && ijy < 0) ijy = -ijy;
                    for (int jx = -ordl; jx <= ordl; ++jx) {
                        dcomplex px = jx * Gx - klong;
                        int ijx = ix - jx;
                        if (symx && ijx < 0) ijx = -ijx;
                        size_t jex = iEx(jx, jy), jey = iEy(jx, jy);
                        size_t jhx = iHx(jx, jy), jhy = iHy(jx, jy);
                        double fx = 1., fy = 1.;
                        if (symx && jx < 0) {
                            fx *= symx;
                            fy *= -symx;
                        }
                        if (symy && jy < 0) {
                            fx *= symy;
                            fy *= -symy;
                        }
                        dcomplex ieps = ((SOLVER->expansion_rule == FourierSolver3D::RULE_OLD) ? iepszz(lay, ijx, ijy)
                                         : diagonal ? ((ix == jx && iy == jy) ? 1. / coeffs[lay][0].c22 : 0.)
                                                    : iepszz(lay, ix, jx, iy, jy)) *
                                        ik0;
                        RH(iex, jhy) += fx * (-gx * px * ieps + k0 * muyy(lay, ijx, ijy));
                        RH(iey, jhy) += fx * (-gy * px * ieps);
                        RH(iex, jhx) += fy * (-gx * py * ieps);
                        RH(iey, jhx) += fy * (-gy * py * ieps + k0 * muxx(lay, ijx, ijy));
                        dcomplex imu = imuzz(lay, ijx, ijy) * ik0;
                        RE(ihy, jex) += fx * (-gy * py * imu + k0 * epsxx(lay, ijx, ijy));
                        RE(ihx, jex) += fx * (gx * py * imu + k0 * epsyx(lay, ijx, ijy));
                        RE(ihy, jey) += fy * (gy * px * imu + k0 * epsxy(lay, ijx, ijy));
                        RE(ihx, jey) += fy * (-gx * px * imu + k0 * epsyy(lay, ijx, ijy));
                    }
                }

                // Ugly hack to avoid singularity
                if (RE(iex, iex) == 0.) RE(iex, iex) = 1e-32;
                if (RE(iey, iey) == 0.) RE(iey, iey) = 1e-32;
                if (RH(ihx, ihx) == 0.) RH(ihx, ihx) = 1e-32;
                if (RH(ihy, ihy) == 0.) RH(ihy, ihy) = 1e-32;
            }
        }
    }

    assert(!RE.isnan());
    assert(!RH.isnan());
}

void ExpansionPW3D::prepareField() {
    if (field_interpolation == INTERPOLATION_DEFAULT) field_interpolation = INTERPOLATION_FOURIER;
    if (symmetric_long() || symmetric_tran()) {
        Component syml = (which_field == FIELD_E || !symmetry_long) ? symmetry_long : Component(3 - symmetry_long),
                  symt = (which_field == FIELD_E || !symmetry_tran) ? symmetry_tran : Component(3 - symmetry_tran);
        size_t nl = (syml == E_UNSPECIFIED) ? Nl + 1 : Nl;
        size_t nt = (symt == E_UNSPECIFIED) ? Nt + 1 : Nt;
        if (field_interpolation != INTERPOLATION_FOURIER) {
            int df = SOLVER->dct2() ? 0 : 4;
            FFT::Symmetry x1, xz2, yz1, y2;
            if (symmetric_long()) {
                x1 = FFT::Symmetry(3 - syml + df);
                yz1 = FFT::Symmetry(syml + df);
            } else {
                x1 = yz1 = FFT::SYMMETRY_NONE;
            }
            if (symmetric_tran()) {
                xz2 = FFT::Symmetry(3 - symt + df);
                y2 = FFT::Symmetry(symt + df);
            } else {
                xz2 = y2 = FFT::SYMMETRY_NONE;
            }
            fft_x = FFT::Backward2D(3, Nl, Nt, x1, xz2, nl);
            fft_y = FFT::Backward2D(3, Nl, Nt, yz1, y2, nl);
            fft_z = FFT::Backward2D(3, Nl, Nt, yz1, xz2, nl);
        }
        field.reset(nl * nt);
    } else {
        if (field_interpolation != INTERPOLATION_FOURIER)
            fft_z = FFT::Backward2D(3, Nl, Nt, FFT::SYMMETRY_NONE, FFT::SYMMETRY_NONE, Nl + 1);
        field.reset((Nl + 1) * (Nt + 1));
    }
}

void ExpansionPW3D::cleanupField() {
    field.reset();
    fft_x = FFT::Backward2D();
    fft_y = FFT::Backward2D();
    fft_z = FFT::Backward2D();
}

// TODO fields must be carefully verified

LazyData<Vec<3, dcomplex>> ExpansionPW3D::getField(size_t l,
                                                   const shared_ptr<const typename LevelsAdapter::Level>& level,
                                                   const cvector& E,
                                                   const cvector& H) {
    Component syml = (which_field == FIELD_E) ? symmetry_long : Component((3 - symmetry_long) % 3);
    Component symt = (which_field == FIELD_E) ? symmetry_tran : Component((3 - symmetry_tran) % 3);

    bool diagonal = diagonals[l];

    size_t nl = (syml == E_UNSPECIFIED) ? Nl + 1 : Nl, nt = (symt == E_UNSPECIFIED) ? Nt + 1 : Nt;

    const dcomplex kx = klong, ky = ktran;

    int ordl = int(SOLVER->getLongSize()), ordt = int(SOLVER->getTranSize());

    double bl = 2 * PI / (front - back) * (symmetric_long() ? 0.5 : 1.0),
           bt = 2 * PI / (right - left) * (symmetric_tran() ? 0.5 : 1.0);

    assert(dynamic_pointer_cast<const MeshD<3>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<3>>(level->mesh());
    double vpos = level->vpos();

    int dxl = 0, dyl = 0, dxt = 0, dyt = 0;
    if (field_interpolation != INTERPOLATION_FOURIER) {
        if (symmetric_long()) {
            if (syml == E_TRAN)
                dxl = 1;
            else
                dyl = 1;
            for (size_t t = 0, end = nl * nt; t != end; t += nl) field[nl - 1 + t] = Vec<3, dcomplex>(0., 0., 0.);
        }
        if (symmetric_tran()) {
            if (symt == E_TRAN)
                dxt = 1;
            else
                dyt = 1;
            for (size_t l = 0, off = nl * (nt - 1); l != Nl; ++l) field[off + l] = Vec<3, dcomplex>(0., 0., 0.);
        }
    }

    if (which_field == FIELD_E) {
        for (int it = symmetric_tran() ? 0 : -ordt; it <= ordt; ++it) {
            for (int il = symmetric_long() ? 0 : -ordl; il <= ordl; ++il) {
                // How expensive is checking conditions in each loop?
                // Fuck it! The code is much more clear this way.
                size_t iex = nl * (((it < 0) ? Nt + it : it) - dxt) + ((il < 0) ? Nl + il : il) - dxl;
                size_t iey = nl * (((it < 0) ? Nt + it : it) - dyt) + ((il < 0) ? Nl + il : il) - dyl;
                size_t iez = nl * (((it < 0) ? Nt + it : it) - dxt) + ((il < 0) ? Nl + il : il) - dyl;
                if (!(it == 0 && dxt) && !(il == 0 && dxl)) field[iex].lon() = E[iEx(il, it)];
                if (!(it == 0 && dyt) && !(il == 0 && dyl)) field[iey].tran() = E[iEy(il, it)];
                if (!(it == 0 && dxt) && !(il == 0 && dyl)) {
                    field[iez].vert() = 0.;
                    for (int jt = -ordt; jt <= ordt; ++jt)
                        for (int jl = -ordl; jl <= ordl; ++jl) {
                            double fhx =
                                ((jl < 0 && symmetry_long == E_LONG) ? -1. : 1) * ((jt < 0 && symmetry_tran == E_LONG) ? -1. : 1);
                            double fhy =
                                ((jl < 0 && symmetry_long == E_TRAN) ? -1. : 1) * ((jt < 0 && symmetry_tran == E_TRAN) ? -1. : 1);
                            field[iez].vert() +=
                                ((SOLVER->expansion_rule == FourierSolver3D::RULE_OLD) ? iepszz(l, il - jl, it - jt)
                                 : diagonal ? ((il == jl && it == jt) ? 1. / coeffs[l][0].c22 : 0.)
                                            : iepszz(l, il, jl, it, jt)) *
                                ((bl * double(jl) - kx) * fhy * H[iHy(jl, jt)] + (bt * double(jt) - ky) * fhx * H[iHx(jl, jt)]);
                        }
                    field[iez].vert() /= k0;
                }
            }
        }
    } else {  // field == FIELD_H
        for (int it = symmetric_tran() ? 0 : -ordt; it <= ordt; ++it) {
            for (int il = symmetric_long() ? 0 : -ordl; il <= ordl; ++il) {
                size_t ihx = nl * (((it < 0) ? Nt + it : it) - dxt) + ((il < 0) ? Nl + il : il) - dxl;
                size_t ihy = nl * (((it < 0) ? Nt + it : it) - dyt) + ((il < 0) ? Nl + il : il) - dyl;
                size_t ihz = nl * (((it < 0) ? Nt + it : it) - dxt) + ((il < 0) ? Nl + il : il) - dyl;
                if (!(it == 0 && dxt) && !(il == 0 && dxl)) field[ihx].lon() = -H[iHx(il, it)];
                if (!(it == 0 && dyt) && !(il == 0 && dyl)) field[ihy].tran() = H[iHy(il, it)];
                if (!(it == 0 && dxt) && !(il == 0 && dyl)) {
                    field[ihz].vert() = 0.;
                    for (int jt = -ordt; jt <= ordt; ++jt)
                        for (int jl = -ordl; jl <= ordl; ++jl) {
                            double fex =
                                ((jl < 0 && symmetry_long == E_TRAN) ? -1. : 1) * ((jt < 0 && symmetry_tran == E_TRAN) ? -1. : 1);
                            double fey =
                                ((jl < 0 && symmetry_long == E_LONG) ? -1. : 1) * ((jt < 0 && symmetry_tran == E_LONG) ? -1. : 1);
                            field[ihz].vert() += imuzz(l, il - jl, it - jt) * (-(bl * double(jl) - kx) * fey * E[iEy(jl, jt)] +
                                                                               (bt * double(jt) - ky) * fex * E[iEx(jl, jt)]);
                        }
                    field[ihz].vert() /= k0;
                }
            }
        }
    }

    if (field_interpolation == INTERPOLATION_FOURIER) {
        const double lo0 = symmetric_long() ? -front : back, hi0 = front, lo1 = symmetric_tran() ? -right : left, hi1 = right;
        DataVector<Vec<3, dcomplex>> result(dest_mesh->size());
        double Ll = (symmetric_long() ? 2. : 1.) * (front - back), Lt = (symmetric_tran() ? 2. : 1.) * (right - left);
        dcomplex bl = 2. * PI * I / Ll, bt = 2. * PI * I / Lt;
        dcomplex ikx = I * kx, iky = I * ky;
        result.reset(dest_mesh->size(), Vec<3, dcomplex>(0., 0., 0.));
        for (int it = -ordt; it <= ordt; ++it) {
            double ftx = 1., fty = 1.;
            size_t iit;
            if (it < 0) {
                if (symmetric_tran()) {
                    if (symt == E_LONG)
                        fty = -1.;
                    else
                        ftx = -1.;
                    iit = nl * (-it);
                } else {
                    iit = nl * (Nt + it);
                }
            } else {
                iit = nl * it;
            }
            dcomplex gt = bt * double(it) - iky;
            for (int il = -ordl; il <= ordl; ++il) {
                double flx = 1., fly = 1.;
                size_t iil;
                if (il < 0) {
                    if (symmetric_long()) {
                        if (syml == E_LONG)
                            fly = -1.;
                        else
                            flx = -1.;
                        iil = -il;
                    } else {
                        iil = Nl + il;
                    }
                } else {
                    iil = il;
                }
                Vec<3, dcomplex> coeff = field[iit + iil];
                coeff.c0 *= ftx * flx;
                coeff.c1 *= fty * fly;
                coeff.c2 *= ftx * fly;
                dcomplex gl = bl * double(il) - ikx;
                for (size_t ip = 0; ip != dest_mesh->size(); ++ip) {
                    auto p = dest_mesh->at(ip);
                    if (!periodic_long) p.c0 = clamp(p.c0, lo0, hi0);
                    if (!periodic_tran) p.c1 = clamp(p.c1, lo1, hi1);
                    result[ip] += coeff * exp(gl * (p.c0 - back) + gt * (p.c1 - left));
                }
            }
        }
        return result;
    } else {
        if (symmetric_long() || symmetric_tran()) {
            fft_x.execute(&(field.data()->lon()), 1);
            fft_y.execute(&(field.data()->tran()), 1);
            fft_z.execute(&(field.data()->vert()), 1);
            double dx, dy;
            if (symmetric_tran()) {
                dy = 0.5 * (right - left) / double(nt);
            } else {
                for (size_t l = 0, off = nl * Nt; l != Nl; ++l) field[off + l] = field[l];
                dy = 0.;
            }
            if (symmetric_long()) {
                dx = 0.5 * (front - back) / double(nl);
            } else {
                for (size_t t = 0, end = nl * nt; t != end; t += nl) field[Nl + t] = field[t];
                dx = 0.;
            }
            auto src_mesh = plask::make_shared<RectangularMesh<3>>(plask::make_shared<RegularAxis>(back + dx, front - dx, nl),
                                                                   plask::make_shared<RegularAxis>(left + dy, right - dy, nt),
                                                                   plask::make_shared<RegularAxis>(vpos, vpos, 1),
                                                                   RectangularMesh<3>::ORDER_210);
            LazyData<Vec<3, dcomplex>> interpolated = interpolate(
                src_mesh, field, dest_mesh, field_interpolation,
                InterpolationFlags(SOLVER->getGeometry(),
                                   symmetric_long() ? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                   symmetric_tran() ? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                   InterpolationFlags::Symmetry::NO),
                false);

            return LazyData<Vec<3, dcomplex>>(interpolated.size(),
                                              [interpolated, dest_mesh, syml, symt, kx, ky, this](size_t i) -> Vec<3, dcomplex> {
                                                  Vec<3, dcomplex> result = interpolated[i];
                                                  if (symmetric_long()) {
                                                      double Ll = 2. * front;
                                                      if (syml == E_TRAN) {
                                                          double x = std::fmod(dest_mesh->at(i)[0], Ll);
                                                          if ((-front <= x && x < 0) || x > front) {
                                                              result.lon() = -result.lon();
                                                              result.vert() = -result.vert();
                                                          }
                                                      } else {
                                                          double x = std::fmod(dest_mesh->at(i)[0], Ll);
                                                          if ((-front <= x && x < 0) || x > front) {
                                                              result.tran() = -result.tran();
                                                          }
                                                      }
                                                  } else {
                                                      dcomplex ikx = I * kx;
                                                      result[i] *= exp(-ikx * dest_mesh->at(i).c0);
                                                  }
                                                  if (symmetric_tran()) {
                                                      double Lt = 2. * right;
                                                      if (symt == E_TRAN) {
                                                          double y = std::fmod(dest_mesh->at(i)[1], Lt);
                                                          if ((-right <= y && y < 0) || y > right) {
                                                              result.lon() = -result.lon();
                                                              result.vert() = -result.vert();
                                                          }
                                                      } else {
                                                          double y = std::fmod(dest_mesh->at(i)[1], Lt);
                                                          if ((-right <= y && y < 0) || y > right) {
                                                              result.tran() = -result.tran();
                                                          }
                                                      }
                                                  } else {
                                                      dcomplex iky = I * ky;
                                                      result *= exp(-iky * dest_mesh->at(i).c1);
                                                  }
                                                  return result;
                                              });
        } else {
            fft_z.execute(reinterpret_cast<dcomplex*>(field.data()));
            for (size_t l = 0, off = nl * Nt; l != Nl; ++l) field[off + l] = field[l];
            for (size_t t = 0, end = nl * nt; t != end; t += nl) field[Nl + t] = field[t];
            auto src_mesh = plask::make_shared<RectangularMesh<3>>(
                plask::make_shared<RegularAxis>(back, front, nl), plask::make_shared<RegularAxis>(left, right, nt),
                plask::make_shared<RegularAxis>(vpos, vpos, 1), RectangularMesh<3>::ORDER_210);
            LazyData<Vec<3, dcomplex>> interpolated =
                interpolate(src_mesh, field, dest_mesh, field_interpolation,
                            InterpolationFlags(SOLVER->getGeometry(), InterpolationFlags::Symmetry::NO,
                                               InterpolationFlags::Symmetry::NO, InterpolationFlags::Symmetry::NO),
                            false);
            dcomplex ikx = I * kx, iky = I * ky;
            return LazyData<Vec<3, dcomplex>>(interpolated.size(), [interpolated, dest_mesh, ikx, iky](size_t i) {
                return interpolated[i] * exp(-ikx * dest_mesh->at(i).c0 - iky * dest_mesh->at(i).c1);
            });
        }
    }
}

double ExpansionPW3D::integratePoyntingVert(const cvector& E, const cvector& H) {
    double P = 0.;

    int ordl = int(SOLVER->getLongSize()), ordt = int(SOLVER->getTranSize());

    for (int iy = -ordt; iy <= ordt; ++iy) {
        for (int ix = -ordl; ix <= ordl; ++ix) {
            P += real(E[iEx(ix, iy)] * conj(H[iHy(ix, iy)]) + E[iEy(ix, iy)] * conj(H[iHx(ix, iy)]));
        }
    }

    double dlong = symmetric_long() ? 2 * front : front - back, dtran = symmetric_tran() ? 2 * right : right - left;
    return P * dlong * dtran * 1e-12;  // µm² -> m²
}

void ExpansionPW3D::getDiagonalEigenvectors(cmatrix& Te, cmatrix& Te1, const cmatrix& RE, const cdiagonal&) {
    size_t nr = Te.rows(), nc = Te.cols();
    std::fill_n(Te.data(), nr * nc, 0.);
    std::fill_n(Te1.data(), nr * nc, 0.);

    // Ensure that for the same gamma E*H [2x2] is diagonal
    assert(nc % 2 == 0);
    size_t n = nc / 2;
    for (std::size_t i = 0; i < n; i++) {
        // Compute Te1 = sqrt(RE)
        // https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        // but after this normalize columns to 1
        dcomplex a = RE(2 * i, 2 * i), b = RE(2 * i, 2 * i + 1), c = RE(2 * i + 1, 2 * i), d = RE(2 * i + 1, 2 * i + 1);
        if (is_zero(b) && is_zero(c)) {
            Te1(2 * i, 2 * i) = Te1(2 * i + 1, 2 * i + 1) = Te(2 * i, 2 * i) = Te(2 * i + 1, 2 * i + 1) = 1.;
            Te1(2 * i, 2 * i + 1) = Te1(2 * i + 1, 2 * i) = Te(2 * i, 2 * i + 1) = Te(2 * i + 1, 2 * i) = 0.;
        } else {
            dcomplex s = sqrt(a * d - b * c);
            a += s;
            d += s;
            // Normalize
            s = 1. / sqrt(a * a + b * b);
            a *= s;
            b *= s;
            s = 1. / sqrt(c * c + d * d);
            c *= s;
            d *= s;
            Te1(2 * i, 2 * i) = a;
            Te1(2 * i, 2 * i + 1) = b;
            Te1(2 * i + 1, 2 * i) = c;
            Te1(2 * i + 1, 2 * i + 1) = d;
            // Invert Te1
            s = 1. / (a * d - b * c);
            Te(2 * i, 2 * i) = s * d;
            Te(2 * i, 2 * i + 1) = -s * b;
            Te(2 * i + 1, 2 * i) = -s * c;
            Te(2 * i + 1, 2 * i + 1) = s * a;
        }
    }
}

double ExpansionPW3D::integrateField(WhichField field,
                                     size_t layer,
                                     const cmatrix& TE,
                                     const cmatrix& TH,
                                     const std::function<std::pair<dcomplex, dcomplex>(size_t, size_t)>& vertical) {
    Component syml = (field == FIELD_E) ? symmetry_long : Component((3 - symmetry_long) % 3);
    Component symt = (field == FIELD_E) ? symmetry_tran : Component((3 - symmetry_tran) % 3);

    bool diagonal = diagonals[layer];

    size_t nl = (syml == E_UNSPECIFIED) ? Nl + 1 : Nl, nt = (symt == E_UNSPECIFIED) ? Nt + 1 : Nt;
    const dcomplex kx = klong, ky = ktran;

    int ordl = int(SOLVER->getLongSize()), ordt = int(SOLVER->getTranSize());

    double bl = 2 * PI / (front - back) * (symmetric_long() ? 0.5 : 1.0),
           bt = 2 * PI / (right - left) * (symmetric_tran() ? 0.5 : 1.0);

    assert(TE.rows() == matrixSize());
    assert(TH.rows() == matrixSize());

    size_t M = TE.cols();
    assert(TH.cols() == M);

    size_t NN = Nl * Nt;
    TempMatrix temp = getTempMatrix();
    cmatrix Fz(NN, M, temp.data());

    if (field == FIELD_E) {
        PLASK_OMP_PARALLEL_FOR
        for (openmp_size_t m = 0; m < M; m++) {
            for (int it = symmetric_tran() ? 0 : -ordt; it <= ordt; ++it) {
                for (int il = symmetric_long() ? 0 : -ordl; il <= ordl; ++il) {
                    dcomplex vert = 0.;
                    for (int jt = -ordt; jt <= ordt; ++jt) {
                        for (int jl = -ordl; jl <= ordl; ++jl) {
                            double fhx =
                                ((jl < 0 && symmetry_long == E_LONG) ? -1. : 1) * ((jt < 0 && symmetry_tran == E_LONG) ? -1. : 1);
                            double fhy =
                                ((jl < 0 && symmetry_long == E_TRAN) ? -1. : 1) * ((jt < 0 && symmetry_tran == E_TRAN) ? -1. : 1);
                            vert += ((SOLVER->expansion_rule == FourierSolver3D::RULE_OLD) ? iepszz(layer, il - jl, it - jt)
                                     : diagonal ? ((il == jl && it == jt) ? 1. / coeffs[layer][0].c22 : 0.)
                                                : iepszz(layer, il, jl, it, jt)) *
                                    ((bl * double(jl) - kx) * fhy * TH(iHy(jl, jt), m) +
                                     (bt * double(jt) - ky) * fhx * TH(iHx(jl, jt), m));
                        }
                    }
                    Fz(idx(il, it), m) = vert / k0;
                }
            }
        }
    } else {  // field == FIELD_H
        PLASK_OMP_PARALLEL_FOR
        for (openmp_size_t m = 0; m < M; m++) {
            for (int it = symmetric_tran() ? 0 : -ordt; it <= ordt; ++it) {
                for (int il = symmetric_long() ? 0 : -ordl; il <= ordl; ++il) {
                    dcomplex vert = 0.;
                    for (int jt = -ordt; jt <= ordt; ++jt) {
                        for (int jl = -ordl; jl <= ordl; ++jl) {
                            double fex =
                                ((jl < 0 && symmetry_long == E_TRAN) ? -1. : 1) * ((jt < 0 && symmetry_tran == E_TRAN) ? -1. : 1);
                            double fey =
                                ((jl < 0 && symmetry_long == E_LONG) ? -1. : 1) * ((jt < 0 && symmetry_tran == E_LONG) ? -1. : 1);
                            vert += imuzz(layer, il - jl, it - jt) * (-(bl * double(jl) - kx) * fey * TE(iEy(jl, jt), m) +
                                                                      (bt * double(jt) - ky) * fex * TE(iEx(jl, jt), m));
                        }
                    }
                    Fz(idx(il, it), m) = vert / k0;
                }
            }
        }
    }

    double result = 0.;

    if (field == FIELD_E) {
        PLASK_OMP_PARALLEL_FOR
        for (openmp_size_t m1 = 0; m1 < M; ++m1) {
            for (openmp_size_t m2 = m1; m2 < M; ++m2) {
                dcomplex resxy = 0., resz = 0.;
                for (int t = -ordt; t <= ordt; ++t) {
                    for (int l = -ordl; l <= ordl; ++l) {
                        size_t ix = iEx(l, t), iy = iEy(l, t), i = idx(l, t);
                        resxy += TE(ix, m1) * conj(TE(ix, m2)) + TE(iy, m1) * conj(TE(iy, m2));
                        resz += Fz(i, m1) * conj(Fz(i, m2));
                    }
                }
                if (!(is_zero(resxy) && is_zero(resz))) {
                    auto vert = vertical(m1, m2);
                    double res = real(resxy * vert.first + resz * vert.second);
                    if (m2 != m1) res *= 2;
#pragma omp atomic
                    result += res;
                }
            }
        }

    } else {  // field == FIELD_H
        PLASK_OMP_PARALLEL_FOR
        for (openmp_size_t m1 = 0; m1 < M; ++m1) {
            for (openmp_size_t m2 = m1; m2 < M; ++m2) {
                dcomplex resxy = 0., resz = 0.;
                for (int t = -ordt; t <= ordt; ++t) {
                    for (int l = -ordl; l <= ordl; ++l) {
                        size_t ix = iHx(l, t), iy = iHy(l, t), i = idx(l, t);
                        resxy += TH(ix, m1) * conj(TH(ix, m2)) + TH(iy, m1) * conj(TH(iy, m2));
                        resz += Fz(i, m1) * conj(Fz(i, m2));
                    }
                }
                if (!(is_zero(resxy) && is_zero(resz))) {
                    auto vert = vertical(m1, m2);
                    double res = real(resxy * vert.second + resz * vert.first);
                    if (m2 != m1) res *= 2;
#pragma omp atomic
                    result += res;
                }
            }
        }
    }

    const double area = (front - back) * (symmetric_long() ? 2. : 1.) * (right - left) * (symmetric_tran() ? 2. : 1.);
    return 0.5 * result * area;
}

LazyData<double> ExpansionPW3D::getGradients(GradientFunctions::EnumType what,
                                             const shared_ptr<const typename LevelsAdapter::Level>& level,
                                             InterpolationMethod interp) {
    double z = level->vpos();
    const size_t lay = SOLVER->stack[solver->getLayerFor(z)];
    const int which = int(what);

    assert(dynamic_pointer_cast<const MeshD<3>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<3>>(level->mesh());

    if (diagonals[lay] || SOLVER->expansion_rule != FourierSolver3D::RULE_COMBINED)
        return LazyData<double>(dest_mesh->size(), [](size_t i) { return 0.0; });

    if (interp == INTERPOLATION_DEFAULT || interp == INTERPOLATION_FOURIER) {
        return LazyData<double>(dest_mesh->size(), [this, lay, which, dest_mesh](size_t i) -> double {
            double res = 0.;
            const int nt = symmetric_tran() ? int(nNt) - 1 : int(nNt / 2), nl = symmetric_long() ? int(nNl) - 1 : int(nNl / 2);
            double Lt = right - left;
            if (symmetric_tran()) Lt *= 2;
            double Ll = front - back;
            if (symmetric_long()) Ll *= 2;
            // double f = 1.;
            for (int kt = -nt; kt <= nt; ++kt) {
                size_t t = (kt >= 0) ? kt : (symmetric_tran()) ? -kt : kt + nNt;
                const double phast = kt * (dest_mesh->at(i).c1 - left) / Lt;
                for (int kl = -nl; kl <= nl; ++kl) {
                    size_t l = (kl >= 0) ? kl : (symmetric_long()) ? -kl : kl + nNl;
                    res += (*(reinterpret_cast<dcomplex*>(gradients[lay].data() + nNl * t + l) + which) *
                            exp(2 * PI * I * (kl * (dest_mesh->at(i).c0 - back) / Ll + phast)))
                               .real();
                }
            }
            if (which && (dest_mesh->at(i).c0 < 0) != (dest_mesh->at(i).c1 < 0)) res = -res;
            return res;
        });
    } else {
        size_t nl = symmetric_long() ? nNl : nNl + 1, nt = symmetric_tran() ? nNt : nNt + 1;
        DataVector<double> grads;
        {
            DataVector<dcomplex> work(nl * nt);
            for (size_t t = 0; t != nNt; ++t) {
                size_t op = nl * t, oc = nNl * t;
                for (size_t l = 0; l != nNl; ++l) {
                    work[op + l] = *(reinterpret_cast<dcomplex*>((gradients[lay].data() + oc + l)) + which);
                }
            }
            auto dct_symmetry = SOLVER->dct2() ? (which ? FFT::SYMMETRY_ODD_2 : FFT::SYMMETRY_EVEN_2)
                                               : (which ? FFT::SYMMETRY_ODD_1 : FFT::SYMMETRY_EVEN_1);
            FFT::Backward2D(1, nNl, nNt, symmetric_long() ? dct_symmetry : FFT::SYMMETRY_NONE,
                            symmetric_tran() ? dct_symmetry : FFT::SYMMETRY_NONE, nl)
                .execute(reinterpret_cast<dcomplex*>(work.data()));
            grads.reset(nl * nt);
            auto dst = grads.begin();
            for (const dcomplex& val : work) *(dst++) = val.real();
        }
        shared_ptr<RegularAxis> lcmesh = plask::make_shared<RegularAxis>(), tcmesh = plask::make_shared<RegularAxis>();
        if (symmetric_long()) {
            if (SOLVER->dct2()) {
                double dx = 0.5 * (front - back) / double(nl);
                lcmesh->reset(back + dx, front - dx, nl);
            } else {
                lcmesh->reset(0., front, nl);
            }
        } else {
            lcmesh->reset(back, front, nl);
            for (size_t t = 0, end = nl * nt, dist = nl - 1; t != end; t += nl) grads[dist + t] = grads[t];
        }
        if (symmetric_tran()) {
            if (SOLVER->dct2()) {
                double dy = 0.5 * right / double(nt);
                tcmesh->reset(dy, right - dy, nt);
            } else {
                tcmesh->reset(0., right, nt);
            }
        } else {
            tcmesh->reset(left, right, nt);
            for (size_t l = 0, last = nl * (nt - 1); l != nl; ++l) grads[last + l] = grads[l];
        }
        auto src_mesh = plask::make_shared<RectangularMesh<3>>(
            lcmesh, tcmesh, plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1), RectangularMesh<3>::ORDER_210);

        auto sym =
            (what == GradientFunctions::COS2) ? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NEGATIVE;

        return interpolate(
            src_mesh, grads, dest_mesh, interp,
            InterpolationFlags(SOLVER->getGeometry(), symmetric_long() ? sym : InterpolationFlags::Symmetry::NO,
                               symmetric_tran() ? sym : InterpolationFlags::Symmetry::NO, InterpolationFlags::Symmetry::POSITIVE));
    }
}

}}}  // namespace plask::optical::modal
