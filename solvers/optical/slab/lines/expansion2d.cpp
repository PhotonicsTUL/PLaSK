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
#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/adaptor/transformed.hpp>
using boost::algorithm::clamp;

#include "../meshadapter.hpp"
#include "expansion2d.hpp"
#include "solver2d.hpp"

#define SOLVER static_cast<LinesSolver2D*>(solver)

namespace plask { namespace optical { namespace slab {

ExpansionFD2D::ExpansionFD2D(LinesSolver2D* solver)
    : Expansion(solver), initialized(false), symmetry(E_UNSPECIFIED), polarization(E_UNSPECIFIED) {}

void ExpansionFD2D::setPolarization(Component pol) {
    if (pol != polarization) {
        solver->clearFields();
        if (!periodic && polarization == E_TRAN) {
            polarization = pol;
            if (initialized) {
                reset();
                init();
            }
            solver->recompute_integrals = true;
        } else if (polarization != E_UNSPECIFIED) {
            polarization = pol;
            solver->recompute_integrals = true;
        } else {
            polarization = pol;
        }
    }
}

void ExpansionFD2D::init() {
    auto geometry = SOLVER->getGeometry();

    periodic = geometry->isPeriodic(Geometry2DCartesian::DIRECTION_TRAN);

    auto bbox = geometry->getChild()->getBoundingBox();
    left = bbox.lower[0];
    right = bbox.upper[0];

    size_t refine = SOLVER->refine;
    if (refine == 0) refine = 1;

    if (symmetry != E_UNSPECIFIED && !geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN))
        throw BadInput(solver->getId(), "Symmetry not allowed for asymmetric structure");

    if (geometry->isSymmetric(Geometry2DCartesian::DIRECTION_TRAN)) {
        if (right <= 0.) {
            left = -left;
            right = -right;
            std::swap(left, right);
        }
        if (left != 0.) throw BadMesh(SOLVER->getId(), "Symmetric geometry must have one of its sides at symmetry axis");
        if (!symmetric()) left = -right;
    }

    double dx;
    size_t N;

    if (SOLVER->mesh) {
        if (SOLVER->mesh->size() < 2) throw BadInput(SOLVER->getId(), "Mesh needs at least two points");
        if (!is_zero(SOLVER->mesh->at(0) - (symmetric() ? -right : left)))
            throw BadInput(SOLVER->getId(), "First mesh point ({}) must match left geometry boundary ({})", SOLVER->mesh->at(0),
                           symmetric() ? -right : left);
        if (!is_zero(SOLVER->mesh->at(SOLVER->mesh->size() - 1) - right))
            throw BadInput(SOLVER->getId(), "Last mesh point ({}) must match right geometry boundary ({})",
                           SOLVER->mesh->at(SOLVER->mesh->size() - 1), right);
        dx = SOLVER->mesh->step();
        N = SOLVER->mesh->size();
    } else {
        if (isnan(SOLVER->density)) throw Exception(format("{0}: No mesh nor density specified", SOLVER->getId()));
        double L = right - left;
        N = size_t(std::ceil(L / SOLVER->density)) + 1;
        dx = L / (N - 1);
    }

    if (periodic && !symmetric()) {
        right -= dx;
        N -= 1;
    }

    double xs = dx * (0.5 - 0.5 / refine);
    auto xmesh = plask::make_shared<RegularAxis>(left - xs, right + xs, N * refine);

    // Add PMLs
    if (!periodic) {
        pml = SOLVER->pml;
        double nd = std::ceil(pml.dist / dx), np = std::ceil(pml.size / dx);
        pml.dist = nd * dx;
        pml.size = np * dx;
        right += pml.size + pml.dist;
        ndr = size_t(nd);
        npr = size_t(np);
        if (!symmetric()) {
            left -= pml.size + pml.dist;
            ndl = ndr;
            npl = npr;
        } else {
            ndl = 0;
            npl = 0;
        }
        N += ndl + npl + ndr + npr;
    } else {
        npl = ndl = ndr = npr = 0;
    }

    mesh = plask::make_shared<RegularAxis>(left, right, N);

    size_t M = matrixSize();
    SOLVER->writelog(LOG_DETAIL, "Creating{2}{3} discretization with {0} points (matrix size: {1})", N, M,
                     symmetric() ? " symmetric" : "", separated() ? " separated" : "");

    if (symmetric()) SOLVER->writelog(LOG_DETAIL, "Symmetry is {0}", (symmetry == E_TRAN) ? "Etran" : "Elong");

    // Compute permeability coefficients
    mag.reset(mesh->size(), 1.);
    if (!periodic) {
        for (size_t i = 0; i < npl; ++i) {
            double h = (npl - i) * dx;
            mag[i] = 1. + (SOLVER->pml.factor - 1.) * pow(h, SOLVER->pml.order);
        }
        for (size_t i = 0, n = N - npr; i < npr; ++i) {
            double h = i * dx;
            mag[n + i] = 1. + (SOLVER->pml.factor - 1.) * pow(h, SOLVER->pml.order);
        }
    }

    material_mesh = plask::make_shared<RectangularMesh<2>>(xmesh, solver->verts, RectangularMesh<2>::ORDER_01);

    initialized = true;

    // Allocate memory for expansion coefficients
    size_t nlayers = solver->lcount;
    epsilon.resize(solver->lcount);
}

void ExpansionFD2D::reset() {
    epsilon.clear();
    initialized = false;
    mesh.reset();
    mag.reset();
    temporary.reset();
}

void ExpansionFD2D::beforeLayersIntegrals(double lam, double glam) {
    SOLVER->prepareExpansionIntegrals(this, material_mesh, lam, glam);
}

void ExpansionFD2D::layerIntegrals(size_t layer, double lam, double glam) {
    auto geometry = SOLVER->getGeometry();

    size_t refine = SOLVER->refine;
    if (refine == 0) refine = 1;

    double maty;
    for (size_t i = 0; i != solver->stack.size(); ++i) {
        if (solver->stack[i] == layer) {
            maty = solver->verts->at(i);
            break;
        }
    }

    size_t shift = npl + ndl;
    size_t N = mesh->size() - shift - npr - ndr;

#if defined(OPENMP_FOUND)  // && !defined(NDEBUG)
    SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {}/{} (sampled at {} points) in thread {}", layer,
                     solver->lcount, refine * N, omp_get_thread_num());
#else
    SOLVER->writelog(LOG_DETAIL, "Getting refractive indices for layer {}/{} (sampled at {} points)", layer, solver->lcount,
                     refine * N);
#endif

    if (isnan(lam)) throw BadInput(SOLVER->getId(), "No wavelength given: specify 'lam' or 'lam0'");

    if (gain_connected && solver->lgained[layer]) {
        SOLVER->writelog(LOG_DEBUG, "Layer {:d} has gain", layer);
        if (isnan(glam)) glam = lam;
    }

    epsilon[layer].reset(mesh->size(), Tensor3<dcomplex>(0.));

    for (size_t i = 0; i < N; i++) {
        size_t i1 = i + shift;
        size_t j0 = i * refine;
        for (size_t j = 0; j < refine; ++j) {
            Tensor3<dcomplex> eps = getEpsilon(geometry, layer, maty, lam, glam, j0 + j);
            if (eps.c01 != 0. && separated())
                throw BadInput(solver->getId(), "Polarization can be specified only for diagonal refractive index tensor (NR)");
            if (eps.c01 != 0. && symmetric())
                throw BadInput(solver->getId(), "Symmetry can be specified only for diagonal refractive index tensor (NR)");
            epsilon[layer][i1] += eps;
        }
        epsilon[layer][i1] /= refine;
    }

    // Add PMLs
    if (!periodic) {
        double dx = mesh->step();
        auto epsl = epsilon[layer][shift];
        std::fill(epsilon[layer].begin() + npl, epsilon[layer].begin() + shift, epsl);
        for (size_t i = 0; i < npl; ++i) {
            double h = (npl - i) * dx;
            dcomplex s = 1. + (SOLVER->pml.factor - 1.) * pow(h, SOLVER->pml.order);
            epsilon[layer][i] = Tensor3<dcomplex>(epsl.c00 * s, epsl.c11 / s, epsl.c22 * s);
        }
        size_t idr = mesh->size() - npr - ndr - 1;
        auto epsr = epsilon[layer][idr];
        std::fill(epsilon[layer].begin() + idr, epsilon[layer].begin() + idr + ndr, epsr);
        for (size_t i = 0, ipr = idr + ndr; i <= npr; ++i) {
            double h = i * dx;
            dcomplex s = 1. + (SOLVER->pml.factor - 1.) * pow(h, SOLVER->pml.order);
            epsilon[layer][ipr + i] = Tensor3<dcomplex>(epsr.c00 * s, epsr.c11 / s, epsr.c22 * s);
        }
    }

    for (Tensor3<dcomplex>& eps : epsilon[layer]) {
        eps.c22 = 1. / eps.c22;
    }
}

LazyData<Tensor3<dcomplex>> ExpansionFD2D::getMaterialNR(size_t l,
                                                         const shared_ptr<const LevelsAdapter::Level>& level,
                                                         InterpolationMethod interp) {
    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());

    auto src_mesh = plask::make_shared<RectangularMesh<2>>(mesh, plask::make_shared<RegularAxis>(level->vpos(), level->vpos(), 1));

    DataVector<Tensor3<dcomplex>> nr(this->epsilon[l].size());
    Tensor3<dcomplex>* to = nr.begin();
    for (const Tensor3<dcomplex>& val : this->epsilon[l]) {
        *(to++) = Tensor3<dcomplex>(sqrt(val.c00), sqrt(val.c11), sqrt(1. / val.c22), sqrt(val.c01));
    }

    return interpolate(src_mesh, nr, dest_mesh, getInterpolationMethod<INTERPOLATION_LINEAR>(interp),
                       InterpolationFlags(SOLVER->getGeometry(),
                                          symmetric() ? InterpolationFlags::Symmetry::POSITIVE : InterpolationFlags::Symmetry::NO,
                                          InterpolationFlags::Symmetry::NO));
}

void ExpansionFD2D::getMatrices(size_t l, cmatrix& RE, cmatrix& RH) {
    assert(initialized);
    if (isnan(k0)) throw BadInput(SOLVER->getId(), "Wavelength or k0 not set");
    if (isinf(k0.real())) throw BadInput(SOLVER->getId(), "Wavelength must not be 0");

    const dcomplex beta{this->beta.real(), this->beta.imag() - SOLVER->getMirrorLosses(this->beta.real() / k0.real())};
    const dcomplex beta2 = beta * beta, ibeta{-beta.imag(), beta.real()};
    const dcomplex rk0 = 1. / k0;
    const dcomplex rk0beta2 = rk0 * beta2;

    const dcomplex ik(-SOLVER->ktran.imag(), SOLVER->ktran.real()), k2 = SOLVER->ktran * SOLVER->ktran;

    const size_t N = mesh->size();

    const double h = 1. / (mesh->step());
    const double h2 = h * h;

    std::fill_n(RE.data(), RE.rows() * RE.cols(), dcomplex(0.));
    std::fill_n(RH.data(), RH.rows() * RH.cols(), dcomplex(0.));

    if (polarization == E_LONG) {
        for (size_t i = 0; i != N; ++i) RH(i, i) = rk0beta2 * repsyy(l, i) - k0 * muxx(i);

        for (size_t i = 0; i != N; ++i) {
            size_t im, ip;
            Component sm, sp;
            checkEdges(i, im, ip, sm, sp);
            const dcomplex rmu = rmuyy(i), drmu = 0.5 * h * (rmuyy(ip) - rmuyy(im));
            dcomplex c1 = (0.5 * drmu + rmu * ik) * h, c2 = rmu * h2;
            if (im != INVALID_INDEX) RE(i, im) -= rk0 * flip<E_LONG>(sm, -c1 + c2);
            RE(i, i) = -rk0 * (drmu * ik - rmu * k2 - 2. * c2) - k0 * epszz(l, i);
            if (ip != INVALID_INDEX) RE(i, ip) -= rk0 * flip<E_LONG>(sp, +c1 + c2);
        }

    } else if (polarization == E_TRAN) {
        for (size_t i = 0; i != N; ++i) RE(i, i) = -rk0beta2 * rmuyy(i) + k0 * epsxx(l, i);

        for (size_t i = 0; i != N; ++i) {
            size_t im, ip;
            Component sm, sp;
            checkEdges(i, im, ip, sm, sp);
            const dcomplex reps = repsyy(l, i), dreps = 0.5 * h * (repsyy(l, ip) - repsyy(l, im));
            dcomplex c1 = (0.5 * dreps + reps * ik) * h, c2 = reps * h2;
            if (im != INVALID_INDEX) RH(i, im) += rk0 * flip<E_TRAN>(sm, -c1 + c2);
            RH(i, i) = rk0 * (dreps * ik - reps * k2 - 2. * c2) + k0 * muzz(i);
            if (ip != INVALID_INDEX) RH(i, ip) += rk0 * flip<E_TRAN>(sp, +c1 + c2);
        }

    } else {
        for (size_t i = 0; i != N; ++i) {
            const size_t iex = iEx(i), iez = iEz(i);
            const size_t ihx = iHx(i), ihz = iHz(i);
            size_t im, ip;
            Component sm, sp;
            checkEdges(i, im, ip, sm, sp);
            const size_t iexm = iEx(im), iezm = iEz(im);
            const size_t iexp = iEx(ip), iezp = iEz(ip);
            const size_t ihxm = iHx(im), ihzm = iHz(im);
            const size_t ihxp = iHx(ip), ihzp = iHz(ip);
            const dcomplex reps = repsyy(l, i), dreps = 0.5 * h * (repsyy(l, ip) - repsyy(l, im));
            const dcomplex rmu = rmuyy(i), drmu = 0.5 * h * (rmuyy(ip) - rmuyy(im));

            dcomplex c1 = (0.5 * dreps + reps * ik) * h, c2 = reps * h2;
            if (im != INVALID_INDEX) RH(iex, ihzm) += rk0 * flip<E_TRAN>(sm, -c1 + c2);
            RH(iex, ihz) = rk0 * (dreps * ik - reps * k2 - 2. * c2) + k0 * muzz(i);
            if (ip != INVALID_INDEX) RH(iex, ihzp) += rk0 * flip<E_TRAN>(sp, +c1 + c2);

            c1 = rk0 * ibeta * reps * 0.5 * h;
            if (im != INVALID_INDEX) RH(iex, ihxm) += flip<E_LONG>(sm, -c1);
            RH(iex, ihx) = rk0 * ibeta * (dreps + reps * ik);
            if (ip != INVALID_INDEX) RH(iex, ihxp) += rk0 * flip<E_LONG>(sp, +c1);

            if (im != INVALID_INDEX) RH(iez, ihzm) -= flip<E_TRAN>(sm, -c1);
            RH(iez, ihz) = -rk0 * ibeta * reps * ik;
            if (ip != INVALID_INDEX) RH(iez, ihzp) -= rk0 * flip<E_TRAN>(sp, +c1);

            RH(iez, ihx) = rk0beta2 * reps - k0 * muxx(i);

            RE(ihz, iex) = -rk0beta2 * rmu + k0 * epsxx(l, i);

            c1 = rk0 * ibeta * rmu * 0.5 * h;
            if (im != INVALID_INDEX) RE(ihz, iezm) += flip<E_LONG>(sm, -c1);
            RE(ihz, iez) = rk0 * ibeta * rmu * ik;
            if (ip != INVALID_INDEX) RE(ihz, iezp) += rk0 * flip<E_LONG>(sp, +c1);

            if (im != INVALID_INDEX) RE(ihx, iexm) -= flip<E_TRAN>(sm, -c1);
            RE(ihx, iex) = -rk0 * ibeta * (drmu + rmu * ik);
            if (ip != INVALID_INDEX) RE(ihx, iexp) -= rk0 * flip<E_TRAN>(sp, +c1);

            c1 = (0.5 * drmu + rmu * ik) * h;
            c2 = rmu * h2;
            if (im != INVALID_INDEX) RE(ihx, iezm) -= rk0 * flip<E_LONG>(sm, -c1 + c2);
            RE(ihx, iez) = -rk0 * (drmu * ik - rmu * k2 - 2. * c2) - k0 * epszz(l, i);
            if (ip != INVALID_INDEX) RE(ihx, iezp) -= rk0 * flip<E_LONG>(sp, +c1 + c2);
        }
    }

    // Ugly hack to avoid singularity
    for (size_t i = 0; i != N; ++i)
        if (RE(i, i) == 0.) RE(i, i) = 1e-32;
    for (size_t i = 0; i != N; ++i)
        if (RH(i, i) == 0.) RH(i, i) = 1e-32;
}

void ExpansionFD2D::prepareField() { field.reset(mesh->size()); }

void ExpansionFD2D::cleanupField() { field.reset(); }

LazyData<Vec<3, dcomplex>> ExpansionFD2D::getField(size_t l,
                                                   const shared_ptr<const typename LevelsAdapter::Level>& level,
                                                   const cvector& E,
                                                   const cvector& H) {
    Component sym = (which_field == FIELD_E) ? symmetry : Component((3 - symmetry) % 3);

    dcomplex ibeta{-this->beta.imag() + SOLVER->getMirrorLosses(this->beta.real() / k0.real()), this->beta.real()};
    dcomplex irk0 = I / k0;

    assert(dynamic_pointer_cast<const MeshD<2>>(level->mesh()));
    auto dest_mesh = static_pointer_cast<const MeshD<2>>(level->mesh());
    double vpos = level->vpos();
    size_t N = mesh->size();
    double h = 0.5 / mesh->step();

    if (which_field == FIELD_E) {
        if (polarization == E_LONG) {
            for (size_t i = 0; i != N; ++i) {
                field[i].tran() = field[i].vert() = 0.;
                field[i].lon() = E[i];
std::cerr << str(E[i]) << "  ";
            }
std::cerr << "\n\n";
        } else if (polarization == E_TRAN) {
            for (size_t i = 0; i != N; ++i) {
                size_t im, ip;
                Component sm, sp;
                checkEdges(i, im, ip, sm, sp);
                field[i].lon() = 0.;
                field[i].tran() = E[i];
                field[i].vert() = 0.;
                dcomplex f = irk0 * repsyy(l, i) * h;
                if (im != INVALID_INDEX) field[i].vert() -= f * flip<E_TRAN>(sm, H[im]);
                if (ip != INVALID_INDEX) field[i].vert() += f * flip<E_TRAN>(sp, H[ip]);
            }
        } else {
            for (size_t i = 0; i != N; ++i) {
                size_t im, ip;
                Component sm, sp;
                checkEdges(i, im, ip, sm, sp);
                field[i].lon() = E[iEz(i)];
                field[i].tran() = E[iEx(i)];
                field[i].vert() = ibeta * H[iHx(i)];
                if (im != INVALID_INDEX) field[i].vert() -= h * flip<E_TRAN>(sm, H[iHz(im)]);
                if (ip != INVALID_INDEX) field[i].vert() += h * flip<E_TRAN>(sp, H[iHz(ip)]);
                field[i].vert() *= irk0 * repsyy(l, i);
            }
        }
    } else {                           // which_field == FIELD_H
        if (polarization == E_TRAN) {  // polarization == H_LONG
            for (size_t i = 0; i != N; ++i) {
                field[i].tran() = field[i].vert() = 0.;
                field[i].lon() = H[i];
            }
        } else if (polarization == E_LONG) {  // polarization == H_TRAN
            for (size_t i = 0; i != N; ++i) {
                size_t im, ip;
                Component sm, sp;
                checkEdges(i, im, ip, sm, sp);
                field[i].lon() = 0.;
                field[i].tran() = H[i];
                field[i].vert() = 0.;
                dcomplex f = -irk0 * rmuyy(i) * h;
                if (im != INVALID_INDEX) field[i].vert() -= f * flip<E_LONG>(sm, E[im]);
                if (ip != INVALID_INDEX) field[i].vert() += f * flip<E_LONG>(sp, E[ip]);
            }
        } else {
            for (size_t i = 0; i != N; ++i) {
                size_t im, ip;
                Component sm, sp;
                checkEdges(i, im, ip, sm, sp);
                field[i].lon() = H[iHz(i)];
                field[i].tran() = H[iHx(i)];
                field[i].vert() = ibeta * H[iHx(i)];
                if (im != INVALID_INDEX) field[i].vert() -= h * flip<E_LONG>(sm, E[iEz(im)]);
                if (ip != INVALID_INDEX) field[i].vert() += h * flip<E_LONG>(sp, E[iEz(ip)]);
                field[i].vert() *= -irk0 * rmuyy(i);
            }
        }
    }

    auto src_mesh = plask::make_shared<RectangularMesh<2>>(mesh, plask::make_shared<RegularAxis>(vpos, vpos, 1));

    LazyData<Vec<3, dcomplex>> interpolated =
        interpolate(src_mesh, field, dest_mesh, getInterpolationMethod<INTERPOLATION_LINEAR>(field_interpolation),
                    InterpolationFlags(SOLVER->getGeometry(),
                                       (sym == E_UNSPECIFIED) ? InterpolationFlags::Symmetry::NO
                                       : (sym == E_TRAN)      ? InterpolationFlags::Symmetry::NPN
                                                              : InterpolationFlags::Symmetry::PNP,
                                       InterpolationFlags::Symmetry::NO),
                    false);

    dcomplex ikx = I * ktran;
    return LazyData<Vec<3,dcomplex>>(interpolated.size(), [interpolated, dest_mesh, ikx] (size_t i) {
        return interpolated[i] * exp(-ikx * dest_mesh->at(i).c0);
    });
}

double ExpansionFD2D::integratePoyntingVert(const cvector& E, const cvector& H) {
    size_t N = mesh->size();
    double P = 0.;
    double dx = mesh->step();

    if (separated()) {
        for (size_t i = 0; i != N; ++i) {
            P += real(E[i] * conj(H[i]));
        }
    } else {
        for (size_t i = 0; i != N; ++i) {
            P += real(E[iEz(i)] * conj(H[iHx(i)])) - real(E[iEx(i)] * conj(H[iHz(i)]));
        }
    }

    if (symmetric()) {
        if (separated()) {
            P -= 0.5 * real(E[0] * conj(H[0]));
        } else {
            P -= 0.5 * (real(E[iEz(0)] * conj(H[iHx(0)])) - real(E[iEx(0)] * conj(H[iHz(0)])));
        }
        if (periodic) {
            size_t N1 = N - 1;
            if (separated()) {
                P -= 0.5 * real(E[N1] * conj(H[N1]));
            } else {
                P -= 0.5 * (real(E[iEz(N1)] * conj(H[iHx(N1)])) - real(E[iEx(N1)] * conj(H[iHz(N1)])));
            }
        }
    }

    double L = SOLVER->geometry->getExtrusion()->getLength();
    if (!isinf(L)) P *= L * 1e-6;

    return P * (symmetric() ? 2. : 1.) * dx * 1e-6;  // µm² -> m²
}

void ExpansionFD2D::getDiagonalEigenvectors(cmatrix& Te, cmatrix Te1, const cmatrix& RE, const cdiagonal& gamma) {
    size_t nr = Te.rows(), nc = Te.cols();
    std::fill_n(Te.data(), nr * nc, 0.);
    std::fill_n(Te1.data(), nr * nc, 0.);

    if (separated()) {
        for (std::size_t i = 0; i < nc; i++) {
            Te(i, i) = Te1(i, i) = 1.;
        }
    } else {
        // Ensure that for the same gamma E*H [2x2] is diagonal
        assert(nc % 2 == 0);
        size_t n = nc / 2;
        for (std::size_t i = 0; i < n; i++) {
            // Compute Te1 = sqrt(RE)
            // https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
            // but after this normalize columns to 1
            dcomplex a = RE(2 * i, 2 * i), b = RE(2 * i, 2 * i + 1), c = RE(2 * i + 1, 2 * i), d = RE(2 * i + 1, 2 * i + 1);
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

double ExpansionFD2D::integrateField(WhichField field, size_t l, const cvector& E, const cvector& H) {
    throw NotImplemented("ExpansionFD2D::integrateField");
    return 1.;
}

}}}  // namespace plask::optical::slab
