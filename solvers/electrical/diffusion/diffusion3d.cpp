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
 * GNU General Public License for more details.tutorial3-20230720-2358.txt
 */
#include "diffusion3d.hpp"

namespace plask { namespace electrical { namespace diffusion {

constexpr double inv_hc = 1.0e-13 / (phys::c * phys::h_J);
using phys::Z0;

Diffusion3DSolver::Diffusion3DSolver(const std::string& name)
    : FemSolverWithMesh<Geometry3D, RectangularMesh<3>>(name),
      loopno(0),
      maxerr(0.05),
      outCarriersConcentration(this, &Diffusion3DSolver::getConcentration) {
    inTemperature = 300.;
}

void Diffusion3DSolver::loadConfiguration(XMLReader& source, Manager& manager) {
    while (source.requireTagOrEnd()) parseConfiguration(source, manager);
}

void Diffusion3DSolver::parseConfiguration(XMLReader& source, Manager& manager) {
    std::string param = source.getNodeName();

    if (param == "loop") {
        maxerr = source.getAttribute<double>("maxerr", maxerr);
        source.requireTagEnd();
    }

    else if (!this->parseFemConfiguration(source, manager)) {
        this->parseStandardConfiguration(source, manager);
    }
}

Diffusion3DSolver::~Diffusion3DSolver() {}

void Diffusion3DSolver::setupActiveRegion3Ds() {
    if (!this->geometry || !this->mesh) return;

    auto points = this->mesh->getElementMesh();

    std::map<size_t, ActiveRegion3D::Region> regions;
    size_t nreg = 0;

    for (size_t lon = 0; lon < points->axis[0]->size(); ++lon) {
        for (size_t tra = 0; tra < points->axis[1]->size(); ++tra) {
            size_t num = 0;
            size_t start = 0;
            for (size_t ver = 0; ver < points->axis[2]->size(); ++ver) {
                auto point = points->at(lon, tra, ver);
                size_t cur = isActive(point);
                if (cur != num) {
                    if (num) {  // summarize current region
                        auto found = regions.find(num);
                        if (found == regions.end()) {  // `num` is a new region
                            regions[num] = ActiveRegion3D::Region(start, ver, lon, tra);
                            if (nreg < num) nreg = num;
                        } else {
                            ActiveRegion3D::Region& region = found->second;
                            if (start != region.bottom || ver != region.top)
                                throw Exception("{0}: Junction {1} does not have top and bottom edges at constant heights",
                                                this->getId(), num - 1);
                            if (tra < region.left) region.left = tra;
                            if (tra >= region.right) region.right = tra + 1;
                            if (lon < region.back) region.back = lon;
                            if (lon >= region.front) region.front = lon + 1;
                        }
                    }
                    num = cur;
                    start = ver;
                }
                if (cur) {
                    auto found = regions.find(cur);
                    if (found != regions.end()) {
                        ActiveRegion3D::Region& region = found->second;
                        if (region.warn && lon != region.lon && tra != region.tra &&
                            *this->geometry->getMaterial(points->at(lon, tra, ver)) !=
                                *this->geometry->getMaterial(points->at(region.lon, region.tra, ver))) {
                            writelog(LOG_WARNING, "Junction {} is laterally non-uniform", num - 1);
                            region.warn = false;
                        }
                    }
                }
            }
            if (num) {  // summarize current region
                auto found = regions.find(num);
                if (found == regions.end()) {  // `current` is a new region
                    regions[num] = ActiveRegion3D::Region(start, points->axis[2]->size(), lon, tra);
                } else {
                    ActiveRegion3D::Region& region = found->second;
                    if (start != region.bottom || points->axis[2]->size() != region.top)
                        throw Exception("{0}: Junction {1} does not have top and bottom edges at constant heights", this->getId(),
                                        num - 1);
                    if (tra < region.left) region.left = tra;
                    if (tra >= region.right) region.right = tra + 1;
                    if (lon < region.back) region.back = lon;
                    if (lon >= region.front) region.front = lon + 1;
                }
            }
        }
    }

    active.clear();
    for (auto& ireg : regions) {
        size_t act = ireg.first - 1;
        ActiveRegion3D::Region& reg = ireg.second;
        // Detect quantum wells in the active region
        std::vector<double> QWz;
        std::vector<std::pair<size_t, size_t>> QWbt;
        double QWheight = 0.;
        std::vector<bool> isQW;
        isQW.reserve(reg.top - reg.bottom);
        for (size_t il = reg.back; il < reg.front; ++il) {
            for (size_t it = reg.left; it < reg.right; ++it) {
                shared_ptr<Material> material;
                for (size_t iv = reg.bottom, j = 0; iv < reg.top; ++iv, ++j) {
                    auto point = points->at(il, it, iv);
                    auto tags = this->geometry->getRolesAt(point);
                    bool QW = tags.find("QW") != tags.end() || tags.find("QD") != tags.end();
                    if (it == reg.left) {
                        isQW.push_back(QW);
                        if (QW) {
                            if (QWbt.empty() || QWbt.back().second != iv)
                                QWbt.emplace_back(iv, iv + 1);
                            else
                                QWbt.back().second = iv + 1;
                            QWz.push_back(point.c1);
                            QWheight += this->mesh->vert()->at(iv + 1) - this->mesh->vert()->at(iv);
                        }
                    } else if (isQW[j] != QW) {
                        throw Exception("{}: Quantum wells in active region {} are not identical", this->getId(), act);
                    }
                    if (QW) {
                        if (!material)
                            material = this->geometry->getMaterial(point);
                        else if (*material != *this->geometry->getMaterial(point)) {
                            throw Exception("{}: Quantum wells in active region {} are not identical", this->getId(), act);
                        }
                    }
                }
            }
        }
        if (QWz.empty()) {
            throw Exception("{}: Active region {} does not contain quantum wells", this->getId(), act);
        }
        active.emplace(std::piecewise_construct, std::forward_as_tuple(act),
                       std::forward_as_tuple(this, reg.back, reg.front, reg.left, reg.right, reg.bottom, reg.top, QWheight,
                                             std::move(QWz), std::move(QWbt)));
        this->writelog(LOG_DETAIL, "Total QWs thickness in active region {}: {}nm", act, 1e3 * QWheight);
        this->writelog(LOG_DEBUG, "Junction {0} span: [{1},{3},{5}]-[{2},{4},{6}]", act, reg.back, reg.front, reg.left, reg.right,
                       reg.bottom, reg.top);
    }
}

void Diffusion3DSolver::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) {
        auto mesh1 = makeGeometryGrid(this->geometry);
        this->mesh = make_shared<RectangularMesh<3>>(refineAxis(mesh1->lon(), DEFAULT_MESH_SPACING),
                                                     refineAxis(mesh1->tran(), DEFAULT_MESH_SPACING), mesh1->vert());
        writelog(LOG_DETAIL, "{}: Setting up default mesh [{}]", this->getId(), this->mesh->tran()->size());
    }
    setupActiveRegion3Ds();
    loopno = 0;
}

void Diffusion3DSolver::onInvalidate() { active.clear(); }

// clang-format off
inline void Diffusion3DSolver::setLocalMatrix(FemMatrix& K, DataVector<double>& F, const ElementParams3D e,
    const double A, const double B, const double C, const double D, const double* U, const double* J) {
#   include "diffusion3d-eval.ipp"
}

inline void Diffusion3DSolver::addLocalBurningMatrix(FemMatrix& K, DataVector<double>& F, const ElementParams3D e,
    const Tensor2<double> G, const Tensor2<double> dG, const double Ug, const Tensor2<double>* P) {
#   include "diffusion3d-eval-shb.ipp"
}
// clang-format on

double Diffusion3DSolver::compute(unsigned loops, bool shb, size_t act) {
    this->initCalculation();

    auto found = this->active.find(act);
    if (found == this->active.end()) throw Exception("{}: Active region {} does not exist", this->getId(), act);
    auto& active = found->second;

    double z = active.vert();

    size_t nn = active.mesh2->size(), ne = active.emesh2->size();
    size_t N = 3 * nn;

    size_t nmodes = 0;

    if (!active.U) active.U.reset(N, 0.);

    DataVector<double> A(ne), B(ne), C(ne), D(ne);

    auto temperature = active.verticallyAverage(inTemperature, active.emesh3, InterpolationMethod::INTERPOLATION_SPLINE);
    for (size_t i = 0; i != ne; ++i) {
        auto material = this->geometry->getMaterial(active.emesh2->at(i));
        double T = temperature[i];
        A[i] = material->A(T);
        B[i] = material->B(T);
        C[i] = material->C(T);
        D[i] = 1e8 * material->D(T);  // cm²/s -> µm²/s
    }

    DataVector<double> J(nn);
    double js = 1e7 / (phys::qe * active.QWheight);
    size_t i = 0;
    for (auto j : inCurrentDensity(active.mesh2, InterpolationMethod::INTERPOLATION_SPLINE)) {
        J[i] = abs(js * j.c1);
        ++i;
    }

    std::vector<DataVector<Tensor2<double>>> Ps;
    std::vector<DataVector<double>> nrs;

    this->writelog(LOG_INFO, "Running diffusion calculations");

    if (shb) {
        nmodes = inWavelength.size();

        if (inLightE.size() != nmodes)
            throw BadInput(this->getId(), "Number of modes in inWavelength ({}) and inLightE ({}) differ", inWavelength.size(),
                           inLightE.size());

        active.modesP.assign(inWavelength.size(), 0.);

        Ps.reserve(nmodes);
        nrs.reserve(nmodes);
        for (size_t i = 0; i != nmodes; ++i) {
            Ps.emplace_back(nn);
            nrs.emplace_back(ne);
        }

        for (size_t i = 0; i != ne; ++i) {
            auto material = this->geometry->getMaterial(active.emesh2->at(i));
            for (size_t n = 0; n != nmodes; ++n) nrs[n][i] = material->Nr(real(inWavelength(n)), temperature[i]).real();
        }

        for (size_t n = 0; n != nmodes; ++n) {
            double wavelength = real(inWavelength(n));
            writelog(LOG_DEBUG, "Mode {} wavelength: {} nm", n, wavelength);
            auto P = active.verticallyAverage(inLightE, active.mesh3);
            for (size_t i = 0; i != nn; ++i) {
                Ps[n][i].c00 = (0.5 / Z0) * real(P[i].c0 * conj(P[i].c0) + P[i].c1 * conj(P[i].c1));
                Ps[n][i].c11 = (0.5 / Z0) * real(P[i].c2 * conj(P[i].c2));
            }
        }
    }

    unsigned loop = 0;

    std::unique_ptr<FemMatrix> K;

    toterr = 0.;

    DataVector<double> F(N);
    DataVector<double> resid(N);

    switch (this->algorithm) {
        case ALGORITHM_CHOLESKY: K.reset(new DpbMatrix(this, N, 3)); break;
        case ALGORITHM_GAUSS: K.reset(new DgbMatrix(this, N, 3)); break;
        case ALGORITHM_ITERATIVE: K.reset(new SparseBandMatrix(this, N, 3)); break;
    }

    while (true) {
        // Set stiffness matrix and load vector
        this->writelog(LOG_DETAIL, "Setting up matrix system (size={})", K->size);
        K->clear();
        F.fill(0.);

        for (size_t ie = 0; ie < ne; ++ie)
            setLocalMatrix(*K, F, ElementParams3D(active, ie), A, B, C, D, active.U.data(), J.data());

        write_debug("{}: Iteration {}", this->getId(), loop);

        // Add SHB
        if (shb) {
            std::fill(active.modesP.begin(), active.modesP.end(), 0.);
            for (size_t n = 0; n != nmodes; ++n) {
                double wavelength = real(inWavelength(n));
                double factor = inv_hc * wavelength;
                auto gain = inGain(active.emesh2, wavelength, InterpolationMethod::INTERPOLATION_SPLINE);
                auto dgdn = inGain(Gain::DGDN, active.emesh2, wavelength, InterpolationMethod::INTERPOLATION_SPLINE);
                const Tensor2<double>* Pdata = Ps[n].data();
                for (size_t ie = 0; ie < ne; ++ie) {
                    ElementParams3D el(active, ie);

                    Tensor2<double> g = nrs[n][ie] * gain[ie];
                    Tensor2<double> dg = nrs[n][ie] * dgdn[ie];
                    Tensor2<double> p = integrateBilinear(el.X, el.Y, Pdata + ie);
                    active.modesP[n] += p.c00 * g.c00 + p.c11 * g.c11;
                    g *= factor;
                    dg *= factor;
                    double ug =
                        0.25 * (active.U[el.i00] + active.U[el.i02] + active.U[el.i20] + active.U[el.i22] +
                                0.25 * (el.X * (active.U[el.i01] - active.U[el.i03] + active.U[el.i21] - active.U[el.i23]) +
                                        el.Y * (active.U[el.i10] + active.U[el.i12] - active.U[el.i30] - active.U[el.i32])));
                    addLocalBurningMatrix(*K, F, el, g, dg, ug, Pdata);
                }
                active.modesP[n] *= 1e-13 * active.QWheight;
                // 10⁻¹³ from µm to cm conversion and conversion to mW (r dr), (...) — photon energy
                writelog(LOG_DEBUG, "{}: Mode {} burned power: {} mW", this->getId(), n, active.modesP[n]);
            }
        }

        // Set derivatives to 0 at the edges
        size_t nl = active.emesh2->lon()->size(), nt = active.emesh2->tran()->size();
        for (size_t i = 0; i <= nt; ++i) {
            K->setBC(F, active.mesh2->index(0, i, 0) + 1, 0.);
            K->setBC(F, active.mesh2->index(nl, i, 0) + 1, 0.);
        }
        for (size_t i = 0; i <= nl; ++i) {
            K->setBC(F, active.mesh2->index(i, 0, 0) + 2, 0.);
            K->setBC(F, active.mesh2->index(i, nt, 0) + 2, 0.);
        }

#ifndef NDEBUG
        double* kend = K->data + K->size * K->kd;
        for (double* pk = K->data; pk != kend; ++pk) {
            if (isnan(*pk) || isinf(*pk))
                throw ComputationError(this->getId(), "Error in stiffness matrix at position {0} ({1})", pk - K->data,
                                       isnan(*pk) ? "nan" : "inf");
        }
        for (auto f = F.begin(); f != F.end(); ++f) {
            if (isnan(*f) || isinf(*f))
                throw ComputationError(this->getId(), "Error in load vector at position {0} ({1})", f - F.begin(),
                                       isnan(*f) ? "nan" : "inf");
        }
#endif

        // Compute current error
        for (auto f = F.begin(), r = resid.begin(); f != F.end(); ++f, ++r) *r = -*f;
        K->addmult(active.U, resid);

        double err = 0.;
        for (auto r = resid.begin(); r != resid.end(); ++r) err += *r * *r;
        double denorm = 0.;
        for (auto f = F.begin(); f != F.end(); ++f) denorm += *f * *f;
        err = 100. * sqrt(err / denorm);

        // Do next calculation step
        if (loop != 0) this->writelog(LOG_RESULT, "Loop {:d}({:d}) @ active region {}: error = {:g}%", loop, loopno, act, err);
        ++loopno;
        ++loop;
        if (err < maxerr || ((loops != 0 && loop >= loops))) break;

        // TODO add linear mixing with the previous solution
        K->solve(F, active.U);
    }

    outCarriersConcentration.fireChanged();

    return toterr;
}

double Diffusion3DSolver::get_burning_integral_for_mode(size_t mode) const {
    if (mode >= inLightE.size()) throw BadInput(this->getId(), "Mode index out of range");
    size_t i = 0;
    double res = 0.;
    for (const auto& iactive : this->active) {
        const auto& active = iactive.second;
        if (mode >= active.modesP.size()) throw Exception("{}: SHB not computed for active region {}", this->getId(), i);
        res += active.modesP[mode];
        ++i;
    }
    return res;
}

double Diffusion3DSolver::get_burning_integral() const {
    double res = 0.;
    for (size_t i = 0; i != inLightE.size(); ++i) res += get_burning_integral_for_mode(i);
    return res;
}

Diffusion3DSolver::ConcentrationDataImpl::ConcentrationDataImpl(const Diffusion3DSolver* solver,
                                                                shared_ptr<const plask::MeshD<3>> dest_mesh,
                                                                InterpolationMethod interp)
    : solver(solver), destination_mesh(dest_mesh), interpolationFlags(InterpolationFlags(solver->geometry)) {
    concentrations.reserve(solver->active.size());

    if (interp == InterpolationMethod::INTERPOLATION_DEFAULT || interp == InterpolationMethod::INTERPOLATION_SPLINE) {
        for (const auto& iactive : solver->active) {
            const auto& active = iactive.second;
            if (!active.U) throw NoValue("Carriers concentration");
            concentrations.emplace_back(LazyData<double>(dest_mesh->size(), [this, active](size_t i) -> double {
                auto point = destination_mesh->at(i);

                double x = interpolationFlags.wrap(0, point.c0);
                assert(active.mesh2->lon()->at(0) <= x && x <= active.mesh2->lon()->at(active.mesh2->lon()->size() - 1));
                size_t il = active.mesh2->lon()->findIndex(x);
                if (il != 0) --il;
                x -= active.mesh2->lon()->at(il);

                double y = interpolationFlags.wrap(1, point.c1);
                assert(active.mesh2->tran()->at(0) <= x && x <= active.mesh2->tran()->at(active.mesh2->tran()->size() - 1));
                size_t it = active.mesh2->tran()->findIndex(x);
                if (it != 0) --it;
                y -= active.mesh2->tran()->at(it);

                ElementParams3D e(active, il, it);

                const double x2 = x * x, y2 = y * y;
                const double x3 = x2 * x, y3 = y2 * y;
                const double X2 = e.X * e.X, Y2 = e.Y * e.Y;
                const double X3 = X2 * e.X, Y3 = Y2 * e.Y;

                return (e.X * x *
                            (-x * y2 * (e.X - x) * (3 * e.Y - 2 * y) * active.U[e.i23] -
                             x * (e.X - x) * (Y3 - 3 * e.Y * y2 + 2 * y3) * active.U[e.i03] +
                             y2 * (3 * e.Y - 2 * y) * (X2 - 2 * e.X * x + x2) * active.U[e.i21] +
                             (X2 - 2 * e.X * x + x2) * (Y3 - 3 * e.Y * y2 + 2 * y3) * active.U[e.i01]) +
                        e.Y * y *
                            (-x2 * y * (3 * e.X - 2 * x) * (e.Y - y) * active.U[e.i32] +
                             x2 * (3 * e.X - 2 * x) * (Y2 - 2 * e.Y * y + y2) * active.U[e.i12] -
                             y * (e.Y - y) * (X3 - 3 * e.X * x2 + 2 * x3) * active.U[e.i30] +
                             (X3 - 3 * e.X * x2 + 2 * x3) * (Y2 - 2 * e.Y * y + y2) * active.U[e.i10]) +
                        x2 * y2 * (3 * e.X - 2 * x) * (3 * e.Y - 2 * y) * active.U[e.i22] +
                        x2 * (3 * e.X - 2 * x) * (Y3 - 3 * e.Y * y2 + 2 * y3) * active.U[e.i02] +
                        y2 * (3 * e.Y - 2 * y) * (X3 - 3 * e.X * x2 + 2 * x3) * active.U[e.i20] +
                        (X3 - 3 * e.X * x2 + 2 * x3) * (Y3 - 3 * e.Y * y2 + 2 * y3) * active.U[e.i00]) /
                       (X3 * Y3);
            }));
        }
    } else {
        for (const auto& iactive : solver->active) {
            const auto& active = iactive.second;
            if (!active.U) throw NoValue("Carriers concentration");
            DataVector<double> conc(active.U.size() / 2);
            DataVector<double>::iterator c = conc.begin();
            for (auto u = active.U.begin(); u < active.U.end(); u += 2, ++c) *c = *u;
            concentrations.emplace_back(interpolate(active.mesh2, conc, dest_mesh, interp, interpolationFlags));
        }
    }
}

double Diffusion3DSolver::ConcentrationDataImpl::at(size_t i) const {
    auto point = interpolationFlags.wrap(destination_mesh->at(i));
    bool found = false;
    size_t an;
    for (const auto& iactive : solver->active) {
        const auto& active = iactive.second;
        if (solver->mesh->vert()->at(active.bottom) <= point.c2 && point.c2 <= solver->mesh->vert()->at(active.top)) {
            // Make sure we have concentration only in the quantum wells
            // TODO maybe more optimal approach would be reasonable?
            if (solver->mesh->lon()->at(active.back) <= point.c0 && point.c0 <= solver->mesh->lon()->at(active.front) &&
                solver->mesh->tran()->at(active.left) <= point.c1 && point.c1 <= solver->mesh->tran()->at(active.right)) {
                for (auto qw : active.QWs)
                    if (qw.first <= point.c2 && point.c2 < qw.second) {
                        found = true;
                        an = iactive.first;
                        break;
                    }
            }
            break;
        }
    }
    if (!found) return 0.;
    return concentrations[an][i];
}

const LazyData<double> Diffusion3DSolver::getConcentration(CarriersConcentration::EnumType what,
                                                           shared_ptr<const plask::MeshD<3>> dest_mesh,
                                                           InterpolationMethod interpolation) const {
    if (what != CarriersConcentration::MAJORITY && what != CarriersConcentration::PAIRS) {
        return LazyData<double>(dest_mesh->size(), NAN);
    }
    return LazyData<double>(new Diffusion3DSolver::ConcentrationDataImpl(this, dest_mesh, interpolation));
}

}}}  // namespace plask::electrical::diffusion
