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
#include <type_traits>

#include "electr3d.hpp"

namespace plask { namespace electrical { namespace shockley {

ElectricalFem3DSolver::ElectricalFem3DSolver(const std::string& name)
    : FemSolverWithMaskedMesh<Geometry3D, plask::RectangularMesh<3>>(name),
      pcond(5.),
      ncond(50.),
      loopno(0),
      default_junction_conductivity(Tensor2<double>(0., 5.)),
      maxerr(0.05),
      outVoltage(this, &ElectricalFem3DSolver::getVoltage),
      outCurrentDensity(this, &ElectricalFem3DSolver::getCurrentDensity),
      outHeat(this, &ElectricalFem3DSolver::getHeatDensity),
      outConductivity(this, &ElectricalFem3DSolver::getConductivity),
      convergence(CONVERGENCE_FAST) {
    potential.reset();
    current.reset();
    inTemperature = 300.;
    junction_conductivity.reset(1, default_junction_conductivity);
    algorithm = ALGORITHM_ITERATIVE;
}

ElectricalFem3DSolver::~ElectricalFem3DSolver() {}

void ElectricalFem3DSolver::loadConfiguration(XMLReader& source, Manager& manager) {
    while (source.requireTagOrEnd()) parseConfiguration(source, manager);
}

void ElectricalFem3DSolver::parseConfiguration(XMLReader& source, Manager& manager) {
    std::string param = source.getNodeName();

    if (param == "voltage")
        readBoundaryConditions(manager, source, voltage_boundary);

    else if (param == "loop") {
        if (source.hasAttribute("start-cond") || source.hasAttribute("start-cond-inplane")) {
            double c0 = default_junction_conductivity.c00, c1 = default_junction_conductivity.c11;
            auto vert = source.getAttribute<std::string>("start-cond");
            if (vert) {
                if (vert->find(',') == std::string::npos) {
                    c1 = boost::lexical_cast<double>(*vert);
                } else {
                    if (source.hasAttribute("start-cond-inplane"))
                        throw XMLException(
                            source, "tag attribute 'start-cond' has two values, but attribute 'start-cond-inplane' is also provided");
                    auto values = splitString2(*vert, ',');
                    c0 = boost::lexical_cast<double>(values.first);
                    c1 = boost::lexical_cast<double>(values.second);
                }
            }
            c0 = source.getAttribute<double>("start-cond-inplane", c0);
            this->setCondJunc(Tensor2<double>(c0, c1));
        }
        convergence = source.enumAttribute<Convergence>("convergence")
                        .value("fast", CONVERGENCE_FAST)
                        .value("stable", CONVERGENCE_STABLE)
                        .get(convergence);
        maxerr = source.getAttribute<double>("maxerr", maxerr);
        source.requireTagEnd();
    }

    else if (param == "contacts") {
        pcond = source.getAttribute<double>("pcond", pcond);
        ncond = source.getAttribute<double>("ncond", ncond);
        source.requireTagEnd();
    }

    else if (!this->parseFemConfiguration(source, manager)) {
        this->parseStandardConfiguration(source, manager);
    }
}

void ElectricalFem3DSolver::setupActiveRegions() {
    if (!geometry || !mesh) {
        if (junction_conductivity.size() != 1) {
            Tensor2<double> condy(0., 0.);
            for (auto cond : junction_conductivity) condy += cond;
            junction_conductivity.reset(1, condy / double(junction_conductivity.size()));
        }
        return;
    }

    setupMaskedMesh();

    shared_ptr<RectangularMesh<3>> points = mesh->getElementMesh();

    std::map<size_t, Active::Region> regions;
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
                            regions[num] = Active::Region(start, ver, lon, tra);
                            if (nreg < num) nreg = num;
                        } else {
                            Active::Region& region = found->second;
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
                        Active::Region& region = found->second;
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
                    regions[num] = Active::Region(start, points->axis[2]->size(), lon, tra);
                } else {
                    Active::Region& region = found->second;
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

    size_t condsize = 0;
    active.resize(nreg);

    for (auto& ireg : regions) {
        size_t num = ireg.first - 1;
        Active::Region& reg = ireg.second;
        double height = this->mesh->axis[2]->at(reg.top) - this->mesh->axis[2]->at(reg.bottom);
        active[num] = Active(condsize, reg, height);
        condsize += (reg.right - reg.left) * (reg.front - reg.back);
        this->writelog(LOG_DETAIL, "Detected junction {0} thickness = {1}nm", num, 1e3 * height);
        this->writelog(LOG_DEBUG, "Junction {0} span: [{1},{3},{5}]-[{2},{4},{6}]", num, reg.back, reg.front, reg.left, reg.right,
                       reg.bottom, reg.top);
    }

    if (junction_conductivity.size() != condsize) {
        Tensor2<double> condy(0., 0.);
        for (auto cond : junction_conductivity) condy += cond;
        junction_conductivity.reset(condsize, condy / double(junction_conductivity.size()));
    }
}

void ElectricalFem3DSolver::onInitialize() {
    if (!geometry) throw NoGeometryException(getId());
    if (!mesh) throw NoMeshException(getId());
    setupActiveRegions();
    loopno = 0;
    potential.reset(maskedMesh->size(), 0.);
    current.reset(maskedMesh->getElementsCount(), vec(0., 0., 0.));
    conds.reset(maskedMesh->getElementsCount());
}

void ElectricalFem3DSolver::onInvalidate() {
    conds.reset();
    potential.reset();
    current.reset();
    heat.reset();
    junction_conductivity.reset(1, default_junction_conductivity);
}

LazyData<double> ElectricalFem3DSolver::loadConductivity() {
    auto midmesh = (this->maskedMesh)->getElementMesh();
    auto temperature = inTemperature(midmesh);

    for (auto e : this->maskedMesh->elements()) {
        size_t i = e.getIndex();
        Vec<3, double> midpoint = e.getMidpoint();

        auto roles = this->geometry->getRolesAt(midpoint);
        if (size_t actn = isActive(midpoint)) {
            const auto& act = active[actn - 1];
            conds[i] = junction_conductivity[act.offset + act.ld * e.getIndex1() + e.getIndex0()];
            if (isnan(conds[i].c11) || abs(conds[i].c11) < 1e-16) conds[i].c11 = 1e-16;
        } else if (roles.find("p-contact") != roles.end()) {
            conds[i] = Tensor2<double>(pcond, pcond);
        } else if (roles.find("n-contact") != roles.end()) {
            conds[i] = Tensor2<double>(ncond, ncond);
        } else
            conds[i] = this->geometry->getMaterial(midpoint)->cond(temperature[i]);
    }

    return temperature;
}

void ElectricalFem3DSolver::saveConductivity() {
    for (size_t n = 0; n < active.size(); ++n) {
        const auto& act = active[n];
        size_t v = (act.top + act.bottom) / 2;
        for (size_t t = act.left; t != act.right; ++t) {
            size_t offset = act.offset + act.ld * t;
            for (size_t l = act.back; l != act.front; ++l)
                junction_conductivity[offset + l] = conds[this->maskedMesh->element(l, t, v).getIndex()];
        }
    }
}

void ElectricalFem3DSolver::setMatrix(FemMatrix& A,
                                      DataVector<double>& B,
                                      const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary, double>& bvoltage,
                                      const LazyData<double>& temperature) {
    this->writelog(LOG_DETAIL, "Setting up matrix system ({})", A.describe());

    // Update junction conductivities
    if (loopno != 0) {
        for (auto elem : maskedMesh->elements()) {
            if (size_t nact = isActive(elem)) {
                size_t index = elem.getIndex(), lll = elem.getLoLoLoIndex(), uuu = elem.getUpUpUpIndex();
                size_t back = maskedMesh->index0(lll), front = maskedMesh->index0(uuu), left = maskedMesh->index1(lll),
                       right = maskedMesh->index1(uuu);
                const Active& act = active[nact - 1];
                double U =
                    0.25 *
                    (-potential[maskedMesh->index(back, left, act.bottom)] - potential[maskedMesh->index(front, left, act.bottom)] -
                     potential[maskedMesh->index(back, right, act.bottom)] -
                     potential[maskedMesh->index(front, right, act.bottom)] + potential[maskedMesh->index(back, left, act.top)] +
                     potential[maskedMesh->index(front, left, act.top)] + potential[maskedMesh->index(back, right, act.top)] +
                     potential[maskedMesh->index(front, right, act.top)]);
                double jy = 0.1 * conds[index].c11 * U / act.height;  // [j] = kA/cm²
                size_t tidx = this->maskedMesh->element(elem.getIndex0(), elem.getIndex1(), (act.top + act.bottom) / 2).getIndex();
                Tensor2<double> cond = activeCond(nact - 1, U, jy, temperature[tidx]);
                switch (convergence) {
                    case CONVERGENCE_STABLE:
                        cond = 0.5 * (conds[index] + cond);
                    case CONVERGENCE_FAST:
                        conds[index] = cond;
                }
                if (isnan(conds[index].c11) || abs(conds[index].c11) < 1e-16) {
                    conds[index].c11 = 1e-16;
                }
            }
        }
    }

    // Zero the matrix and the load vector
    A.clear();
    B.fill(0.);

    // Set stiffness matrix and load vector
    for (auto elem : maskedMesh->elements()) {
        size_t index = elem.getIndex();

        // nodes numbers for the current element
        size_t idx[8];
        idx[0] = elem.getLoLoLoIndex();  //   z              4-----6
        idx[1] = elem.getUpLoLoIndex();  //   |__y          /|    /|
        idx[2] = elem.getLoUpLoIndex();  //  x/            5-----7 |
        idx[3] = elem.getUpUpLoIndex();  //                | 0---|-2
        idx[4] = elem.getLoLoUpIndex();  //                |/    |/
        idx[5] = elem.getUpLoUpIndex();  //                1-----3
        idx[6] = elem.getLoUpUpIndex();  //
        idx[7] = elem.getUpUpUpIndex();  //

        // element size
        double dx = elem.getUpper0() - elem.getLower0();
        double dy = elem.getUpper1() - elem.getLower1();
        double dz = elem.getUpper2() - elem.getLower2();

        // point and material in the middle of the element
        Vec<3> middle = elem.getMidpoint();
        auto material = geometry->getMaterial(middle);

        // average voltage on the element
        double temp = 0.;
        for (int i = 0; i < 8; ++i) temp += potential[idx[i]];
        temp *= 0.125;

        // electrical conductivity
        double kx, ky = conds[index].c00, kz = conds[index].c11;

        ky *= 1e-6;
        kz *= 1e-6;  // 1/m -> 1/µm
        kx = ky;

        kx /= dx;
        kx *= dy;
        kx *= dz;
        ky *= dx;
        ky /= dy;
        ky *= dz;
        kz *= dx;
        kz *= dy;
        kz /= dz;

        // set symmetric matrix components
        double K[8][8];
        K[0][0] = K[1][1] = K[2][2] = K[3][3] = K[4][4] = K[5][5] = K[6][6] = K[7][7] = (kx + ky + kz) / 9.;

        K[1][0] = K[3][2] = K[5][4] = K[7][6] = (-2. * kx + ky + kz) / 18.;
        K[2][0] = K[3][1] = K[6][4] = K[7][5] = (kx - 2. * ky + kz) / 18.;
        K[4][0] = K[5][1] = K[6][2] = K[7][3] = (kx + ky - 2. * kz) / 18.;

        K[4][2] = K[5][3] = K[6][0] = K[7][1] = (kx - 2. * ky - 2. * kz) / 36.;
        K[4][1] = K[5][0] = K[6][3] = K[7][2] = (-2. * kx + ky - 2. * kz) / 36.;
        K[2][1] = K[3][0] = K[6][5] = K[7][4] = (-2. * kx - 2. * ky + kz) / 36.;

        K[4][3] = K[5][2] = K[6][1] = K[7][0] = -(kx + ky + kz) / 36.;

        for (int i = 0; i < 8; ++i)
            for (int j = 0; j <= i; ++j) A(idx[i], idx[j]) += K[i][j];
    }

    A.applyBC(bvoltage, B);

#ifndef NDEBUG
    double* aend = A.data + A.size;
    for (double* pa = A.data; pa != aend; ++pa) {
        if (isnan(*pa) || isinf(*pa))
            throw ComputationError(getId(), "error in stiffness matrix at position {0} ({1})", pa - A.data,
                                   isnan(*pa) ? "nan" : "inf");
    }
#endif
}

double ElectricalFem3DSolver::compute(unsigned loops) {
    this->initCalculation();

    // store boundary conditions for current mesh
    auto bvoltage = voltage_boundary(maskedMesh, geometry);

    this->writelog(LOG_INFO, "Running electrical calculations");

    unsigned loop = 0;
    double err = 0.;
    toterr = 0.;

    std::unique_ptr<FemMatrix> pA(this->getMatrix());
    FemMatrix& A = *pA.get();

#ifndef NDEBUG
    if (!potential.unique()) this->writelog(LOG_DEBUG, "Potentials data held by something else...");
#endif
    potential = potential.claim();

    DataVector<double> rhs(potential.size());

    auto temperature = loadConductivity();

    bool noactive = (active.size() == 0);
    double minj = 100e-7;  // assume no significant heating below this current

    do {
        setMatrix(A, rhs, bvoltage, temperature);
        A.solve(rhs, potential);

        err = 0.;
        double mcur = 0.;
        for (auto el : maskedMesh->elements()) {
            size_t i = el.getIndex();
            size_t lll = el.getLoLoLoIndex();
            size_t llu = el.getLoLoUpIndex();
            size_t lul = el.getLoUpLoIndex();
            size_t luu = el.getLoUpUpIndex();
            size_t ull = el.getUpLoLoIndex();
            size_t ulu = el.getUpLoUpIndex();
            size_t uul = el.getUpUpLoIndex();
            size_t uuu = el.getUpUpUpIndex();
            auto cur = vec(-0.025 * conds[i].c00 *
                               (-potential[lll] - potential[llu] - potential[lul] - potential[luu] + potential[ull] +
                                potential[ulu] + potential[uul] + potential[uuu]) /
                               (el.getUpper0() - el.getLower0()),  // [j] = kA/cm²
                           -0.025 * conds[i].c00 *
                               (-potential[lll] - potential[llu] + potential[lul] + potential[luu] - potential[ull] -
                                potential[ulu] + potential[uul] + potential[uuu]) /
                               (el.getUpper1() - el.getLower1()),  // [j] = kA/cm²
                           -0.025 * conds[i].c11 *
                               (-potential[lll] + potential[llu] - potential[lul] + potential[luu] - potential[ull] +
                                potential[ulu] - potential[uul] + potential[uuu]) /
                               (el.getUpper2() - el.getLower2())  // [j] = kA/cm²
            );
            if (noactive || isActive(el)) {
                double acur = abs2(cur);
                if (acur > mcur) {
                    mcur = acur;
                    maxcur = cur;
                }
            }
            double delta = abs2(current[i] - cur);
            if (delta > err) err = delta;
            current[i] = cur;
        }
        mcur = sqrt(mcur);
        err = 100. * sqrt(err) / max(mcur, minj);
        if ((loop != 0 || mcur >= minj) && err > toterr) toterr = err;

        ++loopno;
        ++loop;

        this->writelog(LOG_RESULT, "Loop {:d}({:d}): max(j{}) = {:g} kA/cm2, error = {:g}%", loop, loopno, noactive ? "" : "@junc",
                       mcur, err);

    } while ((!iter_params.converged || err > maxerr) && (loops == 0 || loop < loops));

    saveConductivity();

    outVoltage.fireChanged();
    outCurrentDensity.fireChanged();
    outHeat.fireChanged();

    return toterr;
}

void ElectricalFem3DSolver::saveHeatDensity() {
    this->writelog(LOG_DETAIL, "Computing heat densities");

    heat.reset(maskedMesh->getElementsCount());

    for (auto el : maskedMesh->elements()) {
        size_t i = el.getIndex();
        size_t lll = el.getLoLoLoIndex();
        size_t llu = el.getLoLoUpIndex();
        size_t lul = el.getLoUpLoIndex();
        size_t luu = el.getLoUpUpIndex();
        size_t ull = el.getUpLoLoIndex();
        size_t ulu = el.getUpLoUpIndex();
        size_t uul = el.getUpUpLoIndex();
        size_t uuu = el.getUpUpUpIndex();
        double dvx = -0.25e6 *
                     (-potential[lll] - potential[llu] - potential[lul] - potential[luu] + potential[ull] + potential[ulu] +
                      potential[uul] + potential[uuu]) /
                     (el.getUpper0() - el.getLower0());  // 1e6 - from µm to m
        double dvy = -0.25e6 *
                     (-potential[lll] - potential[llu] + potential[lul] + potential[luu] - potential[ull] - potential[ulu] +
                      potential[uul] + potential[uuu]) /
                     (el.getUpper1() - el.getLower1());  // 1e6 - from µm to m
        double dvz = -0.25e6 *
                     (-potential[lll] + potential[llu] - potential[lul] + potential[luu] - potential[ull] + potential[ulu] -
                      potential[uul] + potential[uuu]) /
                     (el.getUpper2() - el.getLower2());  // 1e6 - from µm to m
        auto midpoint = el.getMidpoint();
        if (geometry->getMaterial(midpoint)->kind() == Material::EMPTY || geometry->hasRoleAt("noheat", midpoint))
            heat[i] = 0.;
        else {
            heat[i] = conds[i].c00 * dvx * dvx + conds[i].c00 * dvy * dvy + conds[i].c11 * dvz * dvz;
        }
    }
}

double ElectricalFem3DSolver::integrateCurrent(size_t vindex, bool onlyactive) {
    if (!potential) throw NoValue("current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    double result = 0.;
    for (size_t i = 0; i < mesh->axis[0]->size() - 1; ++i) {
        for (size_t j = 0; j < mesh->axis[1]->size() - 1; ++j) {
            auto element = maskedMesh->element(i, j, vindex);
            if (!onlyactive || isActive(element.getMidpoint())) {
                size_t index = element.getIndex();
                if (index != RectangularMaskedMesh3D::Element::UNKNOWN_ELEMENT_INDEX)
                    result += current[index].c2 * element.getSize0() * element.getSize1();
            }
        }
    }
    if (geometry->isSymmetric(Geometry::DIRECTION_LONG)) result *= 2.;
    if (geometry->isSymmetric(Geometry::DIRECTION_TRAN)) result *= 2.;
    return result * 0.01;  // kA/cm² µm² -->  mA
}

double ElectricalFem3DSolver::getTotalCurrent(size_t nact) {
    if (nact >= active.size()) throw BadInput(this->getId(), "wrong active region number");
    const auto& act = active[nact];
    // Find the average of the active region
    size_t level = (act.bottom + act.top) / 2;
    return integrateCurrent(level, true);
}

const LazyData<double> ElectricalFem3DSolver::getVoltage(shared_ptr<const MeshD<3>> dest_mesh, InterpolationMethod method) const {
    if (!potential) throw NoValue("voltage");
    this->writelog(LOG_DEBUG, "Getting potential");
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    if (maskedMesh->full())
        return interpolate(mesh, potential, dest_mesh, method, geometry);
    else
        return interpolate(maskedMesh, potential, dest_mesh, method, geometry);
}

const LazyData<Vec<3>> ElectricalFem3DSolver::getCurrentDensity(shared_ptr<const MeshD<3>> dest_mesh, InterpolationMethod method) {
    if (!potential) throw NoValue("current density");
    this->writelog(LOG_DEBUG, "Getting current density");
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    InterpolationFlags flags(geometry, InterpolationFlags::Symmetry::NPP, InterpolationFlags::Symmetry::PNP,
                             InterpolationFlags::Symmetry::PPN);
    if (maskedMesh->full()) {
        auto result = interpolate(mesh->getElementMesh(), current, dest_mesh, method, flags);
        return LazyData<Vec<3>>(result.size(), [this, dest_mesh, result, flags](size_t i) {
            return this->geometry->getChildBoundingBox().contains(flags.wrap(dest_mesh->at(i))) ? result[i] : Vec<3>(0., 0., 0.);
        });
    } else {
        auto result = interpolate(maskedMesh->getElementMesh(), current, dest_mesh, method, flags);
        return LazyData<Vec<3>>(result.size(), [result](size_t i) {
            // Masked mesh always returns NaN outside of itself
            auto val = result[i];
            return isnan(val) ? Vec<3>(0., 0., 0.) : val;
        });
    }
}

const LazyData<double> ElectricalFem3DSolver::getHeatDensity(shared_ptr<const MeshD<3>> dest_mesh, InterpolationMethod method) {
    if (!potential) throw NoValue("heat density");
    this->writelog(LOG_DEBUG, "Getting heat density");
    if (!heat) saveHeatDensity();  // we will compute heats only if they are needed
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    InterpolationFlags flags(geometry);
    if (maskedMesh->full()) {
        auto result = interpolate(mesh->getElementMesh(), heat, dest_mesh, method, flags);
        return LazyData<double>(result.size(), [this, dest_mesh, result, flags](size_t i) {
            return this->geometry->getChildBoundingBox().contains(flags.wrap(dest_mesh->at(i))) ? result[i] : 0.;
        });
    } else {
        auto result = interpolate(maskedMesh->getElementMesh(), heat, dest_mesh, method, flags);
        return LazyData<double>(result.size(), [result](size_t i) {
            // Masked mesh always returns NaN outside of itself
            auto val = result[i];
            return isnan(val) ? 0. : val;
        });
    }
}

const LazyData<Tensor2<double>> ElectricalFem3DSolver::getConductivity(shared_ptr<const MeshD<3>> dest_mesh,
                                                                       InterpolationMethod /*method*/) {
    initCalculation();
    writelog(LOG_DEBUG, "Getting conductivities");
    loadConductivity();
    InterpolationFlags flags(geometry);
    return interpolate(maskedMesh->getElementMesh(), conds, dest_mesh, INTERPOLATION_NEAREST, flags);
}

double ElectricalFem3DSolver::getTotalEnergy() {
    double W = 0.;
    auto T = inTemperature(maskedMesh->getElementMesh());
    for (auto el : maskedMesh->elements()) {
        size_t lll = el.getLoLoLoIndex();
        size_t llu = el.getLoLoUpIndex();
        size_t lul = el.getLoUpLoIndex();
        size_t luu = el.getLoUpUpIndex();
        size_t ull = el.getUpLoLoIndex();
        size_t ulu = el.getUpLoUpIndex();
        size_t uul = el.getUpUpLoIndex();
        size_t uuu = el.getUpUpUpIndex();
        double dvx = -0.25e6 *
                     (-potential[lll] - potential[llu] - potential[lul] - potential[luu] + potential[ull] + potential[ulu] +
                      potential[uul] + potential[uuu]) /
                     (el.getUpper0() - el.getLower0());  // 1e6 - from µm to m
        double dvy = -0.25e6 *
                     (-potential[lll] - potential[llu] + potential[lul] + potential[luu] - potential[ull] - potential[ulu] +
                      potential[uul] + potential[uuu]) /
                     (el.getUpper1() - el.getLower1());  // 1e6 - from µm to m
        double dvz = -0.25e6 *
                     (-potential[lll] + potential[llu] - potential[lul] + potential[luu] - potential[ull] + potential[ulu] -
                      potential[uul] + potential[uuu]) /
                     (el.getUpper2() - el.getLower2());  // 1e6 - from µm to m
        double w = this->geometry->getMaterial(el.getMidpoint())->eps(T[el.getIndex()]) * (dvx * dvx + dvy * dvy + dvz * dvz);
        double d0 = el.getUpper0() - el.getLower0();
        double d1 = el.getUpper1() - el.getLower1();
        double d2 = el.getUpper2() - el.getLower2();
        // TODO add outsides of computational area
        W += 0.5e-18 * phys::epsilon0 * d0 * d1 * d2 * w;  // 1e-18 µm³ -> m³
    }
    return W;
}

double ElectricalFem3DSolver::getCapacitance() {
    if (this->voltage_boundary.size() != 2) {
        throw BadInput(this->getId(), "cannot estimate applied voltage (exactly 2 voltage boundary conditions required)");
    }

    double U = voltage_boundary[0].value - voltage_boundary[1].value;

    return 2e12 * getTotalEnergy() / (U * U);  // 1e12 F -> pF
}

double ElectricalFem3DSolver::getTotalHeat() {
    double W = 0.;
    if (!heat) saveHeatDensity();  // we will compute heats only if they are needed
    for (auto el : this->maskedMesh->elements()) {
        double d0 = el.getUpper0() - el.getLower0();
        double d1 = el.getUpper1() - el.getLower1();
        double d2 = el.getUpper2() - el.getLower2();
        W += 1e-15 * d0 * d1 * d2 * heat[el.getIndex()];  // 1e-15 µm³ -> m³, W -> mW
    }
    if (geometry->isSymmetric(Geometry::DIRECTION_LONG)) W *= 2.;
    if (geometry->isSymmetric(Geometry::DIRECTION_TRAN)) W *= 2.;
    return W;
}

}}}  // namespace plask::electrical::shockley
