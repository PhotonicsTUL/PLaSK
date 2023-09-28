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
#include "electr2d.hpp"

namespace plask { namespace electrical { namespace shockley {

template <typename Geometry2DType>
ElectricalFem2DSolver<Geometry2DType>::ElectricalFem2DSolver(const std::string& name)
    : FemSolverWithMaskedMesh<Geometry2DType, RectangularMesh<2>>(name),
      pcond(5.),
      ncond(50.),
      loopno(0),
      default_junction_conductivity(Tensor2<double>(0., 5.)),
      maxerr(0.05),
      outVoltage(this, &ElectricalFem2DSolver<Geometry2DType>::getVoltage),
      outCurrentDensity(this, &ElectricalFem2DSolver<Geometry2DType>::getCurrentDensities),
      outHeat(this, &ElectricalFem2DSolver<Geometry2DType>::getHeatDensities),
      outConductivity(this, &ElectricalFem2DSolver<Geometry2DType>::getConductivity),
      convergence(CONVERGENCE_FAST) {
    onInvalidate();
    inTemperature = 300.;
    junction_conductivity.reset(1, default_junction_conductivity);
}

template <typename Geometry2DType>
void ElectricalFem2DSolver<Geometry2DType>::loadConfiguration(XMLReader& source, Manager& manager) {
    while (source.requireTagOrEnd()) parseConfiguration(source, manager);
}

template <typename Geometry2DType>
void ElectricalFem2DSolver<Geometry2DType>::parseConfiguration(XMLReader& source, Manager& manager) {
    std::string param = source.getNodeName();

    if (param == "potential") source.throwException("<potential> boundary conditions have been permanently renamed to <voltage>");

    if (param == "voltage")
        this->readBoundaryConditions(manager, source, voltage_boundary);

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

template <typename Geometry2DType> ElectricalFem2DSolver<Geometry2DType>::~ElectricalFem2DSolver() {}

template <typename Geometry2DType> void ElectricalFem2DSolver<Geometry2DType>::setupActiveRegions() {
    this->invalidate();

    if (!this->geometry || !this->mesh) {
        if (junction_conductivity.size() != 1) {
            Tensor2<double> condy(0., 0.);
            for (auto cond : junction_conductivity) condy += cond;
            junction_conductivity.reset(1, condy / double(junction_conductivity.size()));
        }
        return;
    }

    this->setupMaskedMesh();

    auto points = this->mesh->getElementMesh();

    std::vector<typename Active::Region> regions;

    for (size_t r = 0; r < points->axis[1]->size(); ++r) {
        size_t prev = 0;
        shared_ptr<Material> material;
        for (size_t c = 0; c < points->axis[0]->size(); ++c) {  // In the (possible) active region
            auto point = points->at(c, r);
            size_t num = isActive(point);

            if (num) {  // here we are inside the active region
                if (regions.size() >= num && regions[num - 1].warn) {
                    if (!material)
                        material = this->geometry->getMaterial(points->at(c, r));
                    else if (*material != *this->geometry->getMaterial(points->at(c, r))) {
                        writelog(LOG_WARNING, "Junction {} is laterally non-uniform", num - 1);
                        regions[num - 1].warn = false;
                    }
                }
                regions.resize(max(regions.size(), num));
                auto& reg = regions[num - 1];
                if (prev != num) {  // this region starts in the current row
                    if (reg.top < r) {
                        throw Exception("{0}: Junction {1} is disjoint", this->getId(), num - 1);
                    }
                    if (reg.bottom >= r)
                        reg.bottom = r;  // first row
                    else if (reg.rowr <= c)
                        throw Exception("{0}: Junction {1} is disjoint", this->getId(), num - 1);
                    reg.top = r + 1;
                    reg.rowl = c;
                    if (reg.left > reg.rowl) reg.left = reg.rowl;
                }
            }
            if (prev && prev != num) {  // previous region ended
                auto& reg = regions[prev - 1];
                if (reg.bottom < r && reg.rowl >= c) throw Exception("{0}: Junction {1} is disjoint", this->getId(), prev - 1);
                reg.rowr = c;
                if (reg.right < reg.rowr) reg.right = reg.rowr;
            }
            prev = num;
        }
        if (prev)  // junction reached the edge
            regions[prev - 1].rowr = regions[prev - 1].right = points->axis[0]->size();
    }

    size_t condsize = 0;
    active.clear();
    active.reserve(regions.size());
    size_t i = 0;
    for (auto& reg : regions) {
        if (reg.bottom == std::numeric_limits<size_t>::max()) reg.bottom = reg.top = 0;
        active.emplace_back(condsize, reg.left, reg.right, reg.bottom, reg.top,
                            this->mesh->axis[1]->at(reg.top) - this->mesh->axis[1]->at(reg.bottom));
        condsize += reg.right - reg.left;
        this->writelog(LOG_DETAIL, "Detected junction {0} thickness = {1}nm", i++, 1e3 * active.back().height);
        this->writelog(LOG_DEBUG, "Junction {0} span: [{1},{3}]-[{2},{4}]", i - 1, reg.left, reg.right, reg.bottom, reg.top);
    }

    if (junction_conductivity.size() != condsize) {
        Tensor2<double> condy(0., 0.);
        for (auto cond : junction_conductivity) condy += cond;
        junction_conductivity.reset(max(condsize, size_t(1)), condy / double(junction_conductivity.size()));
    }
}

template <typename Geometry2DType> void ElectricalFem2DSolver<Geometry2DType>::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    setupActiveRegions();
    loopno = 0;
    potentials.reset(this->maskedMesh->size(), 0.);
    currents.reset(this->maskedMesh->getElementsCount(), vec(0., 0.));
    conds.reset(this->maskedMesh->getElementsCount());
}

template <typename Geometry2DType> void ElectricalFem2DSolver<Geometry2DType>::onInvalidate() {
    conds.reset();
    potentials.reset();
    currents.reset();
    heats.reset();
    junction_conductivity.reset(1, default_junction_conductivity);
}

template <>
inline void ElectricalFem2DSolver<Geometry2DCartesian>::setLocalMatrix(double&,
                                                                       double&,
                                                                       double&,
                                                                       double&,
                                                                       double&,
                                                                       double&,
                                                                       double&,
                                                                       double&,
                                                                       double&,
                                                                       double&,
                                                                       double,
                                                                       double,
                                                                       const Vec<2, double>&) {
    return;
}

template <>
inline void ElectricalFem2DSolver<Geometry2DCylindrical>::setLocalMatrix(double& k44,
                                                                         double& k33,
                                                                         double& k22,
                                                                         double& k11,
                                                                         double& k43,
                                                                         double& k21,
                                                                         double& k42,
                                                                         double& k31,
                                                                         double& k32,
                                                                         double& k41,
                                                                         double,
                                                                         double,
                                                                         const Vec<2, double>& midpoint) {
    double r = midpoint.rad_r();
    k44 = r * k44;
    k33 = r * k33;
    k22 = r * k22;
    k11 = r * k11;
    k43 = r * k43;
    k21 = r * k21;
    k42 = r * k42;
    k31 = r * k31;
    k32 = r * k32;
    k41 = r * k41;
}

/// Set stiffness matrix + load vector
template <typename Geometry2DType>
void ElectricalFem2DSolver<Geometry2DType>::setMatrix(
    FemMatrix& A,
    DataVector<double>& B,
    const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary, double>& bvoltage,
    const LazyData<double>& temperature) {
    this->writelog(LOG_DETAIL, "Setting up matrix system ({})", A.describe());

    // Update junction conductivities
    if (loopno != 0) {
        for (auto e : this->maskedMesh->elements()) {
            if (size_t nact = isActive(e)) {
                size_t i = e.getIndex();
                size_t left = this->maskedMesh->index0(e.getLoLoIndex());
                size_t right = this->maskedMesh->index0(e.getUpLoIndex());
                const Active& act = active[nact - 1];
                double U =
                    0.5 *
                    (potentials[this->maskedMesh->index(left, act.top)] - potentials[this->maskedMesh->index(left, act.bottom)] +
                     potentials[this->maskedMesh->index(right, act.top)] - potentials[this->maskedMesh->index(right, act.bottom)]);
                double jy = 0.1 * conds[i].c11 * U / act.height;  // [j] = kA/cm²
                size_t ti = this->maskedMesh->element(e.getIndex0(), (act.top + act.bottom) / 2).getIndex();
                Tensor2<double> cond = activeCond(nact - 1, U, jy, temperature[ti]);
                switch (convergence) {
                    case CONVERGENCE_STABLE:
                        cond = 0.5 * (conds[i] + cond);
                    case CONVERGENCE_FAST:
                        conds[i] = cond;
                }
                if (isnan(conds[i].c11) || abs(conds[i].c11) < 1e-16) conds[i].c11 = 1e-16;
            }
        }
    }

    A.clear();
    B.fill(0.);

    // Set stiffness matrix and load vector
    for (auto e : this->maskedMesh->elements()) {
        size_t i = e.getIndex();

        // nodes numbers for the current element
        size_t loleftno = e.getLoLoIndex();
        size_t lorghtno = e.getUpLoIndex();
        size_t upleftno = e.getLoUpIndex();
        size_t uprghtno = e.getUpUpIndex();

        // element size
        double elemwidth = e.getUpper0() - e.getLower0();
        double elemheight = e.getUpper1() - e.getLower1();

        Vec<2, double> midpoint = e.getMidpoint();

        double kx = conds[i].c00;
        double ky = conds[i].c11;

        kx *= elemheight;
        kx /= elemwidth;
        ky *= elemwidth;
        ky /= elemheight;

        // set symmetric matrix components
        double k44, k33, k22, k11, k43, k21, k42, k31, k32, k41;

        k44 = k33 = k22 = k11 = (kx + ky) / 3.;
        k43 = k21 = (-2. * kx + ky) / 6.;
        k42 = k31 = -(kx + ky) / 6.;
        k32 = k41 = (kx - 2. * ky) / 6.;

        // set stiffness matrix
        setLocalMatrix(k44, k33, k22, k11, k43, k21, k42, k31, k32, k41, ky, elemwidth, midpoint);

        A(loleftno, loleftno) += k11;
        A(lorghtno, lorghtno) += k22;
        A(uprghtno, uprghtno) += k33;
        A(upleftno, upleftno) += k44;

        A(lorghtno, loleftno) += k21;
        A(uprghtno, loleftno) += k31;
        A(upleftno, loleftno) += k41;
        A(uprghtno, lorghtno) += k32;
        A(upleftno, lorghtno) += k42;
        A(upleftno, uprghtno) += k43;
    }

    A.applyBC(bvoltage, B);

#ifndef NDEBUG
    double* aend = A.data + A.size;
    for (double* pa = A.data; pa != aend; ++pa) {
        if (isnan(*pa) || isinf(*pa))
            throw ComputationError(this->getId(), "Error in stiffness matrix at position {0} ({1})", pa - A.data,
                                   isnan(*pa) ? "nan" : "inf");
    }
#endif
}

template <typename Geometry2DType> LazyData<double> ElectricalFem2DSolver<Geometry2DType>::loadConductivities() {
    auto midmesh = this->maskedMesh->getElementMesh();
    auto temperature = inTemperature(midmesh);

    for (auto e : this->maskedMesh->elements()) {
        size_t i = e.getIndex();
        Vec<2, double> midpoint = e.getMidpoint();

        auto roles = this->geometry->getRolesAt(midpoint);
        if (size_t actn = isActive(midpoint)) {
            const auto& act = active[actn - 1];
            conds[i] = junction_conductivity[act.offset + e.getIndex0()];
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

template <typename Geometry2DType> void ElectricalFem2DSolver<Geometry2DType>::saveConductivities() {
    for (size_t n = 0; n < active.size(); ++n) {
        const auto& act = active[n];
        for (size_t i = act.left, r = (act.top + act.bottom) / 2; i != act.right; ++i)
            junction_conductivity[act.offset + i] = conds[this->maskedMesh->element(i, r).getIndex()];
    }
}

template <typename Geometry2DType> double ElectricalFem2DSolver<Geometry2DType>::compute(unsigned loops) {
    this->initCalculation();

    heats.reset();

    // Store boundary conditions for current mesh
    auto vconst = voltage_boundary(this->maskedMesh, this->geometry);

    this->writelog(LOG_INFO, "Running electrical calculations");

    unsigned loop = 0;

    std::unique_ptr<FemMatrix> pA(this->getMatrix());
    FemMatrix& A = *pA.get();

    double err = 0.;
    toterr = 0.;

#ifndef NDEBUG
    if (!potentials.unique()) this->writelog(LOG_DEBUG, "Voltage data held by something else...");
#endif
    potentials = potentials.claim();

    DataVector<double> rhs(potentials.size());

    auto temperature = loadConductivities();

    bool noactive = (active.size() == 0);
    double minj = 100e-7;  // assume no significant heating below this current

    do {
        setMatrix(A, rhs, vconst, temperature);
        A.solve(rhs, potentials);

        err = 0.;
        double mcur = 0.;
        for (auto el : this->maskedMesh->elements()) {
            size_t i = el.getIndex();
            size_t loleftno = el.getLoLoIndex();
            size_t lorghtno = el.getUpLoIndex();
            size_t upleftno = el.getLoUpIndex();
            size_t uprghtno = el.getUpUpIndex();
            double dvx = -0.05 * (-potentials[loleftno] + potentials[lorghtno] - potentials[upleftno] + potentials[uprghtno]) /
                         (el.getUpper0() - el.getLower0());  // [j] = kA/cm²
            double dvy = -0.05 * (-potentials[loleftno] - potentials[lorghtno] + potentials[upleftno] + potentials[uprghtno]) /
                         (el.getUpper1() - el.getLower1());  // [j] = kA/cm²
            auto cur = vec(conds[i].c00 * dvx, conds[i].c11 * dvy);
            if (noactive || isActive(el)) {
                double acur = abs2(cur);
                if (acur > mcur) {
                    mcur = acur;
                    maxcur = cur;
                }
            }
            double delta = abs2(currents[i] - cur);
            if (delta > err) err = delta;
            currents[i] = cur;
        }
        mcur = sqrt(mcur);
        err = 100. * sqrt(err) / max(mcur, minj);
        if ((loop != 0 || mcur >= minj) && err > toterr) toterr = err;

        ++loopno;
        ++loop;

        this->writelog(LOG_RESULT, "Loop {:d}({:d}): max(j{}) = {:g} kA/cm2, error = {:g}%", loop, loopno, noactive ? "" : "@junc",
                       mcur, err);

    } while ((!this->iter_params.converged || err > maxerr) && (loops == 0 || loop < loops));

    saveConductivities();

    outVoltage.fireChanged();
    outCurrentDensity.fireChanged();
    outHeat.fireChanged();

    return toterr;
}

template <typename Geometry2DType> void ElectricalFem2DSolver<Geometry2DType>::saveHeatDensities() {
    this->writelog(LOG_DETAIL, "Computing heat densities");

    heats.reset(this->maskedMesh->getElementsCount());

    for (auto e : this->maskedMesh->elements()) {
        size_t i = e.getIndex();
        size_t loleftno = e.getLoLoIndex();
        size_t lorghtno = e.getUpLoIndex();
        size_t upleftno = e.getLoUpIndex();
        size_t uprghtno = e.getUpUpIndex();
        auto midpoint = e.getMidpoint();
        if (this->geometry->getMaterial(midpoint)->kind() == Material::EMPTY || this->geometry->hasRoleAt("noheat", midpoint))
            heats[i] = 0.;
        else {
            double dvx = 0.5e6 * (-potentials[loleftno] + potentials[lorghtno] - potentials[upleftno] + potentials[uprghtno]) /
                         (e.getUpper0() - e.getLower0());  // [grad(dV)] = V/m
            double dvy = 0.5e6 * (-potentials[loleftno] - potentials[lorghtno] + potentials[upleftno] + potentials[uprghtno]) /
                         (e.getUpper1() - e.getLower1());  // [grad(dV)] = V/m
            heats[i] = conds[i].c00 * dvx * dvx + conds[i].c11 * dvy * dvy;
        }
    }
}

template <> double ElectricalFem2DSolver<Geometry2DCartesian>::integrateCurrent(size_t vindex, bool onlyactive) {
    if (!potentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    double result = 0.;
    for (size_t i = 0; i < mesh->axis[0]->size() - 1; ++i) {
        auto element = maskedMesh->element(i, vindex);
        if (!onlyactive || isActive(element.getMidpoint())) {
            size_t index = element.getIndex();
            if (index != RectangularMaskedMesh2D::Element::UNKNOWN_ELEMENT_INDEX) result += currents[index].c1 * element.getSize0();
        }
    }
    if (this->getGeometry()->isSymmetric(Geometry::DIRECTION_TRAN)) result *= 2.;
    return result * geometry->getExtrusion()->getLength() * 0.01;  // kA/cm² µm² -->  mA;
}

template <> double ElectricalFem2DSolver<Geometry2DCylindrical>::integrateCurrent(size_t vindex, bool onlyactive) {
    if (!potentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    double result = 0.;
    for (size_t i = 0; i < mesh->axis[0]->size() - 1; ++i) {
        auto element = maskedMesh->element(i, vindex);
        if (!onlyactive || isActive(element.getMidpoint())) {
            size_t index = element.getIndex();
            if (index != RectangularMaskedMesh2D::Element::UNKNOWN_ELEMENT_INDEX) {
                double rin = element.getLower0(), rout = element.getUpper0();
                result += currents[index].c1 * (rout * rout - rin * rin);
            }
        }
    }
    return result * plask::PI * 0.01;  // kA/cm² µm² -->  mA
}

template <typename Geometry2DType> double ElectricalFem2DSolver<Geometry2DType>::getTotalCurrent(size_t nact) {
    if (!potentials) throw NoValue("Current");
    if (nact >= active.size()) throw BadInput(this->getId(), "Wrong active region number");
    const auto& act = active[nact];
    // Find the average of the active region
    size_t level = (act.bottom + act.top) / 2;
    return integrateCurrent(level, true);
}

template <typename Geometry2DType>
const LazyData<double> ElectricalFem2DSolver<Geometry2DType>::getVoltage(shared_ptr<const MeshD<2>> dest_mesh,
                                                                         InterpolationMethod method) const {
    if (!potentials) throw NoValue("Voltage");
    this->writelog(LOG_DEBUG, "Getting voltage");
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    if (this->maskedMesh->full())
        return interpolate(this->mesh, potentials, dest_mesh, method, this->geometry);
    else
        return interpolate(this->maskedMesh, potentials, dest_mesh, method, this->geometry);
}

template <typename Geometry2DType>
const LazyData<Vec<2>> ElectricalFem2DSolver<Geometry2DType>::getCurrentDensities(shared_ptr<const MeshD<2>> dest_mesh,
                                                                                  InterpolationMethod method) {
    if (!potentials) throw NoValue("Current density");
    this->writelog(LOG_DEBUG, "Getting current densities");
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    InterpolationFlags flags(this->geometry, InterpolationFlags::Symmetry::NP, InterpolationFlags::Symmetry::PN);
    if (this->maskedMesh->full()) {
        auto result = interpolate(this->mesh->getElementMesh(), currents, dest_mesh, method, flags);
        return LazyData<Vec<2>>(result.size(), [result, this, flags, dest_mesh](size_t i) {
            return this->geometry->getChildBoundingBox().contains(flags.wrap(dest_mesh->at(i))) ? result[i] : Vec<2>(0., 0.);
        });
    } else {
        auto result = interpolate(this->maskedMesh->getElementMesh(), currents, dest_mesh, method, flags);
        return LazyData<Vec<2>>(result.size(), [result](size_t i) {
            // Masked mesh always returns NaN outside of itself
            auto val = result[i];
            return isnan(val) ? Vec<2>(0., 0.) : val;
        });
    }
}

template <typename Geometry2DType>
const LazyData<double> ElectricalFem2DSolver<Geometry2DType>::getHeatDensities(shared_ptr<const MeshD<2>> dest_mesh,
                                                                               InterpolationMethod method) {
    if (!potentials) throw NoValue("Heat density");
    this->writelog(LOG_DEBUG, "Getting heat density");
    if (!heats) saveHeatDensities();  // we will compute heats only if they are needed
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    InterpolationFlags flags(this->geometry);
    if (this->maskedMesh->full()) {
        auto result = interpolate(this->mesh->getElementMesh(), heats, dest_mesh, method, flags);
        return LazyData<double>(result.size(), [result, this, flags, dest_mesh](size_t i) {
            return this->geometry->getChildBoundingBox().contains(flags.wrap(dest_mesh->at(i))) ? result[i] : 0.;
        });
    } else {
        auto result = interpolate(this->maskedMesh->getElementMesh(), heats, dest_mesh, method, flags);
        return LazyData<double>(result.size(), [result](size_t i) {
            // Masked mesh always returns NaN outside of itself
            auto val = result[i];
            return isnan(val) ? 0. : val;
        });
    }
}

template <typename Geometry2DType>
const LazyData<Tensor2<double>> ElectricalFem2DSolver<Geometry2DType>::getConductivity(shared_ptr<const MeshD<2>> dest_mesh,
                                                                                       InterpolationMethod) {
    this->initCalculation();
    this->writelog(LOG_DEBUG, "Getting conductivities");
    loadConductivities();
    InterpolationFlags flags(this->geometry);
    return interpolate(this->maskedMesh->getElementMesh(), conds, dest_mesh, INTERPOLATION_NEAREST, flags);
}

template <> double ElectricalFem2DSolver<Geometry2DCartesian>::getTotalEnergy() {
    double W = 0.;
    auto T = inTemperature(this->maskedMesh->getElementMesh());
    for (auto e : this->maskedMesh->elements()) {
        size_t ll = e.getLoLoIndex();
        size_t lu = e.getUpLoIndex();
        size_t ul = e.getLoUpIndex();
        size_t uu = e.getUpUpIndex();
        double dvx = 0.5e6 * (-potentials[ll] + potentials[lu] - potentials[ul] + potentials[uu]) /
                     (e.getUpper0() - e.getLower0());  // [grad(dV)] = V/m
        double dvy = 0.5e6 * (-potentials[ll] - potentials[lu] + potentials[ul] + potentials[uu]) /
                     (e.getUpper1() - e.getLower1());  // [grad(dV)] = V/m
        double w = this->geometry->getMaterial(e.getMidpoint())->eps(T[e.getIndex()]) * (dvx * dvx + dvy * dvy);
        double width = e.getUpper0() - e.getLower0();
        double height = e.getUpper1() - e.getLower1();
        W += width * height * w;
    }
    // TODO add outsides of comptational areas
    return geometry->getExtrusion()->getLength() * 0.5e-18 * phys::epsilon0 * W;  // 1e-18 µm³ -> m³
}

template <> double ElectricalFem2DSolver<Geometry2DCylindrical>::getTotalEnergy() {
    double W = 0.;
    auto T = inTemperature(this->maskedMesh->getElementMesh());
    for (auto e : this->maskedMesh->elements()) {
        size_t ll = e.getLoLoIndex();
        size_t lu = e.getUpLoIndex();
        size_t ul = e.getLoUpIndex();
        size_t uu = e.getUpUpIndex();
        auto midpoint = e.getMidpoint();
        double dvx = 0.5e6 * (-potentials[ll] + potentials[lu] - potentials[ul] + potentials[uu]) /
                     (e.getUpper0() - e.getLower0());  // [grad(dV)] = V/m
        double dvy = 0.5e6 * (-potentials[ll] - potentials[lu] + potentials[ul] + potentials[uu]) /
                     (e.getUpper1() - e.getLower1());  // [grad(dV)] = V/m
        double w = this->geometry->getMaterial(midpoint)->eps(T[e.getIndex()]) * (dvx * dvx + dvy * dvy);
        double width = e.getUpper0() - e.getLower0();
        double height = e.getUpper1() - e.getLower1();
        W += width * height * midpoint.rad_r() * w;
    }
    // TODO add outsides of computational area
    return 2. * plask::PI * 0.5e-18 * phys::epsilon0 * W;  // 1e-18 µm³ -> m³
}

template <typename Geometry2DType> double ElectricalFem2DSolver<Geometry2DType>::getCapacitance() {
    if (this->voltage_boundary.size() != 2) {
        throw BadInput(this->getId(), "Cannot estimate applied voltage (exactly 2 voltage boundary conditions required)");
    }

    double U = voltage_boundary[0].value - voltage_boundary[1].value;

    return 2e12 * getTotalEnergy() / (U * U);  // 1e12 F -> pF
}

template <> double ElectricalFem2DSolver<Geometry2DCartesian>::getTotalHeat() {
    double W = 0.;
    if (!heats) saveHeatDensities();  // we will compute heats only if they are needed
    for (auto e : this->maskedMesh->elements()) {
        double width = e.getUpper0() - e.getLower0();
        double height = e.getUpper1() - e.getLower1();
        W += width * height * heats[e.getIndex()];
    }
    return geometry->getExtrusion()->getLength() * 1e-15 * W;  // 1e-15 µm³ -> m³, W -> mW
}

template <> double ElectricalFem2DSolver<Geometry2DCylindrical>::getTotalHeat() {
    double W = 0.;
    if (!heats) saveHeatDensities();  // we will compute heats only if they are needed
    for (auto e : this->maskedMesh->elements()) {
        double width = e.getUpper0() - e.getLower0();
        double height = e.getUpper1() - e.getLower1();
        double r = e.getMidpoint().rad_r();
        W += width * height * r * heats[e.getIndex()];
    }
    return 2e-15 * plask::PI * W;  // 1e-15 µm³ -> m³, W -> mW
}

template struct PLASK_SOLVER_API ElectricalFem2DSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API ElectricalFem2DSolver<Geometry2DCylindrical>;

}}}  // namespace plask::electrical::shockley
