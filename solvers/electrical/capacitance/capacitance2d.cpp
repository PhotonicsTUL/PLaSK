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
#include "capacitance2d.hpp"

namespace plask { namespace electrical { namespace capacitance {

template <typename Geometry2DType>
Capacitance2DSolver<Geometry2DType>::Capacitance2DSolver(const std::string& name)
    : ComplexFemSolverWithMaskedMesh<Geometry2DType, RectangularMesh<2>>(name),
      outAcVoltage(this, &Capacitance2DSolver<Geometry2DType>::getVoltage),
      outAcCurrentDensity(this, &Capacitance2DSolver<Geometry2DType>::getCurrentDensities) {
    onInvalidate();
    inTemperature = 300.;
}

template <typename Geometry2DType>
void Capacitance2DSolver<Geometry2DType>::loadConfiguration(XMLReader& source, Manager& manager) {
    while (source.requireTagOrEnd()) parseConfiguration(source, manager);
}

template <typename Geometry2DType>
void Capacitance2DSolver<Geometry2DType>::parseConfiguration(XMLReader& source, Manager& manager) {
    std::string param = source.getNodeName();

    if (param == "ac-voltage") {
        this->readBoundaryConditions(manager, source, voltage_boundary);

    } else if (!this->parseFemConfiguration(source, manager)) {
        this->parseStandardConfiguration(source, manager);
    }
}

template <typename Geometry2DType> Capacitance2DSolver<Geometry2DType>::~Capacitance2DSolver() {}

template <typename Geometry2DType> void Capacitance2DSolver<Geometry2DType>::setupActiveRegions() {
    this->invalidate();

    if (!this->geometry || !this->mesh) return;

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
}

template <typename Geometry2DType> void Capacitance2DSolver<Geometry2DType>::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    setupActiveRegions();
    potentials.reset(this->maskedMesh->size(), 0.);
    currents.reset(this->maskedMesh->getElementsCount(), vec(0., 0.));
}

template <typename Geometry2DType> void Capacitance2DSolver<Geometry2DType>::onInvalidate() {
    potentials.reset();
    currents.reset();
}

template <>
inline void Capacitance2DSolver<Geometry2DCartesian>::setLocalMatrix(dcomplex&,
                                                                     dcomplex&,
                                                                     dcomplex&,
                                                                     dcomplex&,
                                                                     dcomplex&,
                                                                     dcomplex&,
                                                                     dcomplex&,
                                                                     dcomplex&,
                                                                     dcomplex&,
                                                                     dcomplex&,
                                                                     dcomplex,
                                                                     double,
                                                                     const Vec<2, double>&) {
    return;
}

template <>
inline void Capacitance2DSolver<Geometry2DCylindrical>::setLocalMatrix(dcomplex& k44,
                                                                       dcomplex& k33,
                                                                       dcomplex& k22,
                                                                       dcomplex& k11,
                                                                       dcomplex& k43,
                                                                       dcomplex& k21,
                                                                       dcomplex& k42,
                                                                       dcomplex& k31,
                                                                       dcomplex& k32,
                                                                       dcomplex& k41,
                                                                       dcomplex,
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
void Capacitance2DSolver<Geometry2DType>::setMatrix(
    FemMatrix<dcomplex>& A,
    DataVector<dcomplex>& B,
    const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary, dcomplex>& bvoltage,
    const LazyData<Tensor2<dcomplex>>& conds) {
    this->writelog(LOG_DETAIL, "Setting up matrix system ({})", A.describe());

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

        dcomplex kx = conds[i].c00;
        dcomplex ky = conds[i].c11;

        kx *= elemheight;
        kx /= elemwidth;
        ky *= elemwidth;
        ky /= elemheight;

        // set symmetric matrix components
        dcomplex k44, k33, k22, k11, k43, k21, k42, k31, k32, k41;

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
    dcomplex* aend = A.data + A.size;
    for (dcomplex* pa = A.data; pa != aend; ++pa) {
        if (isnan(*pa) || isinf(pa->real()) || isinf(pa->imag()))
            throw ComputationError(this->getId(), "error in stiffness matrix at position {0} ({1})", pa - A.data,
                                   isnan(*pa) ? "nan" : "inf");
    }
#endif
}

template <typename Geometry2DType> LazyData<Tensor2<dcomplex>> Capacitance2DSolver<Geometry2DType>::loadConductivities() {
    auto midmesh = this->maskedMesh->getElementMesh();
    auto temperature = inTemperature(midmesh);
    auto conductivity = inDifferentialConductivity(midmesh);

    DataVector<Tensor2<dcomplex>> conds;  ///< Cached complex conductivities

    for (auto e : this->maskedMesh->elements()) {
        size_t i = e.getIndex();
        Vec<2, double> midpoint = e.getMidpoint();
        double imag =
            2e6 * M_PI * phys::epsilon0 * frequency * this->geometry->getMaterial(midpoint)->eps(temperature[i]);  // MHz -> Hz
        conds[i] = Tensor2<dcomplex>(dcomplex(conductivity[i].c00, imag), dcomplex(conductivity[i].c11, imag));
    }

    return conds;
}

template <typename Geometry2DType> void Capacitance2DSolver<Geometry2DType>::compute() {
    this->initCalculation();

    // Store boundary conditions for current mesh
    auto vconst = voltage_boundary(this->maskedMesh, this->geometry);

    this->writelog(LOG_INFO, "Running AC calculations for {} MHz", frequency);

    std::unique_ptr<FemMatrix<dcomplex>> pA(this->getMatrix());
    FemMatrix<dcomplex>& A = *pA.get();

#ifndef NDEBUG
    if (!potentials.unique()) this->writelog(LOG_DEBUG, "Voltage data held by something else...");
#endif
    potentials = potentials.claim();
    auto conds = loadConductivities();

    DataVector<dcomplex> rhs(potentials.size());

    setMatrix(A, rhs, vconst, conds);
    A.solve(rhs, potentials);

    outAcVoltage.fireChanged();
    outAcCurrentDensity.fireChanged();
}

template <> dcomplex Capacitance2DSolver<Geometry2DCartesian>::integrateCurrent(size_t vindex, bool onlyactive) {
    if (!potentials) throw NoValue("current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    dcomplex result = 0.;
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

template <> dcomplex Capacitance2DSolver<Geometry2DCylindrical>::integrateCurrent(size_t vindex, bool onlyactive) {
    if (!potentials) throw NoValue("current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    dcomplex result = 0.;
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

template <typename Geometry2DType> dcomplex Capacitance2DSolver<Geometry2DType>::getActiveCurrent(size_t nact) {
    if (!potentials) throw NoValue("current");
    if (nact >= active.size()) throw BadInput(this->getId(), "wrong active region number");
    const auto& act = active[nact];
    // Find the average of the active region
    size_t level = (act.bottom + act.top) / 2;
    return integrateCurrent(level, true);
}

template <typename Geometry2DType>
const LazyData<dcomplex> Capacitance2DSolver<Geometry2DType>::getVoltage(shared_ptr<const MeshD<2>> dest_mesh,
                                                                         InterpolationMethod method) const {
    if (!potentials) throw NoValue("voltage");
    this->writelog(LOG_DEBUG, "Getting voltage");
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    if (this->maskedMesh->full())
        return interpolate(this->mesh, potentials, dest_mesh, method, this->geometry);
    else
        return interpolate(this->maskedMesh, potentials, dest_mesh, method, this->geometry);
}

template <typename Geometry2DType>
const LazyData<Vec<2, dcomplex>> Capacitance2DSolver<Geometry2DType>::getCurrentDensities(shared_ptr<const MeshD<2>> dest_mesh,
                                                                                          InterpolationMethod method) {
    if (!potentials) throw NoValue("current density");
    this->writelog(LOG_DEBUG, "Getting current densities");
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    InterpolationFlags flags(this->geometry, InterpolationFlags::Symmetry::NP, InterpolationFlags::Symmetry::PN);
    if (this->maskedMesh->full()) {
        auto result = interpolate(this->mesh->getElementMesh(), currents, dest_mesh, method, flags);
        return LazyData<Vec<2, dcomplex>>(result.size(), [result, this, flags, dest_mesh](size_t i) {
            return this->geometry->getChildBoundingBox().contains(flags.wrap(dest_mesh->at(i))) ? result[i]
                                                                                                : Vec<2, dcomplex>(0., 0.);
        });
    } else {
        auto result = interpolate(this->maskedMesh->getElementMesh(), currents, dest_mesh, method, flags);
        return LazyData<Vec<2, dcomplex>>(result.size(), [result](size_t i) {
            // Masked mesh always returns NaN outside of itself
            auto val = result[i];
            return isnan(val) ? Vec<2, dcomplex>(0., 0.) : val;
        });
    }
    assert(false);  // should not be reached
}

template struct PLASK_SOLVER_API Capacitance2DSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API Capacitance2DSolver<Geometry2DCylindrical>;

}}}  // namespace plask::electrical::capacitance
