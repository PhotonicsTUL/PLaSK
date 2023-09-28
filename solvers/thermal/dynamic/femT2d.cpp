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
#include "femT2d.hpp"

namespace plask { namespace thermal { namespace dynamic {

const double BIG = 1e16;

template<typename Geometry2DType>
DynamicThermalFem2DSolver<Geometry2DType>::DynamicThermalFem2DSolver(const std::string& name) :
    FemSolverWithMaskedMesh<Geometry2DType, RectangularMesh<2>>(name),
    outTemperature(this, &DynamicThermalFem2DSolver<Geometry2DType>::getTemperatures),
    outHeatFlux(this, &DynamicThermalFem2DSolver<Geometry2DType>::getHeatFluxes),
    outThermalConductivity(this, &DynamicThermalFem2DSolver<Geometry2DType>::getThermalConductivity),
    inittemp(300.),
    methodparam(0.5),
    timestep(0.1),
    elapstime(0.),
    lumping(true),
    rebuildfreq(0),
    logfreq(500)
{
    temperatures.reset();
    fluxes.reset();
    inHeat = 0.;
}


template<typename Geometry2DType>
DynamicThermalFem2DSolver<Geometry2DType>::~DynamicThermalFem2DSolver() {
}


template<typename Geometry2DType>
void DynamicThermalFem2DSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "temperature")
            this->readBoundaryConditions(manager, source, temperature_boundary);

        else if (param == "loop") {
            inittemp = source.getAttribute<double>("inittemp", inittemp);
            timestep = source.getAttribute<double>("timestep", timestep);
            rebuildfreq = source.getAttribute<size_t>("rebuildfreq", rebuildfreq);
            logfreq = source.getAttribute<size_t>("logfreq", logfreq);
            source.requireTagEnd();
        }

        else if (source.getNodeName() == "matrix") {
            methodparam = source.getAttribute<double>("methodparam", methodparam);
            lumping = source.getAttribute<bool>("lumping", lumping);
            this->parseFemConfiguration(source, manager);
        } else {
            this->parseStandardConfiguration(source, manager);
        }
    }
}


template<typename Geometry2DType>
void DynamicThermalFem2DSolver<Geometry2DType>::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    elapstime = 0.;

    FemSolverWithMaskedMesh<Geometry2DType, RectangularMesh<2>>::onInitialize();

    temperatures.reset(this->maskedMesh->size(), inittemp);

    thickness.reset(this->maskedMesh->getElementsCount(), NAN);
    // Set stiffness matrix and load vector
    for (auto elem: this->maskedMesh->elements())
    {
        if (!isnan(thickness[elem.getIndex()])) continue;
        auto material = this->geometry->getMaterial(elem.getMidpoint());
        double top = elem.getUpper1(), bottom = elem.getLower1();
        size_t row = elem.getIndex1();
        size_t itop = row+1, ibottom = row;
        size_t c = elem.getIndex0();
        for (size_t r = row; r > 0; r--) {
            auto e = this->mesh->element(c, r-1);
            auto m = this->geometry->getMaterial(e.getMidpoint());
            if (m == material) {                            //TODO ignore doping
                bottom = e.getLower1();
                ibottom = r-1;
            }
            else break;
        }
        for (size_t r = elem.getIndex1()+1; r < this->mesh->axis[1]->size()-1; r++) {
            auto e = this->mesh->element(c, r);
            auto m = this->geometry->getMaterial(e.getMidpoint());
            if (m == material) {                            //TODO ignore doping
                top = e.getUpper1();
                itop = r+1;
            }
            else break;
        }
        double h = top - bottom;
        for (size_t r = ibottom; r != itop; ++r) {
            size_t idx = this->maskedMesh->element(c, r).getIndex();
            if (idx != RectangularMaskedMesh2D::Element::UNKNOWN_ELEMENT_INDEX)
                thickness[idx] = h;
        }
    }
}


template<typename Geometry2DType> void DynamicThermalFem2DSolver<Geometry2DType>::onInvalidate() {
    temperatures.reset();
    fluxes.reset();
}


template<>
void DynamicThermalFem2DSolver<Geometry2DCartesian>::setMatrix(
        FemMatrix& A, FemMatrix& B, DataVector<double>& F,
        const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& btemperature)
{
    this->writelog(LOG_DETAIL, "Setting up matrix system ({})", A.describe());

    auto heatdensities = inHeat(this->maskedMesh->getElementMesh());

    // zero the matrices A, B and the load vector F
    A.clear();
    B.clear();
    F.fill(0.);

    // Set stiffness matrix and load vector
    for (auto elem: this->maskedMesh->elements())
    {
        // nodes numbers for the current element
        size_t loleftno = elem.getLoLoIndex();
        size_t lorghtno = elem.getUpLoIndex();
        size_t upleftno = elem.getLoUpIndex();
        size_t uprghtno = elem.getUpUpIndex();

        // element size
        double elemwidth = elem.getUpper0() - elem.getLower0();
        double elemheight = elem.getUpper1() - elem.getLower1();

        // point and material in the middle of the element
        Vec<2,double> midpoint = elem.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        // average temperature on the element
        double temp = 0.25 * (temperatures[loleftno] + temperatures[lorghtno] + temperatures[upleftno] + temperatures[uprghtno]);

        // thermal conductivity
        double kx, ky;
        std::tie(kx,ky) = std::tuple<double,double>(material->thermk(temp, thickness[elem.getIndex()]));

        // element of heat capacity matrix
        double c = material->cp(temp) * material->dens(temp) * 0.25 * 1E-12 * elemheight * elemwidth / timestep / 1E-9;

        kx *= elemheight; kx /= elemwidth;
        ky *= elemwidth; ky /= elemheight;

        // load vector: heat densities
        double f = 0.25e-12 * elemwidth * elemheight * heatdensities[elem.getIndex()]; // 1e-12 -> to transform µm² into m²

        // set symmetric matrix components in thermal conductivity matrix
        double k44, k33, k22, k11, k43, k21, k42, k31, k32, k41;

        k44 = k33 = k22 = k11 = (kx + ky) / 3.;
        k43 = k21 = (-2. * kx + ky) / 6.;
        k42 = k31 = - (kx + ky) / 6.;
        k32 = k41 = (kx - 2. * ky) / 6.;

        //Wheter lumping the mass matrces A, B?
        if (lumping)
        {
            A(loleftno, loleftno) += methodparam*k11 + c;
            A(lorghtno, lorghtno) += methodparam*k22 + c;
            A(uprghtno, uprghtno) += methodparam*k33 + c;
            A(upleftno, upleftno) += methodparam*k44 + c;

            A(lorghtno, loleftno) += methodparam*k21;
            A(uprghtno, loleftno) += methodparam*k31;
            A(upleftno, loleftno) += methodparam*k41;
            A(uprghtno, lorghtno) += methodparam*k32;
            A(upleftno, lorghtno) += methodparam*k42;
            A(upleftno, uprghtno) += methodparam*k43;

            B(loleftno, loleftno) += -(1-methodparam)*k11 + c;
            B(lorghtno, lorghtno) += -(1-methodparam)*k22 + c;
            B(uprghtno, uprghtno) += -(1-methodparam)*k33 + c;
            B(upleftno, upleftno) += -(1-methodparam)*k44 + c;

            B(lorghtno, loleftno) += -(1-methodparam)*k21;
            B(uprghtno, loleftno) += -(1-methodparam)*k31;
            B(upleftno, loleftno) += -(1-methodparam)*k41;
            B(uprghtno, lorghtno) += -(1-methodparam)*k32;
            B(upleftno, lorghtno) += -(1-methodparam)*k42;
            B(upleftno, uprghtno) += -(1-methodparam)*k43;
        }
        else
        {
            A(loleftno, loleftno) += methodparam*k11 + 4./9.*c;
            A(lorghtno, lorghtno) += methodparam*k22 + 4./9.*c;
            A(uprghtno, uprghtno) += methodparam*k33 + 4./9.*c;
            A(upleftno, upleftno) += methodparam*k44 + 4./9.*c;

            A(lorghtno, loleftno) += methodparam*k21 + 2./9.*c;
            A(uprghtno, loleftno) += methodparam*k31 + 1./9.*c;
            A(upleftno, loleftno) += methodparam*k41 + 2./9.*c;
            A(uprghtno, lorghtno) += methodparam*k32 + 2./9.*c;
            A(upleftno, lorghtno) += methodparam*k42 + 1./9.*c;
            A(upleftno, uprghtno) += methodparam*k43 + 2./9.*c;

            B(loleftno, loleftno) += -(1-methodparam)*k11 + 4./9.*c;
            B(lorghtno, lorghtno) += -(1-methodparam)*k22 + 4./9.*c;
            B(uprghtno, uprghtno) += -(1-methodparam)*k33 + 4./9.*c;
            B(upleftno, upleftno) += -(1-methodparam)*k44 + 4./9.*c;

            B(lorghtno, loleftno) += -(1-methodparam)*k21 + 2./9.*c;
            B(uprghtno, loleftno) += -(1-methodparam)*k31 + 1./9.*c;
            B(upleftno, loleftno) += -(1-methodparam)*k41 + 2./9.*c;
            B(uprghtno, lorghtno) += -(1-methodparam)*k32 + 2./9.*c;
            B(upleftno, lorghtno) += -(1-methodparam)*k42 + 1./9.*c;
            B(upleftno, uprghtno) += -(1-methodparam)*k43 + 2./9.*c;
        }
        // set load vector
        F[loleftno] += f;
        F[lorghtno] += f;
        F[uprghtno] += f;
        F[upleftno] += f;
    }

    A.applyBC(btemperature, F);

    // macierz A -> L L^T
    A.factorize();

#ifndef NDEBUG
    double* aend = A.data + A.size;
    for (double* pa = A.data; pa != aend; ++pa) {
        if (isnan(*pa) || isinf(*pa))
            throw ComputationError(this->getId(), "Error in stiffness matrix at position {0}", pa-A.data);
    }
#endif

}

template<>
void DynamicThermalFem2DSolver<Geometry2DCylindrical>::setMatrix(
        FemMatrix& A, FemMatrix& B, DataVector<double>& F,
        const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& btemperature)
{
    this->writelog(LOG_DETAIL, "Setting up matrix system ({})", A.describe());

    auto heatdensities = inHeat(this->maskedMesh->getElementMesh());

    // zero the matrices A, B and the load vector F
    std::fill_n(A.data, A.size, 0.);
    std::fill_n(B.data, B.size, 0.);
    F.fill(0.);

    // Set stiffness matrix and load vector
    for (auto elem: this->maskedMesh->elements())
    {
        // nodes numbers for the current element
        size_t loleftno = elem.getLoLoIndex();
        size_t lorghtno = elem.getUpLoIndex();
        size_t upleftno = elem.getLoUpIndex();
        size_t uprghtno = elem.getUpUpIndex();

        // element size
        double elemwidth = elem.getUpper0() - elem.getLower0();
        double elemheight = elem.getUpper1() - elem.getLower1();

        // point and material in the middle of the element
        Vec<2,double> midpoint = elem.getMidpoint();
        double r = midpoint.rad_r();
        auto material = this->geometry->getMaterial(midpoint);

        // average temperature on the element
        double temp = 0.25 * (temperatures[loleftno] + temperatures[lorghtno] + temperatures[upleftno] + temperatures[uprghtno]);

        // thermal conductivity
        double kx, ky;
        std::tie(kx,ky) = std::tuple<double,double>(material->thermk(temp, thickness[elem.getIndex()]));

        // element of heat capacity matrix
        double c = material->cp(temp) * material->dens(temp) * 0.25 * 1E-12 * r * elemheight * elemwidth / timestep / 1E-9;

        kx *= elemheight; kx /= elemwidth; kx *= r;
        ky *= elemwidth; ky /= elemheight; ky *= r;

        // load vector: heat densities
        double f = 0.25e-12 * r * elemwidth * elemheight * heatdensities[elem.getIndex()]; // 1e-12 -> to transform µm² into m²

        // set symmetric matrix components in thermal conductivity matrix
        double k44, k33, k22, k11, k43, k21, k42, k31, k32, k41;

        k44 = k33 = k22 = k11 = (kx + ky) / 3.;
        k43 = k21 = (-2. * kx + ky) / 6.;
        k42 = k31 = - (kx + ky) / 6.;
        k32 = k41 = (kx - 2. * ky) / 6.;

        //Wheter lumping the mass matrces A, B?
        if (lumping)
        {
            A(loleftno, loleftno) += methodparam*k11 + c;
            A(lorghtno, lorghtno) += methodparam*k22 + c;
            A(uprghtno, uprghtno) += methodparam*k33 + c;
            A(upleftno, upleftno) += methodparam*k44 + c;

            A(lorghtno, loleftno) += methodparam*k21;
            A(uprghtno, loleftno) += methodparam*k31;
            A(upleftno, loleftno) += methodparam*k41;
            A(uprghtno, lorghtno) += methodparam*k32;
            A(upleftno, lorghtno) += methodparam*k42;
            A(upleftno, uprghtno) += methodparam*k43;

            B(loleftno, loleftno) += -(1-methodparam)*k11 + c;
            B(lorghtno, lorghtno) += -(1-methodparam)*k22 + c;
            B(uprghtno, uprghtno) += -(1-methodparam)*k33 + c;
            B(upleftno, upleftno) += -(1-methodparam)*k44 + c;

            B(lorghtno, loleftno) += -(1-methodparam)*k21;
            B(uprghtno, loleftno) += -(1-methodparam)*k31;
            B(upleftno, loleftno) += -(1-methodparam)*k41;
            B(uprghtno, lorghtno) += -(1-methodparam)*k32;
            B(upleftno, lorghtno) += -(1-methodparam)*k42;
            B(upleftno, uprghtno) += -(1-methodparam)*k43;
        }
        else
        {
            A(loleftno, loleftno) += methodparam*k11 + 4./9.*c;
            A(lorghtno, lorghtno) += methodparam*k22 + 4./9.*c;
            A(uprghtno, uprghtno) += methodparam*k33 + 4./9.*c;
            A(upleftno, upleftno) += methodparam*k44 + 4./9.*c;

            A(lorghtno, loleftno) += methodparam*k21 + 2./9.*c;
            A(uprghtno, loleftno) += methodparam*k31 + 1./9.*c;
            A(upleftno, loleftno) += methodparam*k41 + 2./9.*c;
            A(uprghtno, lorghtno) += methodparam*k32 + 2./9.*c;
            A(upleftno, lorghtno) += methodparam*k42 + 1./9.*c;
            A(upleftno, uprghtno) += methodparam*k43 + 2./9.*c;

            B(loleftno, loleftno) += -(1-methodparam)*k11 + 4./9.*c;
            B(lorghtno, lorghtno) += -(1-methodparam)*k22 + 4./9.*c;
            B(uprghtno, uprghtno) += -(1-methodparam)*k33 + 4./9.*c;
            B(upleftno, upleftno) += -(1-methodparam)*k44 + 4./9.*c;

            B(lorghtno, loleftno) += -(1-methodparam)*k21 + 2./9.*c;
            B(uprghtno, loleftno) += -(1-methodparam)*k31 + 1./9.*c;
            B(upleftno, loleftno) += -(1-methodparam)*k41 + 2./9.*c;
            B(uprghtno, lorghtno) += -(1-methodparam)*k32 + 2./9.*c;
            B(upleftno, lorghtno) += -(1-methodparam)*k42 + 1./9.*c;
            B(upleftno, uprghtno) += -(1-methodparam)*k43 + 2./9.*c;
        }
        // set load vector
        F[loleftno] += f;
        F[lorghtno] += f;
        F[uprghtno] += f;
        F[upleftno] += f;
    }

    A.applyBC(btemperature, F);

    // macierz A -> L L^T
    A.factorize();

#ifndef NDEBUG
    double* aend = A.data + A.size;
    for (double* pa = A.data; pa != aend; ++pa) {
        if (isnan(*pa) || isinf(*pa))
            throw ComputationError(this->getId(), "Error in stiffness matrix at position {0}", pa-A.data);
    }
#endif

}


template<typename Geometry2DType>
double DynamicThermalFem2DSolver<Geometry2DType>::compute(double time)
{
    this->initCalculation();

    fluxes.reset();

    // store boundary conditions for current mesh
    auto btemperature = temperature_boundary(this->maskedMesh, this->geometry);

    size_t size = this->maskedMesh->size();

    std::unique_ptr<FemMatrix> pA(this->getMatrix());
    FemMatrix& A = *pA.get();
    std::unique_ptr<FemMatrix> pB(this->getMatrix());
    FemMatrix& B = *pB.get();

    this->writelog(LOG_INFO, "Running thermal calculations");
    maxT = *std::max_element(temperatures.begin(), temperatures.end());

#   ifndef NDEBUG
        if (!temperatures.unique()) this->writelog(LOG_DEBUG, "Temperature data held by something else...");
#   endif
    temperatures = temperatures.claim();
    DataVector<double> F(size), X(size);

    setMatrix(A, B, F, btemperature);

    size_t r = rebuildfreq,
           l = logfreq;

    time += timestep/2.;
    for (double t = 0.; t < time; t += timestep) {

        if (rebuildfreq && r == 0)
        {
            setMatrix(A, B, F, btemperature);
            r = rebuildfreq;
        }

        B.mult(temperatures, X);
        for (std::size_t i = 0; i < X.size(); ++i) X[i] += F[i];

        DataVector<double> T(X);

        A.solverhs(T, temperatures);

        if (logfreq && l == 0)
        {
            maxT = *std::max_element(temperatures.begin(), temperatures.end());
            this->writelog(LOG_RESULT, "Time {:.2f} ns: max(T) = {:.3f} K", elapstime, maxT);
            l = logfreq;
        }

        r--;
        l--;
        elapstime += timestep;
    }

    elapstime -= timestep;
    outTemperature.fireChanged();
    outHeatFlux.fireChanged();

    return 0.;
}

template<typename Geometry2DType>
void DynamicThermalFem2DSolver<Geometry2DType>::saveHeatFluxes()
{
    this->writelog(LOG_DETAIL, "Computing heat fluxes");

    fluxes.reset(this->maskedMesh->getElementsCount());

    for (auto e: this->maskedMesh->elements())
    {
        Vec<2,double> midpoint = e.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        size_t loleftno = e.getLoLoIndex();
        size_t lorghtno = e.getUpLoIndex();
        size_t upleftno = e.getLoUpIndex();
        size_t uprghtno = e.getUpUpIndex();

        double temp = 0.25 * (temperatures[loleftno] + temperatures[lorghtno] +
                               temperatures[upleftno] + temperatures[uprghtno]);

        double kx, ky;
        auto leaf = dynamic_pointer_cast<const GeometryObjectD<2>>(
                        this->geometry->getMatchingAt(midpoint, &GeometryObject::PredicateIsLeaf)
                     );
        if (leaf)
            std::tie(kx,ky) = std::tuple<double,double>(material->thermk(temp, leaf->getBoundingBox().height()));
        else
            std::tie(kx,ky) = std::tuple<double,double>(material->thermk(temp));


        fluxes[e.getIndex()] = vec(
            - 0.5e6 * kx * (- temperatures[loleftno] + temperatures[lorghtno]
                             - temperatures[upleftno] + temperatures[uprghtno]) / (e.getUpper0() - e.getLower0()), // 1e6 - from um to m
            - 0.5e6 * ky * (- temperatures[loleftno] - temperatures[lorghtno]
                             + temperatures[upleftno] + temperatures[uprghtno]) / (e.getUpper1() - e.getLower1())); // 1e6 - from um to m
    }
}


template<typename Geometry2DType>
const LazyData<double> DynamicThermalFem2DSolver<Geometry2DType>::getTemperatures(const shared_ptr<const MeshD<2>>& dest_mesh, InterpolationMethod method) const {
    this->writelog(LOG_DEBUG, "Getting temperatures");
    if (!temperatures) return LazyData<double>(dest_mesh->size(), inittemp); // in case the receiver is connected and no temperature calculated yet
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    if (this->maskedMesh->full())
        return SafeData<double>(interpolate(this->mesh, temperatures, dest_mesh, method, this->geometry), 300.);
    else
        return SafeData<double>(interpolate(this->maskedMesh, temperatures, dest_mesh, method, this->geometry), 300.);
}


template<typename Geometry2DType>
const LazyData<Vec<2>> DynamicThermalFem2DSolver<Geometry2DType>::getHeatFluxes(const shared_ptr<const MeshD<2>>& dest_mesh, InterpolationMethod method) {
    this->writelog(LOG_DEBUG, "Getting heat fluxes");
    if (!temperatures) return LazyData<Vec<2>>(dest_mesh->size(), Vec<2>(0.,0.)); // in case the receiver is connected and no fluxes calculated yet
    if (!fluxes) saveHeatFluxes(); // we will compute fluxes only if they are needed
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    if (this->maskedMesh->full())
        return SafeData<Vec<2>>(interpolate(this->mesh->getElementMesh(), fluxes, dest_mesh, method,
                                            InterpolationFlags(this->geometry, InterpolationFlags::Symmetry::NP, InterpolationFlags::Symmetry::PN)),
                                Zero<Vec<2>>());
    else
        return SafeData<Vec<2>>(interpolate(this->maskedMesh->getElementMesh(), fluxes, dest_mesh, method,
                                            InterpolationFlags(this->geometry, InterpolationFlags::Symmetry::NP, InterpolationFlags::Symmetry::PN)),
                                Zero<Vec<2>>());
}


template<typename Geometry2DType> DynamicThermalFem2DSolver<Geometry2DType>::
ThermalConductivityData::ThermalConductivityData(const DynamicThermalFem2DSolver<Geometry2DType>* solver, const shared_ptr<const MeshD<2>>& dst_mesh):
    solver(solver), dest_mesh(dst_mesh), flags(solver->geometry)
{
    if (solver->temperatures) temps = interpolate(solver->maskedMesh, solver->temperatures, solver->maskedMesh->getElementMesh(), INTERPOLATION_LINEAR);
    else temps = LazyData<double>(solver->maskedMesh->getElementsCount(), solver->inittemp);
}

template<typename Geometry2DType> Tensor2<double> DynamicThermalFem2DSolver<Geometry2DType>::
ThermalConductivityData::at(std::size_t i) const {
    auto point = flags.wrap(dest_mesh->at(i));
    size_t x = solver->mesh->axis[0]->findUpIndex(point[0]),
           y = solver->mesh->axis[1]->findUpIndex(point[1]);
    if (x == 0 || y == 0 || x == solver->mesh->axis[0]->size() || y == solver->mesh->axis[1]->size())
        return Tensor2<double>(NAN);
    else {
        auto elem = solver->maskedMesh->element(x-1, y-1);
        auto material = solver->geometry->getMaterial(elem.getMidpoint());
        size_t idx = elem.getIndex();
        if (idx == RectangularMaskedMesh2D::Element::UNKNOWN_ELEMENT_INDEX) return Tensor2<double>(NAN);
        return material->thermk(temps[idx], solver->thickness[idx]);
    }
}

template<typename Geometry2DType> std::size_t DynamicThermalFem2DSolver<Geometry2DType>::
ThermalConductivityData::size() const { return dest_mesh->size(); }

template<typename Geometry2DType>
const LazyData<Tensor2<double>> DynamicThermalFem2DSolver<Geometry2DType>::getThermalConductivity(const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod) {
    this->writelog(LOG_DEBUG, "Getting thermal conductivities");
    this->initCalculation();
    return LazyData<Tensor2<double>>(new
        DynamicThermalFem2DSolver<Geometry2DType>::ThermalConductivityData(this, dst_mesh)
    );
}


template<> std::string DynamicThermalFem2DSolver<Geometry2DCartesian>::getClassName() const { return "thermal.Dynamic2D"; }
template<> std::string DynamicThermalFem2DSolver<Geometry2DCylindrical>::getClassName() const { return "thermal.DynamicCyl"; }

template struct PLASK_SOLVER_API DynamicThermalFem2DSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API DynamicThermalFem2DSolver<Geometry2DCylindrical>;

}}} // namespace plask::thermal::thermal
