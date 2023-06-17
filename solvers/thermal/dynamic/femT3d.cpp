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

#include "femT3d.hpp"

namespace plask { namespace thermal { namespace dynamic {

const double BIG = 1e16;

DynamicThermalFem3DSolver::DynamicThermalFem3DSolver(const std::string& name) :
    FemSolverWithMaskedMesh<Geometry3D, RectangularMesh<3>>(name),
    outTemperature(this, &DynamicThermalFem3DSolver::getTemperatures),
    outHeatFlux(this, &DynamicThermalFem3DSolver::getHeatFluxes),
    outThermalConductivity(this, &DynamicThermalFem3DSolver::getThermalConductivity),
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


DynamicThermalFem3DSolver::~DynamicThermalFem3DSolver() {
}


void DynamicThermalFem3DSolver::loadConfiguration(XMLReader &source, Manager &manager)
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

        else if (!this->parseFemConfiguration(source, manager)) {
            this->parseStandardConfiguration(source, manager);
        }
    }
}


void DynamicThermalFem3DSolver::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    elapstime = 0.;

    FemSolverWithMaskedMesh<Geometry3D, RectangularMesh<3>>::onInitialize();

    temperatures.reset(this->maskedMesh->size(), inittemp);

    thickness.reset(this->maskedMesh->getElementsCount(), NAN);
    // Set stiffness matrix and load vector
    for (auto elem: this->maskedMesh->elements())
    {
        if (!isnan(thickness[elem.getIndex()])) continue;
        auto material = this->geometry->getMaterial(elem.getMidpoint());
        double top = elem.getUpper2(), bottom = elem.getLower2();
        size_t row = elem.getIndex2();
        size_t itop = row+1, ibottom = row;
        for (size_t r = row; r > 0; r--) {
            auto e = this->mesh->element(elem.getIndex0(), elem.getIndex1(), r-1);
            auto m = this->geometry->getMaterial(e.getMidpoint());
            if (m == material) {                            //TODO ignore doping
                bottom = e.getLower2();
                ibottom = r-1;
            }
            else break;
        }
        for (size_t r = elem.getIndex2()+1; r < this->mesh->axis[2]->size()-1; r++) {
            auto e = this->mesh->element(elem.getIndex0(), elem.getIndex1(), r);
            auto m = this->geometry->getMaterial(e.getMidpoint());
            if (m == material) {                            //TODO ignore doping
                top = e.getUpper2();
                itop = r+1;
            }
            else break;
        }
        double h = top - bottom;
        for (size_t r = ibottom; r != itop; ++r) {
            size_t idx = this->maskedMesh->element(elem.getIndex0(), elem.getIndex1(), r).getIndex();
            if (idx != RectangularMaskedMesh3D::Element::UNKNOWN_ELEMENT_INDEX)
                thickness[idx] = h;
        }
    }
}


void DynamicThermalFem3DSolver::onInvalidate() {
    temperatures.reset();
    fluxes.reset();
    thickness.reset();
}


void DynamicThermalFem3DSolver::setMatrix(FemMatrix& A, FemMatrix& B, DataVector<double>& F,
        const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary,double>& btemperature)
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size={0}, bands={1}({2}))", A.size, A.kd+1, A.ld+1);

    auto heats = inHeat(maskedMesh->getElementMesh()/*, INTERPOLATION_NEAREST*/);

    // zero the matrices A, B and the load vector F
    A.clear();
    B.clear();
    F.fill(0.);

    // Set stiffness matrix and load vector
    for (auto elem: this->maskedMesh->elements())
    {
        // nodes numbers for the current element
        size_t idx[8];
        idx[0] = elem.getLoLoLoIndex();          //   z y            6-----7
        idx[1] = elem.getUpLoLoIndex();          //   |/__x         /|    /|
        idx[2] = elem.getLoUpLoIndex();          //                4-----5 |
        idx[3] = elem.getUpUpLoIndex();          //                | 2---|-3
        idx[4] = elem.getLoLoUpIndex();          //                |/    |/
        idx[5] = elem.getUpLoUpIndex();          //                0-----1
        idx[6] = elem.getLoUpUpIndex();          //
        idx[7] = elem.getUpUpUpIndex();          //

        // element size
        double dx = elem.getUpper0() - elem.getLower0();
        double dy = elem.getUpper1() - elem.getLower1();
        double dz = elem.getUpper2() - elem.getLower2();

        // point and material in the middle of the element
        Vec<3> middle = elem.getMidpoint();
        auto material = geometry->getMaterial(middle);

        // average temperature on the element
        double temp = 0.; for (int i = 0; i < 8; ++i) temp += temperatures[idx[i]]; temp *= 0.125;

        // thermal conductivity
        double kx, ky, kz;
        std::tie(ky,kz) = std::tuple<double,double>(material->thermk(temp, thickness[elem.getIndex()]));

        ky *= 1e-6; kz *= 1e-6;                                         // W/m -> W/µm
        kx = ky;

        kx /= dx; kx *= dy; kx *= dz;
        ky *= dx; ky /= dy; ky *= dz;
        kz *= dx; kz *= dy; kz /= dz;

        // element of heat capacity matrix
        double c = material->cp(temp) * material->dens(temp) * 0.125e-9 * dx * dy * dz / timestep;  //0.125e-9 = 0.5*0.5*0.5*1e-18/1E-9

        // load vector: heat densities
        double f = 0.125e-18 * dx * dy * dz * heats[elem.getIndex()];   // 1e-18 -> to transform µm³ into m³

        // set components of symmetric matrix K
        double K[8][8];
        K[0][0] = K[1][1] = K[2][2] = K[3][3] = K[4][4] = K[5][5] = K[6][6] = K[7][7] = (kx + ky + kz) / 9.;

        K[1][0] = K[3][2] = K[5][4] = K[7][6] = (-2.*kx +    ky +    kz) / 18.;
        K[2][0] = K[3][1] = K[6][4] = K[7][5] = (    kx - 2.*ky +    kz) / 18.;
        K[4][0] = K[5][1] = K[6][2] = K[7][3] = (    kx +    ky - 2.*kz) / 18.;

        K[4][2] = K[5][3] = K[6][0] = K[7][1] = (    kx - 2.*ky - 2.*kz) / 36.;
        K[4][1] = K[5][0] = K[6][3] = K[7][2] = (-2.*kx +    ky - 2.*kz) / 36.;
        K[2][1] = K[3][0] = K[6][5] = K[7][4] = (-2.*kx - 2.*ky +    kz) / 36.;

        K[4][3] = K[5][2] = K[6][1] = K[7][0] = -(kx + ky + kz) / 36.;

        // updating A, B matrices with K elements and F load vector
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j <= i; ++j) {
                A(idx[i],idx[j]) += methodparam*K[i][j];
                B(idx[i],idx[j]) += -(1-methodparam)*K[i][j];
            }
            F[idx[i]] += f;
        }

        // updating A, B matrices with C elements
        // Wheter lumping the mass matrces A, B?
        if (lumping)
        {
            for (int i = 0; i < 8; ++i) {
                A(idx[i],idx[i]) += c;
                B(idx[i],idx[i]) += c;
            }
        }
        else
        {
            // set components of symmetric matrix K
            double C[8][8];
            C[0][0] = C[1][1] = C[2][2] = C[3][3] = C[4][4] = C[5][5] = C[6][6] = C[7][7] = c * 8 / 27.;
            C[1][0] = C[3][0] = C[4][0] = C[2][1] = C[5][1] = C[3][2] = C[6][2] = C[7][3] = C[5][4] = C[7][4] = C[6][5] = C[7][6] = c * 4 / 27.;
            C[2][0] = C[5][0] = C[7][0] = C[3][1] = C[4][1] = C[6][1] = C[5][2] = C[7][2] = C[4][3] = C[6][3] = C[6][4] = C[7][5] = c * 2 / 27.;
            C[6][0] = C[7][1] = C[4][2] = C[5][3] = c / 27.;

            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j <= i; ++j) {
                    A(idx[i],idx[j]) += C[i][j];
                    B(idx[i],idx[j]) += C[i][j];
                }
            }
        }

    }

    //boundary conditions of the first kind
    A.applyBC(btemperature, F);

    // macierz A -> L L^T
    A.factorize();

#ifndef NDEBUG
    double* aend = A.data + A.size * A.kd;
    for (double* pa = A.data; pa != aend; ++pa) {
        if (isnan(*pa) || isinf(*pa))
            throw ComputationError(this->getId(), "Error in stiffness matrix at position {0}", pa-A.data);
    }
#endif

}


double DynamicThermalFem3DSolver::compute(double time)
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

        A.solve(T, temperatures);

        if (T.data() == X.data()) std::swap(temperatures, X);

        std::swap(temperatures, X);

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


void DynamicThermalFem3DSolver::saveHeatFluxes()
{
    this->writelog(LOG_DETAIL, "Computing heat fluxes");

    fluxes.reset(this->maskedMesh->getElementsCount());

    for (auto el: this->maskedMesh->elements())
    {
        Vec<3,double> midpoint = el.getMidpoint();
        auto material = this->geometry->getMaterial(midpoint);

        size_t lll = el.getLoLoLoIndex();
        size_t llu = el.getLoLoUpIndex();
        size_t lul = el.getLoUpLoIndex();
        size_t luu = el.getLoUpUpIndex();
        size_t ull = el.getUpLoLoIndex();
        size_t ulu = el.getUpLoUpIndex();
        size_t uul = el.getUpUpLoIndex();
        size_t uuu = el.getUpUpUpIndex();

        double temp = 0.125 * (temperatures[lll] + temperatures[llu] + temperatures[lul] + temperatures[luu] +
                               temperatures[ull] + temperatures[ulu] + temperatures[uul] + temperatures[uuu]);

        double kxy, kz;
        auto leaf = dynamic_pointer_cast<const GeometryObjectD<3>>(geometry->getMatchingAt(midpoint, &GeometryObject::PredicateIsLeaf));
        if (leaf)
            std::tie(kxy,kz) = std::tuple<double,double>(material->thermk(temp, leaf->getBoundingBox().height()));
        else
            std::tie(kxy,kz) = std::tuple<double,double>(material->thermk(temp));

        fluxes[el.getIndex()] = vec(
            - 0.25e6 * kxy * (- temperatures[lll] - temperatures[llu] - temperatures[lul] - temperatures[luu]
                              + temperatures[ull] + temperatures[ulu] + temperatures[uul] + temperatures[uuu])
                / (el.getUpper0() - el.getLower0()), // 1e6 - from µm to m
            - 0.25e6 * kxy * (- temperatures[lll] - temperatures[llu] + temperatures[lul] + temperatures[luu]
                              - temperatures[ull] - temperatures[ulu] + temperatures[uul] + temperatures[uuu])
                / (el.getUpper1() - el.getLower1()), // 1e6 - from µm to m
            - 0.25e6 * kz  * (- temperatures[lll] + temperatures[llu] - temperatures[lul] + temperatures[luu]
                              - temperatures[ull] + temperatures[ulu] - temperatures[uul] + temperatures[uuu])
                / (el.getUpper2() - el.getLower2()) // 1e6 - from µm to m
        );
    }
}


const LazyData<double> DynamicThermalFem3DSolver::getTemperatures(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) const {
    this->writelog(LOG_DEBUG, "Getting temperatures");
    if (!temperatures) return LazyData<double>(dst_mesh->size(), inittemp); // in case the receiver is connected and no temperature calculated yet
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    if (this->maskedMesh->full())
        return SafeData<double>(interpolate(this->mesh, temperatures, dst_mesh, method, this->geometry), 300.);
    else
        return SafeData<double>(interpolate(this->maskedMesh, temperatures, dst_mesh, method, this->geometry), 300.);
}


const LazyData<Vec<3>> DynamicThermalFem3DSolver::getHeatFluxes(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) {
    this->writelog(LOG_DEBUG, "Getting heat fluxes");
    if (!temperatures) return LazyData<Vec<3>>(dst_mesh->size(), Vec<3>(0.,0.,0.)); // in case the receiver is connected and no fluxes calculated yet
    if (!fluxes) saveHeatFluxes(); // we will compute fluxes only if they are needed
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    if (this->maskedMesh->full())
        return SafeData<Vec<3>>(interpolate(this->mesh->getElementMesh(), fluxes, dst_mesh, method,
                                            InterpolationFlags(this->geometry, InterpolationFlags::Symmetry::NPP, InterpolationFlags::Symmetry::PNP, InterpolationFlags::Symmetry::PPN)),
                                Zero<Vec<3>>());
    else
        return SafeData<Vec<3>>(interpolate(this->maskedMesh->getElementMesh(), fluxes, dst_mesh, method,
                                            InterpolationFlags(this->geometry,InterpolationFlags::Symmetry::NPP, InterpolationFlags::Symmetry::PNP, InterpolationFlags::Symmetry::PPN)),
                                Zero<Vec<3>>());
}


DynamicThermalFem3DSolver::
ThermalConductivityData::ThermalConductivityData(const DynamicThermalFem3DSolver* solver, const shared_ptr<const MeshD<3>>& dst_mesh):
    solver(solver), dest_mesh(dst_mesh), flags(solver->geometry)
{
    if (solver->temperatures) temps = interpolate(solver->maskedMesh, solver->temperatures, solver->maskedMesh->getElementMesh(), INTERPOLATION_LINEAR);
    else temps = LazyData<double>(solver->mesh->getElementsCount(), solver->inittemp);
}
Tensor2<double> DynamicThermalFem3DSolver::ThermalConductivityData::at(std::size_t i) const {
    auto point = flags.wrap(dest_mesh->at(i));
    std::size_t x = solver->mesh->axis[0]->findUpIndex(point[0]),
                y = solver->mesh->axis[1]->findUpIndex(point[1]),
                z = solver->mesh->axis[2]->findUpIndex(point[2]);
    if (x == 0 || y == 0 || z == 0 || x == solver->mesh->axis[0]->size() || y == solver->mesh->axis[1]->size() || z == solver->mesh->axis[2]->size())
        return Tensor2<double>(NAN);
    else {
        auto elem = solver->maskedMesh->element(x-1, y-1, z-1);
        auto material = solver->geometry->getMaterial(elem.getMidpoint());
        size_t idx = elem.getIndex();
        if (idx == RectangularMaskedMesh3D::Element::UNKNOWN_ELEMENT_INDEX) return Tensor2<double>(NAN);
        return material->thermk(temps[idx], solver->thickness[idx]);
    }
}
std::size_t DynamicThermalFem3DSolver::ThermalConductivityData::size() const { return dest_mesh->size(); }

const LazyData<Tensor2<double>> DynamicThermalFem3DSolver::getThermalConductivity(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod /*method*/) {
    this->initCalculation();
    this->writelog(LOG_DEBUG, "Getting thermal conductivities");
    return LazyData<Tensor2<double>>(new DynamicThermalFem3DSolver::ThermalConductivityData(this, dst_mesh));
}


}}} // namespace plask::thermal::thermal
