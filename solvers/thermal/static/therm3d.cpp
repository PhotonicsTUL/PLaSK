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

#include "therm3d.hpp"

namespace plask { namespace thermal { namespace tstatic {

ThermalFem3DSolver::ThermalFem3DSolver(const std::string& name) :
    FemSolverWithMaskedMesh<Geometry3D, RectangularMesh<3>>(name),
    loopno(0),
    inittemp(300.),
    maxerr(0.05),
    outTemperature(this, &ThermalFem3DSolver::getTemperatures),
    outHeatFlux(this, &ThermalFem3DSolver::getHeatFluxes),
    outThermalConductivity(this, &ThermalFem3DSolver::getThermalConductivity)
{
    temperatures.reset();
    fluxes.reset();
    inHeat = 0.;
}


ThermalFem3DSolver::~ThermalFem3DSolver() {
}


void ThermalFem3DSolver::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "temperature")
            this->readBoundaryConditions(manager, source, temperature_boundary);

        else if (param == "heatflux")
            this->readBoundaryConditions(manager, source, heatflux_boundary);

        else if (param == "convection")
            this->readBoundaryConditions(manager, source, convection_boundary);

        else if (param == "radiation")
            this->readBoundaryConditions(manager, source, radiation_boundary);

        else if (param == "loop") {
            inittemp = source.getAttribute<double>("inittemp", inittemp);
            maxerr = source.getAttribute<double>("maxerr", maxerr);
            source.requireTagEnd();
        }

        else if (!parseFemConfiguration(source, manager)) {
            this->parseStandardConfiguration(source, manager);
        }
    }
}


void ThermalFem3DSolver::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());

    FemSolverWithMaskedMesh<Geometry3D, RectangularMesh<3>>::onInitialize();

    loopno = 0;

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
        for (size_t r = ibottom; r < itop; ++r) {
            size_t idx = this->maskedMesh->element(elem.getIndex0(), elem.getIndex1(), r).getIndex();
            if (idx != RectangularMaskedMesh3D::Element::UNKNOWN_ELEMENT_INDEX)
                thickness[idx] = h;
        }
    }
}


void ThermalFem3DSolver::onInvalidate() {
    temperatures.reset();
    fluxes.reset();
    thickness.reset();
}

void ThermalFem3DSolver::setAlgorithm(Algorithm alg) {
    //TODO
    algorithm = alg;
}

/**
    * Helper function for applying boundary conditions of element edges to stiffness matrix.
    * Boundary conditions must be set for both nodes at the element edge.
    * \param boundary_conditions boundary conditions holder
    * \param idx indices of the element nodes
    * \param dx, dy, dz dimentions of the element
    * \param[out] F the load vector
    * \param[out] K stiffness matrix
    * \param F_function function returning load vector component
    * \param K_function function returning stiffness matrix component
    */
template <typename ConditionT>
static void setBoundaries(const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary,ConditionT>& boundary_conditions,
                          const size_t (&idx)[8], double dx, double dy, double dz, double (&F)[8], double (&K)[8][8],
                          const std::function<double(double,ConditionT,size_t)>& F_function,
                          const std::function<double(double,ConditionT,ConditionT,size_t,size_t,bool)>& K_function
                         )
{
    plask::optional<ConditionT> values[8];
    for (int i = 0; i < 8; ++i) values[i] = boundary_conditions.getValue(idx[i]);

    constexpr int walls[6][4] = { {0,1,2,3}, {4,5,6,7}, {0,2,4,6}, {1,3,5,7}, {0,1,4,5}, {2,3,6,7} };
    const double areas[3] = { dx*dy, dy*dz, dz*dx };

    for (int side = 0; side < 6; ++side) {
        const auto& wall = walls[side];
        if (values[wall[0]] && values[wall[1]] && values[wall[2]] && values[wall[3]]) {
            double area = areas[side/2];
            for (int i = 0; i < 4; ++i) {
                F[i] += F_function(area, *values[wall[i]], wall[i]);
                for (int j = 0; j <= i; ++j) {
                    int ij = i ^ j; // numbers on the single edge differ by one bit only, so detect different bits
                    bool edge = (ij == 1 || ij == 2 || ij == 4);
                    K[i][j] += K_function(area, *values[wall[i]], *values[wall[j]], wall[i], wall[j], edge);
                }
            }
        }

    }
}

template <typename MatrixT>
void ThermalFem3DSolver::setMatrix(MatrixT& A, DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary,double>& btemperature,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary,double>& bheatflux,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary,Convection>& bconvection,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary,Radiation>& bradiation
                  )
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size={0}, bands={1}({2}))", A.size, A.kd+1, A.ld+1);

    auto heats = inHeat(maskedMesh->getElementMesh()/*, INTERPOLATION_NEAREST*/);

    // zero the matrix and the load vector
    A.clear();
    B.fill(0.);

    // Set stiffness matrix and load vector
    for (auto elem: maskedMesh->elements())
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

        // load vector: heat densities
        double f = 0.125e-18 * dx * dy * dz * heats[elem.getIndex()];   // 1e-18 -> to transform µm³ into m³

        // set symmetric matrix components
        double K[8][8];
        K[0][0] = K[1][1] = K[2][2] = K[3][3] = K[4][4] = K[5][5] = K[6][6] = K[7][7] = (kx + ky + kz) / 9.;

        K[1][0] = K[3][2] = K[5][4] = K[7][6] = (-2.*kx +    ky +    kz) / 18.;
        K[2][0] = K[3][1] = K[6][4] = K[7][5] = (    kx - 2.*ky +    kz) / 18.;
        K[4][0] = K[5][1] = K[6][2] = K[7][3] = (    kx +    ky - 2.*kz) / 18.;

        K[4][2] = K[5][3] = K[6][0] = K[7][1] = (    kx - 2.*ky - 2.*kz) / 36.;
        K[4][1] = K[5][0] = K[6][3] = K[7][2] = (-2.*kx +    ky - 2.*kz) / 36.;
        K[2][1] = K[3][0] = K[6][5] = K[7][4] = (-2.*kx - 2.*ky +    kz) / 36.;

        K[4][3] = K[5][2] = K[6][1] = K[7][0] = -(kx + ky + kz) / 36.;

        double F[8];
        std::fill_n(F, 8, f);

        // boundary conditions: heat flux
        setBoundaries<double>(bheatflux, idx, dx, dy, dz, F, K,
                              [](double area, double value, size_t) { // F
                                  return - 0.25e-12 * area * value;
                              },
                              [](double, double, double, size_t, size_t, bool) { return 0.; }  // K
                             );

        // boundary conditions: convection
        setBoundaries<Convection>(bconvection, idx, dx, dy, dz, F, K,
                                  [](double area, Convection value, size_t) { // F
                                      return 0.25e-12 * area * value.coeff * value.ambient;
                                  },
                                  [](double area, Convection value1, Convection value2, size_t i1, size_t i2, bool edge) -> double { // K
                                      double v = 0.125e-12 * area * (value1.coeff + value2.coeff);
                                      return v / (i2==i1? 9. : edge? 18. : 36.);
                                  }
                                 );

        // boundary conditions: radiation
        setBoundaries<Radiation>(bradiation, idx, dx, dy, dz, F, K,
                                 [this](double area, Radiation value, size_t i) -> double { // F
                                     double a = value.ambient; a = a*a;
                                     double T = this->temperatures[i]; T = T*T;
                                     return - 0.25e-12 * area * value.emissivity * phys::SB * (T*T - a*a);},
                                 [](double, Radiation, Radiation, size_t, size_t, bool) {return 0.;} // K
                                );

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j <= i; ++j) {
                A(idx[i],idx[j]) += K[i][j];
            }
            B[idx[i]] += F[i];
        }
    }

    A.applyBC(B, btemperature);
}

double ThermalFem3DSolver::compute(int loops)
{
    this->initCalculation();

    fluxes.reset();

    // store boundary conditions for current mesh
    auto btemperature = temperature_boundary(this->maskedMesh, this->geometry);
    auto bheatflux = heatflux_boundary(this->maskedMesh, this->geometry);
    auto bconvection = convection_boundary(this->maskedMesh, this->geometry);
    auto bradiation = radiation_boundary(this->maskedMesh, this->geometry);

    this->writelog(LOG_INFO, "Running thermal calculations");

    int loop = 0;
    size_t size = maskedMesh->size();

    std::unique_ptr<FemMatrix> pA(getMatrix());
    FemMatrix& A = *pA.get();

    double err = 0.;
    toterr = 0.;

#   ifndef NDEBUG
        if (!temperatures.unique()) this->writelog(LOG_DEBUG, "Temperature data held by something else...");
#   endif
    temperatures = temperatures.claim();

    DataVector<double> BT(size), temp0(size);

    do {
        setMatrix(A, BT, btemperature, bheatflux, bconvection, bradiation);
        std::copy(temperatures.begin(), temperatures.end(), temp0.begin());

        A.solve(BT, temperatures);

        if (BT.data() != temperatures.data()) std::swap(BT, temperatures);

        // Update error
        err = 0.;
        maxT = 0.;
        for (auto temp = temperatures.begin(), t = temp0.begin(); t != temp0.end(); ++temp, ++t)
        {
            double corr = std::abs(*t - *temp); // for boundary with constant temperature this will be zero anyway
            if (corr > err) err = corr;
            if (*t > maxT) maxT = *t;
        }
        if (err > toterr) toterr = err;

        ++loopno;
        ++loop;

        // show max correction
        this->writelog(LOG_RESULT, "Loop {:d}({:d}): max(T) = {:.3f} K, error = {:g} K", loop, loopno, maxT, err);

    } while (err > maxerr && (loops == 0 || loop < loops));

    outTemperature.fireChanged();
    outHeatFlux.fireChanged();

    return toterr;
}

void ThermalFem3DSolver::saveHeatFluxes()
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


const LazyData<double> ThermalFem3DSolver::getTemperatures(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) const {
    this->writelog(LOG_DEBUG, "Getting temperatures");
    if (!temperatures) return LazyData<double>(dst_mesh->size(), inittemp); // in case the receiver is connected and no temperature calculated yet
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    if (this->maskedMesh->full())
        return SafeData<double>(interpolate(this->mesh, temperatures, dst_mesh, method, this->geometry), 300.);
    else
        return SafeData<double>(interpolate(this->maskedMesh, temperatures, dst_mesh, method, this->geometry), 300.);
}


const LazyData<Vec<3>> ThermalFem3DSolver::getHeatFluxes(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) {
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


ThermalFem3DSolver::
ThermalConductivityData::ThermalConductivityData(const ThermalFem3DSolver* solver, const shared_ptr<const MeshD<3>>& dst_mesh):
    solver(solver), dest_mesh(dst_mesh), flags(solver->geometry)
{
    if (solver->temperatures) temps = interpolate(solver->maskedMesh, solver->temperatures, solver->maskedMesh->getElementMesh(), INTERPOLATION_LINEAR);
    else temps = LazyData<double>(solver->maskedMesh->getElementsCount(), solver->inittemp);
}
Tensor2<double> ThermalFem3DSolver::ThermalConductivityData::at(std::size_t i) const {
    auto point = flags.wrap(dest_mesh->at(i));
    std::size_t x = solver->mesh->axis[0]->findUpIndex(point[0]),
                y = solver->mesh->axis[1]->findUpIndex(point[1]),
                z = solver->mesh->axis[2]->findUpIndex(point[2]);
    if (x == 0 || y == 0 || z == 0 || x == solver->mesh->axis[0]->size() || y == solver->mesh->axis[1]->size() || z == solver->mesh->axis[2]->size())
        return Tensor2<double>(NAN);
    else {
        auto elem = solver->maskedMesh->element(x-1, y-1, z-1);
        size_t idx = elem.getIndex();
        if (idx == RectangularMaskedMesh3D::Element::UNKNOWN_ELEMENT_INDEX) return Tensor2<double>(NAN);
        auto material = solver->geometry->getMaterial(elem.getMidpoint());
        return material->thermk(temps[idx], solver->thickness[idx]);
    }
}
std::size_t ThermalFem3DSolver::ThermalConductivityData::size() const { return dest_mesh->size(); }

const LazyData<Tensor2<double>> ThermalFem3DSolver::getThermalConductivity(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod /*method*/) {
    this->initCalculation();
    this->writelog(LOG_DEBUG, "Getting thermal conductivities");
    return LazyData<Tensor2<double>>(new ThermalFem3DSolver::ThermalConductivityData(this, dst_mesh));
}


}}} // namespace plask::thermal::tstatic
