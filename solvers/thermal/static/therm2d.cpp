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
#include "therm2d.hpp"

namespace plask { namespace thermal { namespace tstatic {

template<typename Geometry2DType>
ThermalFem2DSolver<Geometry2DType>::ThermalFem2DSolver(const std::string& name) :
    FemSolverWithMaskedMesh<Geometry2DType, RectangularMesh<2>>(name),
    loopno(0),
    outTemperature(this, &ThermalFem2DSolver<Geometry2DType>::getTemperatures),
    outHeatFlux(this, &ThermalFem2DSolver<Geometry2DType>::getHeatFluxes),
    outThermalConductivity(this, &ThermalFem2DSolver<Geometry2DType>::getThermalConductivity),
    maxerr(0.05),
    inittemp(300.)
{
    temperatures.reset();
    fluxes.reset();
    inHeat = 0.;
}


template<typename Geometry2DType>
ThermalFem2DSolver<Geometry2DType>::~ThermalFem2DSolver() {
}


template<typename Geometry2DType>
void ThermalFem2DSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
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

        else if (!this->parseFemConfiguration(source, manager)) {
            this->parseStandardConfiguration(source, manager);
        }
    }
}


template<typename Geometry2DType>
void ThermalFem2DSolver<Geometry2DType>::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());

    FemSolverWithMaskedMesh<Geometry2DType, RectangularMesh<2>>::onInitialize();

    loopno = 0;
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
            } else break;
        }
        for (size_t r = elem.getIndex1()+1; r < this->mesh->axis[1]->size()-1; r++) {
            auto e = this->mesh->element(c, r);
            auto m = this->geometry->getMaterial(e.getMidpoint());
            if (m == material) {                            //TODO ignore doping
                top = e.getUpper1();
                itop = r+1;
            } else break;
        }
        double h = top - bottom;
        for (size_t r = ibottom; r != itop; ++r) {
            size_t idx = this->maskedMesh->element(c, r).getIndex();
            if (idx != RectangularMaskedMesh2D::Element::UNKNOWN_ELEMENT_INDEX)
                thickness[idx] = h;
        }
    }
}


template<typename Geometry2DType> void ThermalFem2DSolver<Geometry2DType>::onInvalidate() {
    temperatures.reset();
    fluxes.reset();
    thickness.reset();
}

enum BoundarySide { LEFT, RIGHT, TOP, BOTTOM };

/**
    * Helper function for applying boundary conditions of element edges to stiffness matrix.
    * Boundary conditions must be set for both nodes at the element edge.
    * \param boundary_conditions boundary conditions holder
    * \param i1, i2, i3, i4 indices of the lower left, lower right, upper right, and upper left node
    * \param width width of the element
    * \param height height of the element
    * \param[out] F1, F2, F3, F4 references to the load vector components
    * \param[out] K11, K22, K33, K44, K12, K14, K24, K34 references to the stiffness matrix components
    * \param F_function function returning load vector component
    * \param Kmm_function function returning stiffness matrix diagonal component
    * \param Kmn_function function returning stiffness matrix off-diagonal component
    */
template <typename ConditionT>
static void setBoundaries(const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,ConditionT>& boundary_conditions,
                          size_t i1, size_t i2, size_t i3, size_t i4, double width, double height,
                          double& F1, double& F2, double& F3, double& F4,
                          double& K11, double& K22, double& K33, double& K44,
                          double& K12, double& K23, double& K34, double& K41,
                          const std::function<double(double,ConditionT,ConditionT,size_t,size_t,BoundarySide)>& F_function,
                          const std::function<double(double,ConditionT,ConditionT,size_t,size_t,BoundarySide)>& Kmm_function,
                          const std::function<double(double,ConditionT,ConditionT,size_t,size_t,BoundarySide)>& Kmn_function
                         )
{
    auto val1 = boundary_conditions.getValue(i1);
    auto val2 = boundary_conditions.getValue(i2);
    auto val3 = boundary_conditions.getValue(i3);
    auto val4 = boundary_conditions.getValue(i4);
    if (val1 && val2) { // bottom
        F1 += F_function(width, *val1, *val2, i1, i2, BOTTOM); F2 += F_function(width, *val2, *val1, i2, i1, BOTTOM);
        K11 += Kmm_function(width, *val1, *val2, i1, i2, BOTTOM); K22 += Kmm_function(width, *val2, *val1, i2, i1, BOTTOM);
        K12 += Kmn_function(width, *val1, *val2, i1, i2, BOTTOM);
    }
    if (val2 && val3) { // right
        F2 += F_function(height, *val2, *val3, i2, i3, RIGHT); F3 += F_function(height, *val3, *val2, i3, i2, RIGHT);
        K22 += Kmm_function(height, *val2, *val3, i2, i3, RIGHT); K33 += Kmm_function(height, *val3, *val2, i3, i2, RIGHT);
        K23 += Kmn_function(height, *val2, *val3, i2, i3, RIGHT);
    }
    if (val3 && val4) { // top
        F3 += F_function(width, *val3, *val4, i3, i4, TOP); F4 += F_function(width, *val4, *val3, i4, i3, TOP);
        K33 += Kmm_function(width, *val3, *val4, i3, i4, TOP); K44 += Kmm_function(width, *val4, *val3, i4, i3, TOP);
        K34 += Kmn_function(width, *val3, *val4, i3, i4, TOP);
    }
    if (val4 && val1) { // left
        F1 += F_function(height, *val1, *val4, i1, i4, LEFT); F4 += F_function(height, *val4, *val1, i4, i1, LEFT);
        K11 += Kmm_function(height, *val1, *val4, i1, i4, LEFT); K44 += Kmm_function(height, *val4, *val1, i4, i1, LEFT);
        K41 += Kmn_function(height, *val1, *val4, i1, i4, LEFT);
    }
}


template<>
void ThermalFem2DSolver<Geometry2DCartesian>::setMatrix(FemMatrix& A, DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& btemperature,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& bheatflux,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,Convection>& bconvection,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,Radiation>& bradiation
                  )
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size={0}, bands={1}({2}))", A.size, A.kd+1, A.ld+1);

    auto iMesh = (this->maskedMesh)->getElementMesh();
    auto heatdensities = inHeat(iMesh);

    A.clear();
    B.fill(0.);

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

        kx *= elemheight; kx /= elemwidth;
        ky *= elemwidth; ky /= elemheight;

        // load vector: heat densities
        double f = 0.25e-12 * elemwidth * elemheight * heatdensities[elem.getIndex()]; // 1e-12 -> to transform µm² into m²

        // set symmetric matrix components
        double k44, k33, k22, k11, k43, k21, k42, k31, k32, k41;

        k44 = k33 = k22 = k11 = (kx + ky) / 3.;
        k43 = k21 = (-2. * kx + ky) / 6.;
        k42 = k31 = - (kx + ky) / 6.;
        k32 = k41 = (kx - 2. * ky) / 6.;

        double f1 = f, f2 = f, f3 = f, f4 = f;

        // boundary conditions: heat flux
        setBoundaries<double>(bheatflux, loleftno, lorghtno, uprghtno, upleftno, elemwidth, elemheight,
                      f1, f2, f3, f4, k11, k22, k33, k44, k21, k32, k43, k41,
                      [](double len, double val, double, size_t, size_t, BoundarySide) { // F
                          return - 0.5e-6 * len * val;
                      },
                      [](double, double, double, size_t, size_t, BoundarySide){return 0.;}, // K diagonal
                      [](double, double, double, size_t, size_t, BoundarySide){return 0.;}  // K off-diagonal
                     );

        // boundary conditions: convection
        setBoundaries<Convection>(bconvection, loleftno, lorghtno, uprghtno, upleftno, elemwidth, elemheight,
                      f1, f2, f3, f4, k11, k22, k33, k44, k21, k32, k43, k41,
                      [](double len, Convection val, Convection, size_t, size_t, BoundarySide) { // F
                          return 0.5e-6 * len * val.coeff * val.ambient;
                      },
                      [](double len, Convection val1, Convection val2, size_t, size_t, BoundarySide) { // K diagonal
                          return (val1.coeff + val2.coeff) * len / 6.;
                      },
                      [](double len, Convection val1, Convection val2, size_t, size_t, BoundarySide) { // K off-diagonal
                          return (val1.coeff + val2.coeff) * len / 12.;
                      }
                     );

        // boundary conditions: radiation
        setBoundaries<Radiation>(bradiation, loleftno, lorghtno, uprghtno, upleftno, elemwidth, elemheight,
                      f1, f2, f3, f4, k11, k22, k33, k44, k21, k32, k43, k41,
                      [this](double len, Radiation val, Radiation, size_t i, size_t, BoundarySide) -> double { // F
                          double a = val.ambient; a = a*a;
                          double T = this->temperatures[i]; T = T*T;
                          return - 0.5e-6 * len * val.emissivity * phys::SB * (T*T - a*a);},
                      [](double, Radiation, Radiation, size_t, size_t, BoundarySide){return 0.;}, // K diagonal
                      [](double, Radiation, Radiation, size_t, size_t, BoundarySide){return 0.;}  // K off-diagonal
                     );

        // set stiffness matrix
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

        // set load vector
        B[loleftno] += f1;
        B[lorghtno] += f2;
        B[uprghtno] += f3;
        B[upleftno] += f4;
    }

    // boundary conditions of the first kind
    A.applyBC(btemperature, B);

#ifndef NDEBUG
    double* aend = A.data + A.size * A.kd;
    for (double* pa = A.data; pa != aend; ++pa) {
        if (isnan(*pa) || isinf(*pa))
            throw ComputationError(this->getId(), "Error in stiffness matrix at position {0}", pa-A.data);
    }
#endif

}


template<>
void ThermalFem2DSolver<Geometry2DCylindrical>::setMatrix(FemMatrix& A, DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& btemperature,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& bheatflux,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,Convection>& bconvection,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,Radiation>& bradiation
                  )
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size={0}, bands={1}({2}))", A.size, A.kd+1, A.ld+1);

    auto iMesh = (this->maskedMesh)->getElementMesh();
    auto heatdensities = inHeat(iMesh);

    std::fill_n(A.data, A.size*(A.ld+1), 0.); // zero the matrix
    B.fill(0.);

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
        auto material = geometry->getMaterial(midpoint);
        double r = midpoint.rad_r();

        // average temperature on the element
        double temp = 0.25 * (temperatures[loleftno] + temperatures[lorghtno] + temperatures[upleftno] + temperatures[uprghtno]);

        // thermal conductivity
        double kx, ky;
        std::tie(kx,ky) = std::tuple<double,double>(material->thermk(temp, thickness[elem.getIndex()]));

        kx = kx * elemheight / elemwidth;
        ky = ky * elemwidth / elemheight;

        // load vector: heat densities
        double f = 0.25e-12 * r * elemwidth * elemheight * heatdensities[elem.getIndex()]; // 1e-12 -> to transform µm² into m²

        // set symmetric matrix components
        double k44, k33, k22, k11, k43, k21, k42, k31, k32, k41;

        k44 = k33 = k22 = k11 = (kx + ky) / 3.;
        k43 = k21 = (-2. * kx + ky) / 6.;
        k42 = k31 = - (kx + ky) / 6.;
        k32 = k41 = (kx - 2. * ky) / 6.;

        double f1 = f, f2 = f, f3 = f, f4 = f;

        // boundary conditions: heat flux
        setBoundaries<double>(bheatflux, loleftno, lorghtno, uprghtno, upleftno, elemwidth, elemheight,
                      f1, f2, f3, f4, k11, k22, k33, k44, k21, k32, k43, k41,
                      [&](double len, double val, double, size_t i1, size_t i2, BoundarySide side) -> double { // F
                            if (side == LEFT) return - 0.5e-6 * len * val * elem.getLower0();
                            else if (side == RIGHT) return - 0.5e-6 * len * val * elem.getUpper0();
                            else return - 0.5e-6 * len * val * (r + (i1<i2? -len/6. : len/6.));
                      },
                      [](double, double, double, size_t, size_t, BoundarySide){return 0.;}, // K diagonal
                      [](double, double, double, size_t, size_t, BoundarySide){return 0.;}  // K off-diagonal
                     );

        // boundary conditions: convection
        setBoundaries<Convection>(bconvection, loleftno, lorghtno, uprghtno, upleftno, elemwidth, elemheight,
                      f1, f2, f3, f4, k11, k22, k33, k44, k21, k32, k43, k41,
                      [&](double len, Convection val1, Convection val2, size_t i1, size_t i2, BoundarySide side) -> double { // F
                          double a = 0.125e-6 * len * (val1.coeff + val2.coeff) * (val1.ambient + val2.ambient);
                            if (side == LEFT) return a * elem.getLower0();
                            else if (side == RIGHT) return a * elem.getUpper0();
                            else return a * (r + (i1<i2? -len/6. : len/6.));

                      },
                      [&](double len, Convection val1, Convection val2, size_t i1, size_t i2, BoundarySide side) -> double { // K diagonal
                            double a = (val1.coeff + val2.coeff) * len / 6.;
                            if (side == LEFT) return a * elem.getLower0();
                            else if (side == RIGHT) return a * elem.getUpper0();
                            else return a * (r + (i1<i2? -len/6. : len/6.));
                      },
                      [&](double len, Convection val1, Convection val2, size_t, size_t, BoundarySide side) -> double { // K off-diagonal
                            double a = (val1.coeff + val2.coeff) * len / 12.;
                            if (side == LEFT) return a * elem.getLower0();
                            else if (side == RIGHT) return a * elem.getUpper0();
                            else return a * r;
                      }
                     );

        // boundary conditions: radiation
        setBoundaries<Radiation>(bradiation, loleftno, lorghtno, uprghtno, upleftno, elemwidth, elemheight,
                      f1, f2, f3, f4, k11, k22, k33, k44, k21, k32, k43, k41,
                      [&,this](double len, Radiation val, Radiation, size_t i1,  size_t i2, BoundarySide side) -> double { // F
                            double amb = val.ambient; amb = amb*amb;
                            double T = this->temperatures[i1]; T = T*T;
                            double a = - 0.5e-6 * len * val.emissivity * phys::SB * (T*T - amb*amb);
                            if (side == LEFT) return a * elem.getLower0();
                            else if (side == RIGHT) return a * elem.getUpper0();
                            else return a * (r + (i1<i2? -len/6. : len/6.));
                      },
                      [](double, Radiation, Radiation, size_t, size_t, BoundarySide){return 0.;}, // K diagonal
                      [](double, Radiation, Radiation, size_t, size_t, BoundarySide){return 0.;}  // K off-diagonal
                     );

        // set stiffness matrix
        A(loleftno, loleftno) += r * k11;
        A(lorghtno, lorghtno) += r * k22;
        A(uprghtno, uprghtno) += r * k33;
        A(upleftno, upleftno) += r * k44;

        A(lorghtno, loleftno) += r * k21;
        A(uprghtno, loleftno) += r * k31;
        A(upleftno, loleftno) += r * k41;
        A(uprghtno, lorghtno) += r * k32;
        A(upleftno, lorghtno) += r * k42;
        A(upleftno, uprghtno) += r * k43;

        // set load vector
        B[loleftno] += f1;
        B[lorghtno] += f2;
        B[uprghtno] += f3;
        B[upleftno] += f4;
    }

    // boundary conditions of the first kind
    A.applyBC(btemperature, B);

#ifndef NDEBUG
    double* aend = A.data + A.size * A.kd;
    for (double* pa = A.data; pa != aend; ++pa) {
        if (isnan(*pa) || isinf(*pa))
            throw ComputationError(this->getId(), "Error in stiffness matrix at position {0}", pa-A.data);
    }
#endif

}


template<typename Geometry2DType>
double ThermalFem2DSolver<Geometry2DType>::compute(int loops){
    this->initCalculation();

    fluxes.reset();

    // store boundary conditions for current mesh
    auto btemperature = temperature_boundary(this->maskedMesh, this->geometry);
    auto bheatflux = heatflux_boundary(this->maskedMesh, this->geometry);
    auto bconvection = convection_boundary(this->maskedMesh, this->geometry);
    auto bradiation = radiation_boundary(this->maskedMesh, this->geometry);

    this->writelog(LOG_INFO, "Running thermal calculations");

    int loop = 0;
    size_t size = this->maskedMesh->size();

    std::unique_ptr<FemMatrix> pA(this->getMatrix());
    FemMatrix& A = *pA.get();

    double err;
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
            if (*temp > maxT) maxT = *temp;
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

template<typename Geometry2DType>
void ThermalFem2DSolver<Geometry2DType>::saveHeatFluxes()
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
const LazyData<double> ThermalFem2DSolver<Geometry2DType>::getTemperatures(const shared_ptr<const MeshD<2>>& dest_mesh, InterpolationMethod method) const {
    this->writelog(LOG_DEBUG, "Getting temperatures");
    if (!temperatures) return LazyData<double>(dest_mesh->size(), inittemp); // in case the receiver is connected and no temperature calculated yet
    if (method == INTERPOLATION_DEFAULT) method = INTERPOLATION_LINEAR;
    if (this->maskedMesh->full())
        return SafeData<double>(interpolate(this->mesh, temperatures, dest_mesh, method, this->geometry), 300.);
    else
        return SafeData<double>(interpolate(this->maskedMesh, temperatures, dest_mesh, method, this->geometry), 300.);
}


template<typename Geometry2DType>
const LazyData<Vec<2>> ThermalFem2DSolver<Geometry2DType>::getHeatFluxes(const shared_ptr<const MeshD<2>>& dest_mesh, InterpolationMethod method) {
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


template<typename Geometry2DType> ThermalFem2DSolver<Geometry2DType>::
ThermalConductivityData::ThermalConductivityData(const ThermalFem2DSolver<Geometry2DType>* solver, const shared_ptr<const MeshD<2>>& dst_mesh):
    solver(solver), dest_mesh(dst_mesh), flags(solver->geometry)
{
    if (solver->temperatures) temps = interpolate(solver->maskedMesh, solver->temperatures, solver->maskedMesh->getElementMesh(), INTERPOLATION_LINEAR);
    else temps = LazyData<double>(solver->maskedMesh->getElementsCount(), solver->inittemp);
}

template<typename Geometry2DType> Tensor2<double> ThermalFem2DSolver<Geometry2DType>::
ThermalConductivityData::at(std::size_t i) const {
    auto point = flags.wrap(dest_mesh->at(i));
    std::size_t x = solver->mesh->axis[0]->findUpIndex(point[0]),
                y = solver->mesh->axis[1]->findUpIndex(point[1]);
    if (x == 0 || y == 0 || x == solver->mesh->axis[0]->size() || y == solver->mesh->axis[1]->size())
        return Tensor2<double>(NAN);
    else {
        auto elem = solver->maskedMesh->element(x-1, y-1);
        size_t idx = elem.getIndex();
        if (idx == RectangularMaskedMesh2D::Element::UNKNOWN_ELEMENT_INDEX) return Tensor2<double>(NAN);
        auto material = solver->geometry->getMaterial(elem.getMidpoint());
        return material->thermk(temps[idx], solver->thickness[idx]);
    }
}

template<typename Geometry2DType> std::size_t ThermalFem2DSolver<Geometry2DType>::
ThermalConductivityData::size() const { return dest_mesh->size(); }

template<typename Geometry2DType>
const LazyData<Tensor2<double>> ThermalFem2DSolver<Geometry2DType>::getThermalConductivity(const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod) {
    this->initCalculation();
    this->writelog(LOG_DEBUG, "Getting thermal conductivities");
    return LazyData<Tensor2<double>>(new
        ThermalFem2DSolver<Geometry2DType>::ThermalConductivityData(this, dst_mesh)
    );
}


template<> std::string ThermalFem2DSolver<Geometry2DCartesian>::getClassName() const { return "thermal.Static2D"; }
template<> std::string ThermalFem2DSolver<Geometry2DCylindrical>::getClassName() const { return "thermal.StaticCyl"; }

template struct PLASK_SOLVER_API ThermalFem2DSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API ThermalFem2DSolver<Geometry2DCylindrical>;

}}} // namespace plask::thermal::tstatic
