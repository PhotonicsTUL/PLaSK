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
#ifndef PLASK__MODULE_ELECTRICAL_CAPACITANCE2D_H
#define PLASK__MODULE_ELECTRICAL_CAPACITANCE2D_H

#include <plask/plask.hpp>

#include "complex_fem_solver.hpp"

namespace plask { namespace electrical { namespace capacitance {

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template <typename Geometry2DType>
struct PLASK_SOLVER_API Capacitance2DSolver : public ComplexFemSolverWithMaskedMesh<Geometry2DType, RectangularMesh<2>> {
  protected:
    /// Details of active region
    struct Active {
        struct Region {
            size_t left, right, bottom, top;
            size_t rowl, rowr;
            bool warn;
            Region()
                : left(0),
                  right(0),
                  bottom(std::numeric_limits<size_t>::max()),
                  top(std::numeric_limits<size_t>::max()),
                  rowl(std::numeric_limits<size_t>::max()),
                  rowr(0),
                  warn(true) {}
        };
        size_t left, right, bottom, top;
        ptrdiff_t offset;
        double height;
        Active() : left(0), right(0), bottom(0), top(0), offset(0), height(0.) {}
        Active(size_t tot, size_t l, size_t r, size_t b, size_t t, double h)
            : left(l), right(r), bottom(b), top(t), offset(tot - l), height(h) {}
    };

    std::vector<Active> active;  ///< Active regions information

    DataVector<dcomplex> potentials;               ///< Computed potentials
    DataVector<Tensor2<dcomplex>> conductivities;  ///< Conductivities in mesh points
    DataVector<Vec<2, dcomplex>> currents;         ///< Computed current densities

    double frequency = 20.;  ///< Frequency of AC signal [GHz]

    /// Save locate stiffness matrix to global one
    inline void setLocalMatrix(dcomplex& k44,
                               dcomplex& k33,
                               dcomplex& k22,
                               dcomplex& k11,
                               dcomplex& k43,
                               dcomplex& k21,
                               dcomplex& k42,
                               dcomplex& k31,
                               dcomplex& k32,
                               dcomplex& k41,
                               dcomplex ky,
                               double width,
                               const Vec<2, double>& midpoint);

    /// Initialize the solver
    void onInitialize() override;

    /// Invalidate the data
    void onInvalidate() override;

    /** Return \c true if the specified point is at junction
     * \param point point to test
     * \returns number of active region + 1 (0 for none)
     */
    size_t isActive(const Vec<2>& point) const {
        size_t no(0);
        auto roles = this->geometry->getRolesAt(point);
        for (auto role : roles) {
            size_t l = 0;
            if (role.substr(0, 6) == "active")
                l = 6;
            else if (role.substr(0, 8) == "junction")
                l = 8;
            else
                continue;
            if (no != 0) throw BadInput(this->getId(), "multiple 'active'/'junction' roles specified");
            if (role.size() == l)
                no = 1;
            else {
                try {
                    no = boost::lexical_cast<size_t>(role.substr(l)) + 1;
                } catch (boost::bad_lexical_cast&) {
                    throw BadInput(this->getId(), "bad junction number in role '{0}'", role);
                }
            }
        }
        return no;
    }

    /// Return \c true if the specified element is a junction
    size_t isActive(const RectangularMaskedMesh2D::Element& element) const { return isActive(element.getMidpoint()); }

    /// Get info on active region
    void setupActiveRegions();

    /// Set stiffness matrix + load vector
    void setMatrix(FemMatrix<dcomplex>& A,
                   DataVector<dcomplex>& B,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary, dcomplex>& bvoltage);

    void loadConductivities();

    Vec<2, dcomplex> getElementCurrentDensity(const RectangularMaskedMesh2D::Element& element, bool active_current = false) const;

    void computeCurrentDensities();

  public:
    /// Boundary condition
    BoundaryConditions<RectangularMesh<2>::Boundary, dcomplex> voltage_boundary;

    typename ProviderFor<AcVoltage, Geometry2DType>::Delegate outAcVoltage;

    typename ProviderFor<AcCurrentDensity, Geometry2DType>::Delegate outAcCurrentDensity;

    ReceiverFor<Conductivity, Geometry2DType> inDifferentialConductivity;

    ReceiverFor<Temperature, Geometry2DType> inTemperature;

    std::string getClassName() const override;

    /// Return current frequency
    double getFrequency() const { return frequency; }

    /// Set new frequency
    void setFrequency(double freq) {
        frequency = freq;
        this->invalidate();
    }

    /**
     * Run electrical calculations
     * \param loops maximum number of loops to run
     * \return max correction of potential against the last call
     **/
    void compute();

    /**
     * Integrate vertical total current (AC only) at certain level.
     * \param vindex vertical index of the element mesh to perform integration at
     * \param active_current if true, the active current (in phase with the voltage) is returned, otherwise the total current is returned
     * \param onlyactive if true only current in the active region is considered
     * \return computed total current
     */
    dcomplex integrateCurrent(size_t vindex, bool active_current = false, bool onlyactive = false) const;

    /**
     * Integrate vertical current (AC only) flowing vertically through active region.
     * \param nact number of the active region
     * \param active_current if true, the active current (in phase with the voltage) is returned, otherwise the total current is returned
     * \return computed total current
     */
    dcomplex getAcCurrent(size_t nact = 0, bool active_current = false) const;

    /**
     * Compute impedance at given frequency.
     */
    dcomplex getImpedance() const;

    /**
     * Compute structure S11 parameter at given frequency.
     * \param Z0 reference impedance (default 50 Ohm)
     * \return S11 parameter (complex)
     */
    dcomplex getS11(dcomplex Z0 = 50.) const;

    void loadConfiguration(XMLReader& source, Manager& manager) override;

    void parseConfiguration(XMLReader& source, Manager& manager);

    Capacitance2DSolver(const std::string& name = "");

    ~Capacitance2DSolver();

  protected:
    const LazyData<dcomplex> getVoltage(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method) const;

    const LazyData<Vec<2, dcomplex>> getCurrentDensities(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);
};

}}}  // namespace plask::electrical::capacitance

#endif
