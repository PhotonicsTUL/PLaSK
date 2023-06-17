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
#ifndef PLASK__MODULE_ELECTRICAL_ELECTR2D_H
#define PLASK__MODULE_ELECTRICAL_ELECTR2D_H

#include "common.hpp"

namespace plask { namespace electrical { namespace shockley {

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template <typename Geometry2DType>
struct PLASK_SOLVER_API ElectricalFem2DSolver : public FemSolverWithMaskedMesh<Geometry2DType, RectangularMesh<2>> {
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

    double pcond;  ///< p-contact electrical conductivity [S/m]
    double ncond;  ///< n-contact electrical conductivity [S/m]

    int loopno;     ///< Number of completed loops
    double toterr;  ///< Maximum estimated error during all iterations (useful for single calculations managed by external python
                    ///< script)
    Vec<2, double> maxcur;  ///< Maximum current in the structure

    DataVector<Tensor2<double>> junction_conductivity;  ///< electrical conductivity for p-n junction in y-direction [S/m]
    Tensor2<double> default_junction_conductivity;      ///< default electrical conductivity for p-n junction in y-direction [S/m]

    DataVector<Tensor2<double>> conds;    ///< Cached element conductivities
    DataVector<double> potentials;        ///< Computed potentials
    DataVector<Vec<2, double>> currents;  ///< Computed current densities
    DataVector<double> heats;             ///< Computed and cached heat source densities

    std::vector<Active> active;  ///< Active regions information

    /// Save locate stiffness matrix to global one
    inline void setLocalMatrix(double& k44,
                               double& k33,
                               double& k22,
                               double& k11,
                               double& k43,
                               double& k21,
                               double& k42,
                               double& k31,
                               double& k32,
                               double& k41,
                               double ky,
                               double width,
                               const Vec<2, double>& midpoint);

    /** Compute conductivity int the the active region
     *  \param n active region number
     *  \param U junction voltage [V]
     *  \param jy vertical current [kA/cm²]
     *  \param T temperature [K]
     */
    virtual Tensor2<double> activeCond(size_t n, double U, double jy, double T) = 0;

    /** Load conductivities
     *  \return current temperature
     */
    LazyData<double> loadConductivities();

    /// Save conductivities of active region
    void saveConductivities();

    /// Create 2D-vector with calculated heat densities
    void saveHeatDensities();

    /// Initialize the solver
    void onInitialize() override;

    /// Invalidate the data
    void onInvalidate() override;

    /// Get info on active region
    void setActiveRegions();

    void onMeshChange(const typename RectangularMesh<2>::Event& evt) override {
        SolverWithMesh<Geometry2DType, RectangularMesh<2>>::onMeshChange(evt);
        setActiveRegions();
    }

    void onGeometryChange(const Geometry::Event& evt) override {
        SolverWithMesh<Geometry2DType, RectangularMesh<2>>::onGeometryChange(evt);
        setActiveRegions();
    }

    /// Set stiffness matrix + load vector
    void setMatrix(FemMatrix& A,
                   DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary, double>& bvoltage,
                   const LazyData<double>& temperature);

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
            if (no != 0) throw BadInput(this->getId(), "Multiple 'active'/'junction' roles specified");
            if (role.size() == l)
                no = 1;
            else {
                try {
                    no = boost::lexical_cast<size_t>(role.substr(l)) + 1;
                } catch (boost::bad_lexical_cast&) {
                    throw BadInput(this->getId(), "Bad junction number in role '{0}'", role);
                }
            }
        }
        return no;
    }

    /// Return \c true if the specified element is a junction
    size_t isActive(const RectangularMaskedMesh2D::Element& element) const { return isActive(element.getMidpoint()); }

  public:
    double maxerr;  ///< Maximum relative current density correction accepted as convergence

    /// Boundary condition
    BoundaryConditions<RectangularMesh<2>::Boundary, double> voltage_boundary;

    typename ProviderFor<Voltage, Geometry2DType>::Delegate outVoltage;

    typename ProviderFor<CurrentDensity, Geometry2DType>::Delegate outCurrentDensity;

    typename ProviderFor<Heat, Geometry2DType>::Delegate outHeat;

    typename ProviderFor<Conductivity, Geometry2DType>::Delegate outConductivity;

    ReceiverFor<Temperature, Geometry2DType> inTemperature;

    Convergence convergence;    ///< Convergence method

    /**
     * Run electrical calculations
     * \return max correction of potential against the last call
     **/
    double compute(unsigned loops = 1);

    /**
     * Integrate vertical total current at certain level.
     * \param vindex vertical index of the element mesh to perform integration at
     * \param onlyactive if true only current in the active region is considered
     * \return computed total current
     */
    double integrateCurrent(size_t vindex, bool onlyactive = false);

    /**
     * Integrate vertical total current flowing vertically through active region.
     * \param nact number of the active region
     * \return computed total current
     */
    double getTotalCurrent(size_t nact = 0);

    /**
     * Compute total electrostatic energy stored in the structure.
     * \return total electrostatic energy [J]
     */
    double getTotalEnergy();

    /**
     * Estimate structure capacitance.
     * \return static structure capacitance [pF]
     */
    double getCapacitance();

    /**
     * Compute total heat generated by the structure in unit time
     * \return total generated heat [mW]
     */
    double getTotalHeat();

    /// Return the maximum estimated error.
    double getErr() const { return toterr; }

    /// Get p-contact layer conductivity [S/m]
    double getCondPcontact() const { return pcond; }
    /// Set p-contact layer conductivity [S/m]
    void setCondPcontact(double cond) {
        pcond = cond;
        this->invalidate();
    }

    /// Get n-contact layer conductivity [S/m]
    double getCondNcontact() const { return ncond; }
    /// Set n-contact layer conductivity [S/m]
    void setCondNcontact(double cond) {
        ncond = cond;
        this->invalidate();
    }

    /// Get data with junction effective conductivity
    DataVector<const Tensor2<double>> getCondJunc() const { return junction_conductivity; }
    /// Set junction effective conductivity to the single value
    void setCondJunc(double cond) {
        junction_conductivity.reset(max(junction_conductivity.size(), size_t(1)), cond);
        default_junction_conductivity = Tensor2<double>(0., cond);
    }
    /// Set junction effective conductivity to the single value
    void setCondJunc(Tensor2<double> cond) {
        junction_conductivity.reset(max(junction_conductivity.size(), size_t(1)), cond);
        default_junction_conductivity = cond;
    }
    /// Set junction effective conductivity to previously read data
    void setCondJunc(const DataVector<Tensor2<double>>& cond) {
        size_t condsize = 0;
        condsize = max(condsize, size_t(1));
        if (!this->mesh || cond.size() != condsize)
            throw BadInput(this->getId(), "Provided junction conductivity vector has wrong size");
        junction_conductivity = cond.claim();
    }

    void loadConfiguration(XMLReader& source, Manager& manager) override;

    void parseConfiguration(XMLReader& source, Manager& manager);

    ElectricalFem2DSolver(const std::string& name = "");

    ~ElectricalFem2DSolver();

  protected:
    const LazyData<double> getVoltage(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method) const;

    const LazyData<double> getHeatDensities(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);

    const LazyData<Vec<2>> getCurrentDensities(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);

    const LazyData<Tensor2<double>> getConductivity(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);
};

}}}  // namespace plask::electrical::shockley

#endif
