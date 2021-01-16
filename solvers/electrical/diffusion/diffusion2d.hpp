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
#ifndef PLASK__MODULE_ELECTRICAL_DIFFUSION2D_H
#define PLASK__MODULE_ELECTRICAL_DIFFUSION2D_H

#include <plask/common/fem.hpp>
#include <plask/plask.hpp>

namespace plask { namespace electrical { namespace diffusion {

#define DEFAULT_MESH_SPACING 0.01  // µm

struct ActiveRegion {
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
    double QWheight;

    shared_ptr<RectangularMesh<2>> mesh2d;
    std::vector<std::pair<double, double>> QWs;

    shared_ptr<RectangularMesh<2>> emesh;
    shared_ptr<RectangularMesh<2>> jmesh;

    DataVector<double> conc;
    DataVector<double> dconc;

    template <typename SolverT>
    ActiveRegion(const SolverT* solver,
                 size_t l,
                 size_t r,
                 size_t b,
                 size_t t,
                 double h,
                 std::vector<double> QWz,
                 std::vector<std::pair<size_t, size_t>> QWbt)
        : left(l),
          right(r),
          bottom(b),
          top(t),
          QWheight(h),
          mesh2d(
              new RectangularMesh<2>(shared_ptr<OrderedAxis>(new OrderedAxis()), shared_ptr<OrderedAxis>(new OrderedAxis(QWz)))) {
        auto begin = solver->getMesh()->axis[0]->begin();
        dynamic_pointer_cast<OrderedAxis>(mesh2d->axis[0])->addOrderedPoints(begin + left, begin + right + 1);

        QWs.reserve(QWbt.size());
        for (auto& bt : QWbt) QWs.emplace_back(solver->getMesh()->axis[1]->at(bt.first), solver->getMesh()->axis[1]->at(bt.second));

        shared_ptr<OnePointAxis> vert(new OnePointAxis(this->vert()));
        shared_ptr<OrderedAxis> mesh = this->mesh();
        emesh.reset(new RectangularMesh<2>(mesh->getMidpointAxis(), vert));
        jmesh.reset(new RectangularMesh<2>(mesh, vert));
    }

    shared_ptr<OrderedAxis> mesh() const { return dynamic_pointer_cast<OrderedAxis>(mesh2d->axis[0]); }

    double vert() const { return mesh2d->axis[1]->at((mesh2d->axis[1]->size() + 1) / 2 - 1); }
};

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template <typename Geometry2DType>
struct PLASK_SOLVER_API Diffusion2DSolver : public FemSolverWithMesh<Geometry2DType, RectangularMesh<2>> {
  protected:
    /// Data for active region
    int loopno;     ///< Number of completed loops
    double toterr;  ///< Maximum estimated error during all iterations

    struct ConcentrationDataImpl : public LazyDataImpl<double> {
        const Diffusion2DSolver* solver;
        shared_ptr<const MeshD<2>> destination_mesh;
        InterpolationFlags interpolationFlags;
        std::vector<LazyData<double>> concentrations;
        ConcentrationDataImpl(const Diffusion2DSolver* solver, shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod interp);
        double at(size_t i) const override;
        size_t size() const override { return destination_mesh->size(); }
    };

    std::vector<ActiveRegion> active;  ///< Active regions information

    /// Make stiffness matrix
    void setLocalMatrix(const double R,
                        const double L,
                        const double A,
                        const double B,
                        const double C,
                        const double D,
                        const double* U,
                        const double* J,
                        double& K00,
                        double& K01,
                        double& K02,
                        double& K03,
                        double& K11,
                        double& K12,
                        double& K13,
                        double& K22,
                        double& K23,
                        double& K33,
                        double& F0,
                        double& F1,
                        double& F2,
                        double& F3);

    /// Initialize the solver
    void onInitialize() override;

    /// Invalidate the data
    void onInvalidate() override;

    /// Get info on active region
    void setActiveRegions();

    /// Set stiffness matrix + load vector
    void setMatrix(FemMatrix& K,
                   DataVector<double>& F,
                   const DataVector<double>& U0,
                   const shared_ptr<OrderedAxis> mesh,
                   double z,
                   const LazyData<double>& temp,
                   const DataVector<double>& J);

    // void computeInitial(ActiveRegion& active);

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
    size_t isActive(const RectangularMesh2D::Element& element) const { return isActive(element.getMidpoint()); }

    struct PLASK_API From1DGenerator : public MeshGeneratorD<2> {
        shared_ptr<MeshGeneratorD<1>> generator;

        From1DGenerator(const shared_ptr<MeshGeneratorD<1>>& generator) : generator(generator) {}

        shared_ptr<MeshD<2>> generate(const shared_ptr<GeometryObjectD<2>>& geometry) override {
            auto simple_mesh = makeGeometryGrid(geometry);
            auto mesh1d = (*generator)(geometry);
            if (shared_ptr<MeshAxis> axis = dynamic_pointer_cast<MeshAxis>(mesh1d))
                return make_shared<RectangularMesh<2>>(axis, simple_mesh->axis[1]);
            throw BadInput("Generator1D", "1D mesh must be MeshAxis");
        }
    };

  public:
    double maxerr;  ///< Maximum relative current density correction accepted as convergence

    /// Boundary condition
    BoundaryConditions<RectangularMesh<2>::Boundary, double> voltage_boundary;

    ReceiverFor<CurrentDensity, Geometry2DType> inCurrentDensity;

    ReceiverFor<Temperature, Geometry2DType> inTemperature;

    ReceiverFor<Gain, Geometry2DType> inGain;

    ReceiverFor<ModeWavelength> inWavelength;

    ReceiverFor<ModeLightE, Geometry2DType> inLightE;

    typename ProviderFor<CarriersConcentration, Geometry2DType>::Delegate outCarriersConcentration;

    std::string getClassName() const override;

    /**
     * Run calculations for specified active region
     * \param loops maximum number of loops to run
     * \param act active region number to calculate
     * \return max correction of potential against the last call
     **/
    double compute(unsigned loops, size_t act);

    /**
     * Run calculations for all active regions
     * \param loops maximum number of loops to run
     * \return max correction of potential against the last call
     **/
    double compute(unsigned loops = 0) {
        this->initCalculation();
        double maxerr = 0.;
        for (size_t i = 0; i < active.size(); ++i) maxerr = max(maxerr, compute(loops, i));
        return maxerr;
    }

    void loadConfiguration(XMLReader& source, Manager& manager) override;

    void parseConfiguration(XMLReader& source, Manager& manager);

    Diffusion2DSolver(const std::string& name = "");

    ~Diffusion2DSolver();

    using SolverWithMesh<Geometry2DType, RectangularMesh<2>>::setMesh;

    void setMesh(shared_ptr<MeshD<1>> mesh) {
        auto simple_mesh = makeGeometryGrid(this->geometry);
        if (shared_ptr<MeshAxis> axis = dynamic_pointer_cast<MeshAxis>(mesh)) {
            shared_ptr<RectangularMesh<2>> mesh2d(new RectangularMesh<2>(axis, simple_mesh->axis[1]));
            this->setMesh(mesh2d);
        } else
            throw BadInput(this->getId(), "1D mesh must be MeshAxis");
    }

    void setMesh(shared_ptr<MeshGeneratorD<1>> generator) {
        this->setMesh(make_shared<From1DGenerator>(generator));
    }

    size_t activeRegionsCount() const { return active.size(); }

  protected:
    const LazyData<double> getConcentration(CarriersConcentration::EnumType what,
                                            shared_ptr<const MeshD<2>> dest_mesh,
                                            InterpolationMethod interpolation = INTERPOLATION_DEFAULT) const;
};

}}}  // namespace plask::electrical::diffusion

#endif
