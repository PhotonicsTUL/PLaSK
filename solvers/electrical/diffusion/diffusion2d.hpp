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

struct ActiveRegion2D {
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

    shared_ptr<RectangularMesh<2>> mesh2, emesh2, mesh1, emesh1;
    std::vector<std::pair<double, double>> QWs;

    DataVector<double> U;

    std::vector<double> modesP;

    template <typename SolverT>
    ActiveRegion2D(const SolverT* solver,
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
          mesh2(new RectangularMesh<2>(shared_ptr<OrderedAxis>(new OrderedAxis()),
                                       shared_ptr<OrderedAxis>(new OrderedAxis(QWz)),
                                       RectangularMesh<2>::ORDER_01)) {
        auto begin = solver->getMesh()->tran()->begin();
        dynamic_pointer_cast<OrderedAxis>(mesh2->tran())->addOrderedPoints(begin + left, begin + right + 1);

        QWs.reserve(QWbt.size());
        for (auto& bt : QWbt) QWs.emplace_back(solver->getMesh()->vert()->at(bt.first), solver->getMesh()->vert()->at(bt.second));

        shared_ptr<OrderedAxis> mesh = this->mesh();
        emesh2.reset(new RectangularMesh<2>(mesh->getMidpointAxis(), mesh2->vert(), RectangularMesh<2>::ORDER_01));
        mesh1.reset(new RectangularMesh<2>(mesh, make_shared<OnePointAxis>(this->vert())));
        emesh1.reset(new RectangularMesh<2>(emesh2->tran(), mesh1->vert()));

        // OrderedAxis faxis0;
        // faxis0.addOrderedPoints(mesh->begin(), mesh->end(), mesh->size(), 0.);
        // faxis0.addOrderedPoints(emesh1->tran()->begin(), emesh1->tran()->end(), emesh1->tran()->size(), 0.);
        // fmesh1.reset(new RectangularMesh<2>(faxis0.getMidpointAxis(), mesh1->vert(), RectangularMesh<2>::ORDER_01));
        // fmesh1.reset(new RectangularMesh<2>(fmesh1->tran(), emesh1->vert()));
        // assert(fmesh1->size() == 2 * emesh1->size());
    }

    shared_ptr<OrderedAxis> mesh() const {
        assert(dynamic_pointer_cast<OrderedAxis>(mesh2->tran()));
        return static_pointer_cast<OrderedAxis>(mesh2->tran());
    }

    double vert() const { return mesh2->vert()->at((mesh2->vert()->size() + 1) / 2 - 1); }

    template <typename ReceiverType>
    LazyData<typename ReceiverType::ValueType> verticallyAverage(
        const ReceiverType& receiver,
        const shared_ptr<const RectangularMesh<2>>& mesh,
        InterpolationMethod interp = InterpolationMethod::INTERPOLATION_DEFAULT) const {
        assert(mesh->getIterationOrder() == RectangularMesh<2>::ORDER_01);
        auto data = receiver(mesh, interp);
        const size_t n = mesh->vert()->size();
        return LazyData<typename ReceiverType::ValueType>(
            mesh->tran()->size(), [this, data, n](size_t i) -> typename ReceiverType::ValueType {
                typename ReceiverType::ValueType val(Zero<typename ReceiverType::ValueType>());
                for (size_t j = n * i, end = n * (i + 1); j < end; ++j) val += data[j];
                return val / n;
            });
    }
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

    std::map<size_t, ActiveRegion2D> active;  ///< Active regions information

    /// Make local stiffness matrix and load vector
    inline void setLocalMatrix(const double R,
                               const double L,
                               const double L2,
                               const double L3,
                               const double L4,
                               const double L5,
                               const double L6,
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

    /// Add local stiffness matrix and load vector for SHB
    inline void addLocalBurningMatrix(const double R,
                                      const double L,
                                      const double L2,
                                      const double L3,
                                      const double* P,
                                      const double* g,
                                      const double* dg,
                                      const double ug,
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

    /// Integrate linearly changing function over an element
    template <typename T> inline T integrateLinear(const double R, const double L, const T* P);

    /// Initialize the solver
    void onInitialize() override;

    /// Invalidate the data
    void onInvalidate() override;

    /// Get info on active region
    void setupActiveRegion2Ds();

    // void computeInitial(ActiveRegion2D& active);

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

    struct PLASK_SOLVER_API From1DGenerator : public MeshGeneratorD<2> {
        shared_ptr<MeshGeneratorD<1>> generator;

        From1DGenerator(const shared_ptr<MeshGeneratorD<1>>& generator) : generator(generator) {}

        shared_ptr<MeshD<2>> generate(const shared_ptr<GeometryObjectD<2>>& geometry) override {
            auto simple_mesh = makeGeometryGrid(geometry);
            auto mesh1d = (*generator)(geometry);
            if (shared_ptr<MeshAxis> axis = dynamic_pointer_cast<MeshAxis>(mesh1d))
                return make_shared<RectangularMesh<2>>(axis, simple_mesh->vert());
            throw BadInput("generator1D", "1D mesh must be MeshAxis");
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
     * \param shb \c true if spatial hole burning should be taken into account
     * \return max correction of potential against the last call
     **/
    double compute(unsigned loops, bool shb, size_t act);

    /**
     * Run calculations for all active regions
     * \param loops maximum number of loops to run
     * \param shb \c true if spatial hole burning should be taken into account
     * \return max correction of potential against the last call
     **/
    double compute(unsigned loops = 0, bool shb = false) {
        this->initCalculation();
        double maxerr = 0.;
        for (const auto& act: active) maxerr = max(maxerr, compute(loops, shb, act.first));
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
            shared_ptr<RectangularMesh<2>> mesh2d(new RectangularMesh<2>(axis, simple_mesh->vert()));
            this->setMesh(mesh2d);
        } else
            throw BadInput(this->getId(), "1D mesh must be MeshAxis");
    }

    void setMesh(shared_ptr<MeshGeneratorD<1>> generator) { this->setMesh(make_shared<From1DGenerator>(generator)); }

    size_t activeRegionsCount() const { return active.size(); }

    double get_burning_integral_for_mode(size_t mode) const;

    double get_burning_integral() const;

  protected:
    const LazyData<double> getConcentration(CarriersConcentration::EnumType what,
                                            shared_ptr<const MeshD<2>> dest_mesh,
                                            InterpolationMethod interpolation = INTERPOLATION_DEFAULT) const;
};

}}}  // namespace plask::electrical::diffusion

#endif
