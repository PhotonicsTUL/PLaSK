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
#ifndef PLASK__MODULE_ELECTRICAL_DIFFUSION3D_H
#define PLASK__MODULE_ELECTRICAL_DIFFUSION3D_H

#include <plask/common/fem.hpp>
#include <plask/plask.hpp>

namespace plask { namespace electrical { namespace diffusion {

#define DEFAULT_MESH_SPACING 0.005  // µm

struct ActiveRegion3D {
    struct Region {
        size_t bottom, top, left, right, back, front, lon, tra;
        bool warn;
        Region() {}
        Region(size_t b, size_t t, size_t x, size_t y)
            : bottom(b),
              top(t),
              left(std::numeric_limits<size_t>::max()),
              right(0),
              back(std::numeric_limits<size_t>::max()),
              front(0),
              lon(x),
              tra(y),
              warn(true) {}
    };

    size_t bottom, top, left, right, back, front;

    double QWheight;

    shared_ptr<RectangularMesh<3>> mesh3, emesh3, mesh2, emesh2;
    // shared_ptr<RectangularMesh<3>> fmesh3, fmesh2;
    std::vector<std::pair<double, double>> QWs;

    DataVector<double> U;

    std::vector<double> modesP;

    template <typename SolverT>
    ActiveRegion3D(const SolverT* solver,
                   size_t l0,
                   size_t l1,
                   size_t t0,
                   size_t t1,
                   size_t v0,
                   size_t v1,
                   double h,
                   std::vector<double> QWz,
                   std::vector<std::pair<size_t, size_t>> QWbt)
        : bottom(v0),
          top(v1),
          left(t0),
          right(t1),
          back(l0),
          front(l1),
          QWheight(h),
          mesh3(new RectangularMesh<3>(
              shared_ptr<OrderedAxis>(new OrderedAxis()),
              shared_ptr<OrderedAxis>(new OrderedAxis()),
              shared_ptr<OrderedAxis>(new OrderedAxis(QWz)),
              (right - left <= front - back) ? RectangularMesh<3>::ORDER_012 : RectangularMesh<3>::ORDER_102)) {
        auto lbegin = solver->getMesh()->lon()->begin();
        dynamic_pointer_cast<OrderedAxis>(mesh3->lon())->addOrderedPoints(lbegin + back, lbegin + front + 1);
        auto tbegin = solver->getMesh()->tran()->begin();
        dynamic_pointer_cast<OrderedAxis>(mesh3->tran())->addOrderedPoints(tbegin + left, tbegin + right + 1);

        QWs.reserve(QWbt.size());
        for (auto& bt : QWbt) QWs.emplace_back(solver->getMesh()->vert()->at(bt.first), solver->getMesh()->vert()->at(bt.second));

        emesh3.reset(new RectangularMesh<3>(mesh3->lon()->getMidpointAxis(), mesh3->tran()->getMidpointAxis(), mesh3->vert(),
                                            mesh3->getIterationOrder()));
        mesh2.reset(new RectangularMesh<3>(mesh3->lon(), mesh3->tran(), make_shared<OnePointAxis>(this->vert()),
                                           mesh3->getIterationOrder()));
        emesh2.reset(new RectangularMesh<3>(emesh3->lon(), emesh3->tran(), mesh2->vert(), mesh3->getIterationOrder()));
    }

    double vert() const { return mesh3->vert()->at((mesh3->vert()->size() + 1) / 2 - 1); }

    template <typename ReceiverType>
    LazyData<typename ReceiverType::ValueType> verticallyAverage(
        const ReceiverType& receiver,
        const shared_ptr<const RectangularMesh<3>>& mesh,
        InterpolationMethod interp = InterpolationMethod::INTERPOLATION_DEFAULT) const {
        assert(mesh->getIterationOrder() == RectangularMesh<3>::ORDER_012 ||
               mesh->getIterationOrder() == RectangularMesh<3>::ORDER_102);
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

struct ElementParams3D {
    const size_t n00, n01, n10, n11, i00, i01, i10, i02, i03, i12, i20, i21, i30, i22, i23, i32;
    const double X, Y;
    ElementParams3D(double X, double Y, size_t n00, size_t n01, size_t n10, size_t n11)
        : n00(n00),
          n01(n01),
          n10(n10),
          n11(n11),
          i00(3 * n00),
          i01(i00 + 1),
          i10(i00 + 2),
          i02(3 * n01),
          i03(i02 + 1),
          i12(i02 + 2),
          i20(3 * n10),
          i21(i20 + 1),
          i30(i20 + 2),
          i22(3 * n11),
          i23(i22 + 1),
          i32(i22 + 2),
          X(X),
          Y(Y) {}
    ElementParams3D(const ActiveRegion3D& active, size_t il, size_t it)
        : ElementParams3D(active.mesh2->lon()->at(il + 1) - active.mesh2->lon()->at(il),
                          active.mesh2->tran()->at(it + 1) - active.mesh2->tran()->at(it),
                          active.mesh2->index(il, it, 0),
                          active.mesh2->index(il, it + 1, 0),
                          active.mesh2->index(il + 1, it, 0),
                          active.mesh2->index(il + 1, it + 1, 0)) {}
    ElementParams3D(const ActiveRegion3D& active, size_t ie)
        : ElementParams3D(active, active.emesh2->index0(ie), active.emesh2->index1(ie)) {}
};

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
struct PLASK_SOLVER_API Diffusion3DSolver : public FemSolverWithMesh<Geometry3D, RectangularMesh<3>> {
  protected:
    /// Data for active region
    int loopno;     ///< Number of completed loops
    double toterr;  ///< Maximum estimated error during all iterations

    struct ConcentrationDataImpl : public LazyDataImpl<double> {
        const Diffusion3DSolver* solver;
        shared_ptr<const MeshD<3>> destination_mesh;
        InterpolationFlags interpolationFlags;
        std::vector<LazyData<double>> concentrations;
        ConcentrationDataImpl(const Diffusion3DSolver* solver, shared_ptr<const MeshD<3>> dest_mesh, InterpolationMethod interp);
        double at(size_t i) const override;
        size_t size() const override { return destination_mesh->size(); }
    };

    std::map<size_t, ActiveRegion3D> active;  ///< Active regions information

    /// Make local stiffness matrix and load vector
    inline void setLocalMatrix(FemMatrix& K,
                               DataVector<double>& F,
                               const ElementParams3D e,
                               const double A,
                               const double B,
                               const double C,
                               const double D,
                               const double* U,
                               const double* J);

    /// Add local stiffness matrix and load vector for SHB
    inline void addLocalBurningMatrix(FemMatrix& K,
                                      DataVector<double>& F,
                                      const ElementParams3D e,
                                      const Tensor2<double> G,
                                      const Tensor2<double> dG,
                                      const double Ug,
                                      const Tensor2<double>* P);

    /// Integrate bilinearly changing function over an element
    template <typename T> inline T integrateBilinear(const double Lx, const double Ly, const T* P) {
        return 0.25 * (P[0] + P[1] + P[2] + P[3]) * Lx * Lx;
    }

    /// Initialize the solver
    void onInitialize() override;

    /// Invalidate the data
    void onInvalidate() override;

    /// Get info on active region
    void setupActiveRegion3Ds();

    // void computeInitial(ActiveRegion3D& active);

    /** Return \c true if the specified point is at junction
     * \param point point to test
     * \returns number of active region + 1 (0 for none)
     */
    size_t isActive(const Vec<3>& point) const {
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
    size_t isActive(const RectangularMesh3D::Element& element) const { return isActive(element.getMidpoint()); }

  public:
    double maxerr;  ///< Maximum relative current density correction accepted as convergence

    /// Boundary condition
    BoundaryConditions<RectangularMesh<2>::Boundary, double> voltage_boundary;

    ReceiverFor<CurrentDensity, Geometry3D> inCurrentDensity;

    ReceiverFor<Temperature, Geometry3D> inTemperature;

    ReceiverFor<Gain, Geometry3D> inGain;

    ReceiverFor<ModeWavelength> inWavelength;

    ReceiverFor<ModeLightE, Geometry3D> inLightE;

    typename ProviderFor<CarriersConcentration, Geometry3D>::Delegate outCarriersConcentration;

    std::string getClassName() const override { return "electrical.Diffusion3D"; }

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
        for (const auto& act : active) maxerr = max(maxerr, compute(loops, shb, act.first));
        return maxerr;
    }

    void loadConfiguration(XMLReader& source, Manager& manager) override;

    void parseConfiguration(XMLReader& source, Manager& manager);

    Diffusion3DSolver(const std::string& name = "");

    ~Diffusion3DSolver();

    size_t activeRegionsCount() const { return active.size(); }

    double get_burning_integral_for_mode(size_t mode) const;

    double get_burning_integral() const;

  protected:
    const LazyData<double> getConcentration(CarriersConcentration::EnumType what,
                                            shared_ptr<const MeshD<3>> dest_mesh,
                                            InterpolationMethod interpolation = INTERPOLATION_DEFAULT) const;
};

}}}  // namespace plask::electrical::diffusion

#endif
