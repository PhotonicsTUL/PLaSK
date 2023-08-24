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

template <typename T> struct LateralMesh3D : MeshD<3> {
    shared_ptr<T> lateral;
    double z;

    LateralMesh3D(const shared_ptr<T>& lateral, double z) : lateral(lateral), z(z) {}

    std::size_t size() const override { return lateral->size(); }

    plask::Vec<3> at(std::size_t index) const override {
        Vec<2> p = lateral->at(index);
        return Vec<3>(p.c0, p.c1, z);
    }

    shared_ptr<LateralMesh3D<typename T::ElementMesh>> getElementMesh() const {
        return make_shared<LateralMesh3D<typename T::ElementMesh>>(lateral->getElementMesh(), z);
    }
};

template <typename T> struct ExtendedLateralMesh3D : MeshD<3> {
    shared_ptr<T> lateral;
    std::vector<double> zz;

    ExtendedLateralMesh3D(const shared_ptr<T>& lateral, std::vector<double> zz) : lateral(lateral), zz(std::move(zz)) {}

    std::size_t size() const override { return lateral->size() * zz.size(); }

    plask::Vec<3> at(std::size_t index) const override {
        size_t i = index / zz.size(), j = index % zz.size();
        Vec<2> p = lateral->at(i);
        return Vec<3>(p.c0, p.c1, zz[j]);
    }

    shared_ptr<ExtendedLateralMesh3D<typename T::ElementMesh>> getElementMesh() const {
        return shared_ptr<ExtendedLateralMesh3D<typename T::ElementMesh>>(
            new ExtendedLateralMesh3D<typename T::ElementMesh>(lateral->getElementMesh(), zz));
    }
};

struct ActiveRegion3D {
    struct Region {
        size_t bottom, top, lon, tra;
        bool warn;
        std::vector<bool> isQW;
        Region() {}
        Region(size_t b, size_t t, size_t x, size_t y, const std::vector<bool>& isQW)
            : bottom(b), top(t), lon(x), tra(y), warn(true), isQW(isQW) {}
    };

    size_t bottom, top;

    double QWheight;

    shared_ptr<LateralMesh3D<RectangularMaskedMesh2D>> mesh2;
    shared_ptr<LateralMesh3D<RectangularMaskedMesh2D::ElementMesh>> emesh2;
    shared_ptr<ExtendedLateralMesh3D<RectangularMaskedMesh2D>> mesh3;
    shared_ptr<ExtendedLateralMesh3D<RectangularMaskedMesh2D::ElementMesh>> emesh3;

    std::vector<std::pair<double, double>> QWs;

    DataVector<double> U;

    std::vector<double> modesP;

    template <typename SolverT>
    ActiveRegion3D(const SolverT* solver,
                   size_t bottom,
                   size_t top,
                   double h,
                   std::vector<double> QWz,
                   std::vector<std::pair<size_t, size_t>> QWbt)
        : bottom(bottom), top(top), QWheight(h) {
        QWs.reserve(QWbt.size());
        for (auto& bt : QWbt) QWs.emplace_back(solver->getMesh()->vert()->at(bt.first), solver->getMesh()->vert()->at(bt.second));

        double z = QWz[(QWz.size() + 1) / 2 - 1];
        auto lateral_mesh_unmasked = make_shared<RectangularMesh2D>(solver->getMesh()->lon(), solver->getMesh()->tran());
        auto lateral_mesh = make_shared<RectangularMaskedMesh2D>(
            *lateral_mesh_unmasked, [solver, z](const RectangularMesh2D::Element& element) -> bool {
                auto point = element.getMidpoint();
                auto roles = solver->getGeometry()->getRolesAt(Vec<3>(point.c0, point.c1, z));
                return roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("carriers") != roles.end();
            });

        mesh2 = make_shared<LateralMesh3D<RectangularMaskedMesh2D>>(lateral_mesh, z);
        emesh2 = mesh2->getElementMesh();
        // mesh3 = make_shared<ExtendedLateralMesh3D<RectangularMaskedMesh2D>>(lateral_mesh, std::move(QWz));
        mesh3.reset(new ExtendedLateralMesh3D<RectangularMaskedMesh2D>(lateral_mesh, std::move(QWz)));
        emesh3 = mesh3->getElementMesh();
    }

    double vert() const { return mesh2->z; }

    template <typename ReceiverType, typename MeshType>
    LazyData<typename ReceiverType::ValueType> verticallyAverage(
        const ReceiverType& receiver,
        const shared_ptr<ExtendedLateralMesh3D<MeshType>>& mesh,
        InterpolationMethod interp = InterpolationMethod::INTERPOLATION_DEFAULT) const {
        auto data = receiver(mesh, interp);
        const size_t n = mesh->zz.size();
        return LazyData<typename ReceiverType::ValueType>(
            mesh->lateral->size(), [this, data, n](size_t i) -> typename ReceiverType::ValueType {
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
    ElementParams3D(const RectangularMaskedMesh2D::Element& element)
        : ElementParams3D(element.getSize0(),
                          element.getSize1(),
                          element.getLoLoIndex(),
                          element.getLoUpIndex(),
                          element.getUpLoIndex(),
                          element.getUpUpIndex()) {}
    ElementParams3D(const ActiveRegion3D& active, size_t ie) : ElementParams3D(active.mesh2->lateral->element(ie)) {}
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
    void setupActiveRegions();

    // void computeInitial(ActiveRegion3D& active);

  private:
    void summarizeActiveRegion(std::map<size_t, ActiveRegion3D::Region>& regions,
                               size_t num,
                               size_t start,
                               size_t ver,
                               size_t lon,
                               size_t tra,
                               const std::vector<bool>& isQW,
                               const shared_ptr<RectangularMesh3D::ElementMesh>& points);

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
