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
#include "freecarrier2d.hpp"
#include "fd.hpp"
#include "gauss_matrix.hpp"

namespace plask { namespace gain { namespace freecarrier {

constexpr double DIFF_STEP = 0.001;

template <typename GeometryT>
FreeCarrierGainSolver2D<GeometryT>::FreeCarrierGainSolver2D(const std::string& name)
    : FreeCarrierGainSolver<SolverWithMesh<GeometryT, MeshAxis>>(name) {}

struct Region2D {
    size_t left, right, bottom, top;
    size_t rowl, rowr;
    Region2D()
        : left(0),
          right(0),
          bottom(std::numeric_limits<size_t>::max()),
          top(std::numeric_limits<size_t>::max()),
          rowl(std::numeric_limits<size_t>::max()),
          rowr(0) {}
};

template <typename GeometryT> void FreeCarrierGainSolver2D<GeometryT>::detectActiveRegions() {
    shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(this->geometry->getChild());
    shared_ptr<RectangularMesh<2>> points = mesh->getElementMesh();

    std::vector<Region2D> regions;

    for (size_t r = 0; r < points->vert()->size(); ++r) {
        size_t prev = 0;
        shared_ptr<Material> material;
        for (size_t c = 0; c < points->tran()->size(); ++c) {  // In the (possible) active region
            auto point = points->at(c, r);
            size_t num = 0;

            auto roles = this->geometry->getRolesAt(point);
            for (const auto& role : roles) {
                if (role.substr(0, 6) == "active") {
                    if (num != 0) throw BadInput(this->getId(), "Multiple 'active' roles specified");
                    if (role.size() == 6) {
                        num = 1;
                    } else {
                        try {
                            num = boost::lexical_cast<size_t>(role.substr(6)) + 1;
                        } catch (boost::bad_lexical_cast&) {
                            throw BadInput(this->getId(), "Bad active region number in role '{0}'", role);
                        }
                    }
                } else if (role == "substrate") {
                    if (this->explicitSubstrate)
                        this->writelog(LOG_WARNING, "Explicit substrate layer specified, role 'substrate' ignored");
                    else {
                        if (!this->substrateMaterial)
                            this->substrateMaterial = this->geometry->getMaterial(point);
                        else if (*this->substrateMaterial != *this->geometry->getMaterial(point))
                            throw Exception("{0}: Non-uniform substrate layer.", this->getId());
                    }
                }
            }
            if (num == 0 && roles.find("QW") != roles.end())
                throw Exception("{0}: All marked quantum wells must belong to marked active region.", this->getId());

            if (num) {  // here we are inside the active region
                if (regions.size() >= num) {
                    if (!material)
                        material = this->geometry->getMaterial(points->at(c, r));
                    else if (*material != *this->geometry->getMaterial(points->at(c, r))) {
                        throw Exception("{0}: Active region {1} is laterally non-uniform", this->getId(), num - 1);
                    }
                }
                regions.resize(max(regions.size(), num));
                auto& reg = regions[num - 1];
                if (prev != num) {  // this region starts in the current row
                    if (reg.top < r) {
                        throw Exception("{0}: Active region {1} is disjoint", this->getId(), num - 1);
                    }
                    if (reg.bottom >= r)
                        reg.bottom = r;  // first row
                    else if (reg.rowr <= c)
                        throw Exception("{0}: Active region {1} is disjoint", this->getId(), num - 1);
                    reg.top = r + 1;
                    reg.rowl = c;
                    if (reg.left > reg.rowl) reg.left = reg.rowl;
                }
            }
            if (prev && prev != num) {  // previous region ended
                auto& reg = regions[prev - 1];
                if (reg.bottom < r && reg.rowl >= c) throw Exception("{0}: Active region {1} is disjoint", this->getId(), prev - 1);
                reg.rowr = c;
                if (reg.right < reg.rowr) reg.right = reg.rowr;
            }
            prev = num;
        }
        if (prev)  // junction reached the edge
            regions[prev - 1].rowr = regions[prev - 1].right = points->tran()->size();
    }

    this->regions.clear();
    size_t act = 0;
    for (auto& reg : regions) {
        if (reg.bottom == std::numeric_limits<size_t>::max()) {
            ++act;
            continue;
        }
        if (reg.bottom == 0)  //
            throw Exception("{0}: Active region cannot be located at the bottom of the structure.", this->getId());
        if (reg.top == points->axis[1]->size())
            throw Exception("{0}: Active region cannot be located at the top of the structure.", this->getId());
        this->regions.emplace_back(mesh->at(reg.left, reg.bottom - 1));
        auto region = &this->regions.back();
        region->bottom = mesh->axis[1]->at(1) - mesh->axis[1]->at(0);
        region->top = mesh->axis[1]->at(mesh->axis[1]->size() - 1) - mesh->axis[1]->at(mesh->axis[1]->size() - 2);
        double width = mesh->axis[0]->at(reg.right) - mesh->axis[0]->at(reg.left);
        for (size_t r = reg.bottom - 1, j = 0; r <= reg.top; ++r, ++j) {
            bool layerQW = false;
            for (size_t c = reg.left; c < reg.right; ++c) {
                shared_ptr<Material> material;
                auto point = points->at(c, r);
                double height = mesh->axis[1]->at(r + 1) - mesh->axis[1]->at(r);
                auto roles = this->geometry->getRolesAt(point);
                bool QW = roles.find("QW") != roles.end() /*  || roles.find("QD") != roles.end() */;
                if (c == reg.left) {
                    auto material = this->geometry->getMaterial(point);
                    size_t n = region->layers->getChildrenCount();
                    shared_ptr<Block<2>> last;
                    if (n > 0) {
                        last = static_pointer_cast<Block<2>>(
                            static_pointer_cast<Translation<2>>(region->layers->getChildNo(n - 1))->getChild());
                        assert(!last || last->size.c0 == width);
                        if (last && material == last->getRepresentativeMaterial() && QW == region->isQW(region->size() - 1)) {
                            // TODO check if usage of getRepresentativeMaterial is fine here (was material)
                            last->setSize(width, last->size.c1 + height);
                            continue;
                        }
                    }
                    auto layer = plask::make_shared<Block<2>>(Vec<2>(width, height), material);
                    if (QW) layer->addRole("QW");
                    region->layers->push_back(layer);
                    layerQW = QW;
                } else if (layerQW != QW)
                    throw Exception("{}: Quantum wells in active region {} are not consistent", this->getId(), act);
            }
        }
        ++act;
    }

    if (this->strained && !this->substrateMaterial)
        throw BadInput(this->getId(), "Strained quantum wells requested but no layer with substrate role set");

    this->writelog(LOG_DETAIL, "Found {0} active region{1}", this->regions.size(), (this->regions.size() == 1) ? "" : "s");
    for (auto& region : this->regions) region.summarize(this);
}

static const shared_ptr<OrderedAxis> zero_axis(new OrderedAxis({0.}));

/// Base for lazy data implementation
template <typename GeometryT> template <typename DT> struct FreeCarrierGainSolver2D<GeometryT>::DataBase : public LazyDataImpl<DT> {
    struct AveragedData {
        shared_ptr<const RectangularMesh<2>> mesh;
        LazyData<double> data;
        double factor;
        const FreeCarrierGainSolver2D<GeometryT>* solver;
        const char* name;

        AveragedData(const FreeCarrierGainSolver2D<GeometryT>* solver,
                     const char* name,
                     const shared_ptr<const MeshAxis>& haxis,
                     const ActiveRegionInfo& region)
            : solver(solver), name(name) {
            auto vaxis = plask::make_shared<OrderedAxis>();
            OrderedAxis::WarningOff vaxiswoff(vaxis);
            for (size_t n = 0; n != region.size(); ++n) {
                if (region.isQW(n)) {
                    auto box = region.getLayerBox(n);
                    vaxis->addPoint(0.5 * (box.lower.c1 + box.upper.c1));
                }
            }
            mesh = plask::make_shared<const RectangularMesh<2>>(const_pointer_cast<MeshAxis>(haxis), vaxis,
                                                                RectangularMesh<2>::ORDER_01);
            factor = 1. / double(vaxis->size());
        }

        size_t size() const { return mesh->axis[0]->size(); }

        double operator[](size_t i) const {
            double val = 0.;
            for (size_t j = 0; j != mesh->axis[1]->size(); ++j) {
                double v = data[mesh->index(i, j)];
                if (isnan(v)) throw ComputationError(solver->getId(), "Wrong {0} ({1}) at {2}", name, v, mesh->at(i, j));
                val += v;
            }
            return val * factor;
        }
    };

    typedef typename FreeCarrierGainSolver2D<GeometryT>::ActiveRegionParams ActiveRegionParams;

    FreeCarrierGainSolver2D<GeometryT>* solver;   ///< Solver
    std::vector<shared_ptr<MeshAxis>> regpoints;  ///< Points in each active region
    shared_ptr<const MeshD<2>> dest_mesh;         ///< Destination mesh
    InterpolationFlags interpolation_flags;       ///< Interpolation flags

    void setupFromAxis(const shared_ptr<MeshAxis>& axis) {
        regpoints.reserve(solver->regions.size());
        InterpolationFlags flags(solver->geometry);
        for (size_t r = 0; r != solver->regions.size(); ++r) {
            std::set<double> pts;
            auto box = solver->regions[r].getBoundingBox();
            double y = 0.5 * (box.lower.c1 + box.upper.c1);
            for (double x : *axis) {
                auto p = flags.wrap(vec(x, y));
                if (solver->regions[r].contains(p)) pts.insert(p.c0);
            }
            auto msh = plask::make_shared<OrderedAxis>();
            OrderedAxis::WarningOff mshw(msh);
            ;
            msh->addOrderedPoints(pts.begin(), pts.end(), pts.size());
            regpoints.emplace_back(std::move(msh));
        }
    }

    DataBase(FreeCarrierGainSolver2D<GeometryT>* solver, const shared_ptr<const MeshD<2>>& dst_mesh)
        : solver(solver), dest_mesh(dst_mesh), interpolation_flags(solver->geometry) {
        // Create horizontal points lists
        if (solver->mesh) {
            setupFromAxis(solver->mesh);
        } else if (auto rect_mesh = dynamic_pointer_cast<const RectangularMesh<2>>(dst_mesh)) {
            setupFromAxis(rect_mesh->axis[0]);
        } else {
            regpoints.reserve(solver->regions.size());
            InterpolationFlags flags(solver->geometry);
            for (size_t r = 0; r != solver->regions.size(); ++r) {
                std::set<double> pts;
                for (auto point : *dest_mesh) {
                    auto p = flags.wrap(point);
                    if (solver->regions[r].contains(p)) pts.insert(p.c0);
                }
                auto msh = plask::make_shared<OrderedAxis>();
                OrderedAxis::WarningOff mshw(msh);
                msh->addOrderedPoints(pts.begin(), pts.end(), pts.size());
                regpoints.emplace_back(std::move(msh));
            }
        }
    }

    size_t size() const override { return dest_mesh->size(); }
};

template <typename GeometryT>
struct FreeCarrierGainSolver2D<GeometryT>::ComputedData : public FreeCarrierGainSolver2D<GeometryT>::DataBaseTensor2 {
    using typename DataBaseTensor2::AveragedData;

    /// Computed interpolations in each active region
    std::vector<LazyData<Tensor2<double>>> data;

    template <typename... Args> ComputedData(Args... args) : DataBaseTensor2(args...) {}

    void compute(double wavelength, InterpolationMethod interp) {
        // Compute gains on mesh for each active region
        OmpLockGuard<OmpNestLock> lock(gain_omp_lock);
        this->data.resize(this->solver->regions.size());
        for (size_t reg = 0; reg != this->solver->regions.size(); ++reg) {
            if (this->regpoints[reg]->size() == 0) {
                this->data[reg] = LazyData<Tensor2<double>>(this->dest_mesh->size(), Tensor2<double>(0., 0.));
                continue;
            }
            AveragedData temps(this->solver, "temperature", this->regpoints[reg], this->solver->regions[reg]);
            AveragedData concs(temps);
            concs.name = "carriers concentration";
            temps.data = this->solver->inTemperature(temps.mesh, interp);
            concs.data = this->solver->inCarriersConcentration(CarriersConcentration::PAIRS, temps.mesh, interp);
            this->data[reg] =
                interpolate(plask::make_shared<RectangularMesh<2>>(this->regpoints[reg], zero_axis),
                            getValues(wavelength, interp, reg, concs, temps), this->dest_mesh, interp, this->interpolation_flags);
        }
    }

    virtual DataVector<Tensor2<double>> getValues(double wavelength,
                                                  InterpolationMethod interp,
                                                  size_t reg,
                                                  const AveragedData& concs,
                                                  const AveragedData& temps) = 0;

    Tensor2<double> at(size_t i) const override {
        for (size_t reg = 0; reg != this->solver->regions.size(); ++reg)
            if (this->solver->regions[reg].inQW(this->interpolation_flags.wrap(this->dest_mesh->at(i)))) return this->data[reg][i];
        return Tensor2<double>(0., 0.);
    }
};

template <typename GeometryT>
struct FreeCarrierGainSolver2D<GeometryT>::GainData : public FreeCarrierGainSolver2D<GeometryT>::ComputedData {
    using typename DataBaseTensor2::AveragedData;

    template <typename... Args> GainData(Args... args) : ComputedData(args...) {}

    DataVector<Tensor2<double>> getValues(double wavelength,
                                          InterpolationMethod interp,
                                          size_t reg,
                                          const AveragedData& concs,
                                          const AveragedData& temps) override {
        double hw = phys::h_eVc1e9 / wavelength;
        DataVector<Tensor2<double>> values(this->regpoints[reg]->size());
        std::exception_ptr error;

        if (this->solver->inFermiLevels.hasProvider()) {
            AveragedData Fcs(temps);
            Fcs.name = "quasi Fermi level for electrons";
            AveragedData Fvs(temps);
            Fvs.name = "quasi Fermi level for holes";
            Fcs.data = this->solver->inFermiLevels(FermiLevels::ELECTRONS, temps.mesh, interp);
            Fvs.data = this->solver->inFermiLevels(FermiLevels::HOLES, temps.mesh, interp);
            plask::openmp_size_t end = this->regpoints[reg]->size();
#pragma omp parallel for
            for (plask::openmp_size_t i = 0; i < end; ++i) {
                if (error) continue;
                try {
                    double T = temps[i];
                    double conc = max(concs[i], 1e-6);  // To avoid hangs
                    double nr = this->solver->regions[reg].averageNr(wavelength, T, conc);
                    ActiveRegionParams params(this->solver, this->solver->params0[reg], T, bool(i));
                    values[i] = this->solver->getGain(hw, Fcs[i], Fvs[i], T, nr, params);
                } catch (...) {
#pragma omp critical
                    error = std::current_exception();
                }
            }
            if (error) std::rethrow_exception(error);
        } else {
            plask::openmp_size_t end = this->regpoints[reg]->size();
#pragma omp parallel for
            for (plask::openmp_size_t i = 0; i < end; ++i) {
                if (error) continue;
                try {
                    double T = temps[i];
                    double conc = max(concs[i], 1e-6);  // To avoid hangs
                    double nr = this->solver->regions[reg].averageNr(wavelength, T, conc);
                    ActiveRegionParams params(this->solver, this->solver->params0[reg], T, bool(i));
                    double Fc = NAN, Fv = NAN;
                    this->solver->findFermiLevels(Fc, Fv, conc, T, params);
                    values[i] = this->solver->getGain(hw, Fc, Fv, T, nr, params);
                } catch (...) {
#pragma omp critical
                    error = std::current_exception();
                }
            }
            if (error) std::rethrow_exception(error);
        }
        return values;
    }
};

template <typename GeometryT>
struct FreeCarrierGainSolver2D<GeometryT>::DgdnData : public FreeCarrierGainSolver2D<GeometryT>::ComputedData {
    using typename DataBaseTensor2::AveragedData;

    template <typename... Args> DgdnData(Args... args) : ComputedData(args...) {}

    DataVector<Tensor2<double>> getValues(double wavelength,
                                          InterpolationMethod /*interp*/,
                                          size_t reg,
                                          const AveragedData& concs,
                                          const AveragedData& temps) override {
        double hw = phys::h_eVc1e9 / wavelength;
        const double h = 0.5 * DIFF_STEP;
        DataVector<Tensor2<double>> values(this->regpoints[reg]->size());
        std::exception_ptr error;
        plask::openmp_size_t end = this->regpoints[reg]->size();
#pragma omp parallel for
        for (plask::openmp_size_t i = 0; i < end; ++i) {
            if (error) continue;
            try {
                double T = temps[i];
                double conc = max(concs[i], 1e-6);  // To avoid hangs
                double nr = this->solver->regions[reg].averageNr(wavelength, T, conc);
                ActiveRegionParams params(this->solver, this->solver->params0[reg], T, bool(i));
                double Fc = NAN, Fv = NAN;
                this->solver->findFermiLevels(Fc, Fv, (1. - h) * conc, T, params);
                Tensor2<double> gain1 = this->solver->getGain(hw, Fc, Fv, T, nr, params);
                this->solver->findFermiLevels(Fc, Fv, (1. + h) * conc, T, params);
                Tensor2<double> gain2 = this->solver->getGain(hw, Fc, Fv, T, nr, params);
                values[i] = (gain2 - gain1) / (2. * h * conc);
            } catch (...) {
#pragma omp critical
                error = std::current_exception();
            }
        }
        if (error) std::rethrow_exception(error);
        return values;
    }
};

template <typename GeometryT>
const LazyData<Tensor2<double>> FreeCarrierGainSolver2D<GeometryT>::getGainData(Gain::EnumType what,
                                                                                const shared_ptr<const MeshD<2>>& dst_mesh,
                                                                                double wavelength,
                                                                                InterpolationMethod interp) {
    if (what == Gain::GAIN) {
        this->initCalculation();  // This must be called before any calculation!
        this->writelog(LOG_DETAIL, "Calculating gain");
        GainData* data = new GainData(this, dst_mesh);
        data->compute(wavelength, getInterpolationMethod<INTERPOLATION_SPLINE>(interp));
        return LazyData<Tensor2<double>>(data);
    } else if (what == Gain::DGDN) {
        this->initCalculation();  // This must be called before any calculation!
        this->writelog(LOG_DETAIL, "Calculating gain over carriers concentration derivative");
        DgdnData* data = new DgdnData(this, dst_mesh);
        data->compute(wavelength, getInterpolationMethod<INTERPOLATION_SPLINE>(interp));
        return LazyData<Tensor2<double>>(data);
    } else {
        throw BadInput(this->getId(), "Wrong gain type requested");
    }
}

template <typename GeometryT>
struct FreeCarrierGainSolver2D<GeometryT>::EnergyLevelsData
    : public FreeCarrierGainSolver2D<GeometryT>::DataBaseVector {
    using typename DataBaseVector::AveragedData;

    size_t which;
    std::vector<LazyData<double>> temps;

    EnergyLevelsData(EnergyLevels::EnumType which,
                     FreeCarrierGainSolver2D<GeometryT>* solver,
                     const shared_ptr<const MeshD<2>>& dst_mesh,
                     InterpolationMethod interp)
        : DataBaseVector(solver, dst_mesh), which(size_t(which)) {
        temps.reserve(solver->regions.size());
        for (size_t reg = 0; reg != solver->regions.size(); ++reg) {
            AveragedData temp(this->solver, "temperature", this->regpoints[reg], this->solver->regions[reg]);
            temp.data = this->solver->inTemperature(temp.mesh, interp);
            temps.emplace_back(
                interpolate(temp.mesh, DataVector<const double>(temp.data), this->dest_mesh, interp, this->solver->geometry));
        }
    }

    std::vector<double> at(size_t i) const override {
        for (size_t reg = 0; reg != this->solver->regions.size(); ++reg)
            if (this->solver->regions[reg].contains(this->interpolation_flags.wrap(this->dest_mesh->at(i)))) {
                double T = temps[reg][i];
                ActiveRegionParams params(this->solver, this->solver->params0[reg], T, bool(i));
                std::vector<double> result;
                result.reserve(params.levels[which].size());
                for (const auto& level : params.levels[which]) result.push_back(level.E);
                return result;
            }
        return std::vector<double>();
    }
};

template <typename GeometryT>
const LazyData<std::vector<double>> FreeCarrierGainSolver2D<GeometryT>::getEnergyLevels(EnergyLevels::EnumType which,
                                                                                        const shared_ptr<const MeshD<2>>& dst_mesh,
                                                                                        InterpolationMethod interp) {
    this->initCalculation();
    EnergyLevelsData* data = new EnergyLevelsData(which, this, dst_mesh, getInterpolationMethod<INTERPOLATION_LINEAR>(interp));
    return LazyData<std::vector<double>>(data);
}

template <> std::string FreeCarrierGainSolver2D<Geometry2DCartesian>::getClassName() const { return "gain.FreeCarrier2D"; }
template <> std::string FreeCarrierGainSolver2D<Geometry2DCylindrical>::getClassName() const { return "gain.FreeCarrierCyl"; }

template struct PLASK_SOLVER_API FreeCarrierGainSolver2D<Geometry2DCartesian>;
template struct PLASK_SOLVER_API FreeCarrierGainSolver2D<Geometry2DCylindrical>;

}}}  // namespace plask::gain::freecarrier
