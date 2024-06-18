/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (t) 2022 Lodz University of Technology
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
#include "freecarrier3d.hpp"
#include "fd.hpp"
#include "gauss_matrix.hpp"

namespace plask { namespace gain { namespace freecarrier {

constexpr double DIFF_STEP = 0.001;

FreeCarrierGainSolver3D::FreeCarrierGainSolver3D(const std::string& name) : FreeCarrierGainSolver<SolverOver<Geometry3D>>(name) {}

void FreeCarrierGainSolver3D::detectActiveRegions() {
    shared_ptr<RectangularMesh<3>> mesh = makeGeometryGrid(this->geometry->getChild());
    shared_ptr<RectangularMesh<3>::ElementMesh> points = mesh->getElementMesh();

    std::map<size_t, FreeCarrierGainSolver3D::Region> regions;
    std::vector<bool> isQW(points->axis[2]->size());
    std::vector<shared_ptr<Material>> materials(points->axis[2]->size());

    for (size_t lon = 0; lon < points->axis[0]->size(); ++lon) {
        for (size_t tra = 0; tra < points->axis[1]->size(); ++tra) {
            std::fill(isQW.begin(), isQW.end(), false);
            std::fill(materials.begin(), materials.end(), shared_ptr<Material>());
            size_t num = 0;
            size_t start = 0;
            for (size_t ver = 0; ver < points->axis[2]->size(); ++ver) {
                auto point = points->at(lon, tra, ver);
                shared_ptr<Material> material = this->geometry->getMaterial(point);
                materials[ver] = material;
                auto roles = this->geometry->getRolesAt(point);
                size_t cur = 0;
                for (auto role : roles) {  // find the active region, point belongs to
                    if (role.substr(0, 6) == "active") {
                        if (cur != 0) throw BadInput(this->getId(), "multiple 'active' roles specified");
                        if (role.size() == 6) {
                            cur = 1;
                        } else {
                            try {
                                cur = boost::lexical_cast<size_t>(role.substr(6)) + 1;
                            } catch (boost::bad_lexical_cast&) {
                                throw BadInput(this->getId(), "bad active region number in role '{0}'", role);
                            }
                        }
                    } else if (role == "substrate") {
                        if (this->explicitSubstrate)
                            this->writelog(LOG_WARNING, "Explicit substrate layer specified, role 'substrate' ignored");
                        else {
                            if (!this->substrateMaterial)
                                this->substrateMaterial = material;
                            else if (*this->substrateMaterial != *material)
                                throw Exception("{0}: Non-uniform substrate layer.", this->getId());
                        }
                    }
                }
                if (cur == 0 && roles.find("QW") != roles.end())
                    throw Exception("{0}: All marked quantum wells must belong to marked active region.", this->getId());
                if (cur != num) {
                    if (cur * num != 0)
                        throw Exception("{0}: Different active regions {1} and {2} may not be directly adjacent", this->getId(),
                                        num - 1, cur - 1);
                    if (num) {
                        auto found = regions.find(num);
                        FreeCarrierGainSolver3D::Region& region =
                            (found == regions.end()) ? regions
                                                           .emplace(std::piecewise_construct, std::forward_as_tuple(num),
                                                                    std::forward_as_tuple(start, ver, lon, tra, isQW, materials))
                                                           .first->second
                                                     : found->second;

                        if (start != region.bottom || ver != region.top)
                            throw Exception("{0}: Active region {1} does not have top and bottom edges at constant heights",
                                            this->getId(), num - 1);

                        for (size_t i = region.bottom; i < region.top; ++i) {
                            if (isQW[i] != region.isQW[i])
                                throw Exception("{0}: Active region {1} does not have QWs at constant heights", this->getId(),
                                                num - 1);
                        }

                        if (found != regions.end() && lon != region.lon && tra != region.tra &&
                            *region.materials.back() != *material)
                            throw Exception("{0}: Active region {1} has non-uniform top cladding", this->getId(), num - 1);
                    }
                    num = cur;
                    start = ver;
                }
                if (cur) {
                    isQW[ver] = roles.find("QW") != roles.end() /* || roles.find("QD") != roles.end() */;
                    auto found = regions.find(cur);
                    if (found != regions.end()) {
                        FreeCarrierGainSolver3D::Region& region = found->second;
                        if (lon != region.lon && tra != region.tra) {
                            if (*materials[ver] != *region.materials[ver - region.bottom + 1])
                                throw Exception("{0}: Active region {1} is laterally non-uniform", this->getId(), num - 1);
                            if (ver == region.bottom && *materials[ver - 1] != *region.materials[0])
                                throw Exception("{0}: Active region {1} has non-uniform bottom cladding", this->getId(), num - 1);
                        }
                    }
                }
            }
            // Summarize the last region
            if (num) throw Exception("{0}: Active region cannot be located at the top of the structure.", this->getId());
        }
    }

    this->regions.clear();
    for (auto& ireg : regions) {
        size_t act = ireg.first - 1;
        FreeCarrierGainSolver3D::Region& reg = ireg.second;
        // Detect quantum wells in the active region
        if (reg.bottom == 0)  //
            throw Exception("{0}: Active region cannot be located at the bottom of the structure.", this->getId());
        if (reg.top == points->axis[2]->size())
            throw Exception("{0}: Active region cannot be located at the top of the structure.", this->getId());
        this->regions.emplace_back(mesh->at(0, 0, reg.bottom - 1));
        auto region = &this->regions.back();
        region->bottom = mesh->axis[2]->at(reg.bottom) - mesh->axis[2]->at(reg.bottom - 1);
        region->top = mesh->axis[2]->at(reg.top + 1) - mesh->axis[2]->at(reg.top);
        double dx = mesh->axis[0]->at(mesh->axis[0]->size() - 1) - mesh->axis[0]->at(0);
        double dy = mesh->axis[1]->at(mesh->axis[1]->size() - 1) - mesh->axis[1]->at(0);
        for (size_t i = reg.bottom - 1, j = 0; i <= reg.top; ++i, ++j) {
            bool QW = reg.isQW[i];
            auto material = reg.materials[j];
            double dz = mesh->axis[2]->at(i + 1) - mesh->axis[2]->at(i);
            size_t n = region->layers->getChildrenCount();
            shared_ptr<Block<3>> last;
            if (n > 0) {
                last = static_pointer_cast<Block<3>>(
                    static_pointer_cast<Translation<3>>(region->layers->getChildNo(n - 1))->getChild());
                assert(!last || (last->size.c0 == dx && last->size.c1 == dy));
                if (last && material == last->getRepresentativeMaterial() && QW == region->isQW(region->size() - 1)) {
                    // TODO check if usage of getRepresentativeMaterial is fine here (was material)
                    last->setSize(dx, dy, last->size.c1 + dz);
                    continue;
                }
            }
            auto layer = plask::make_shared<Block<3>>(Vec<3>(dx, dy, dz), material);
            if (QW) layer->addRole("QW");
            region->layers->push_back(layer);
        }
    }

    if (this->strained && !this->substrateMaterial)
        throw BadInput(this->getId(), "strained quantum wells requested but no layer with substrate role set");

    this->writelog(LOG_DETAIL, "Found {0} active region{1}", this->regions.size(), (this->regions.size() == 1) ? "" : "s");
    for (auto& region : this->regions) region.summarize(this);
}

struct ActiveRegionMesh : MeshD<2> {
    const MeshD<3>* original_mesh;
    const CompressedSetOfNumbers<>& indices;

    template <typename DT>
    ActiveRegionMesh(const FreeCarrierGainSolver3D::DataBase<DT>* parent, size_t reg)
        : original_mesh(parent->dest_mesh.get()), indices(parent->regions[reg]) {}

    size_t size() const override { return indices.size(); }

    Vec<2> at(size_t i) const override {
        auto p = original_mesh->at(indices.at(i));
        return Vec<2>(p.c0, p.c1);
    }
};

/// Base for lazy data implementation
template <typename DT> struct FreeCarrierGainSolver3D::DataBase : public LazyDataImpl<DT> {
    //
    struct AveragedData {
        shared_ptr<MultiLateralMesh3D<MeshD<2>>> mesh;
        LazyData<double> data;
        double factor;
        const FreeCarrierGainSolver3D* solver;
        const char* name;

        AveragedData(const FreeCarrierGainSolver3D* solver,
                     const char* name,
                     const shared_ptr<MeshD<2>>& lateral,
                     const ActiveRegionInfo& region)
            : solver(solver), name(name) {
            auto vaxis = plask::make_shared<OrderedAxis>();
            OrderedAxis::WarningOff vaxiswoff(vaxis);
            for (size_t n = 0; n != region.size(); ++n) {
                if (region.isQW(n)) {
                    auto box = region.getLayerBox(n);
                    vaxis->addPoint(0.5 * (box.lower.c2 + box.upper.c2));
                }
            }
            mesh = plask::make_shared<MultiLateralMesh3D<MeshD<2>>>(lateral, vaxis);
            factor = 1. / double(vaxis->size());
        }

        size_t size() const { return mesh->lateralMesh->size(); }

        double operator[](size_t i) const {
            double val = 0.;
            size_t offset = i * mesh->vertAxis->size();
            for (size_t j = 0; j != mesh->vertAxis->size(); ++j) {
                double v = data[offset + j];
                if (isnan(v)) throw ComputationError(solver->getId(), "wrong {0} ({1}) at {2}", name, v, mesh->at(offset + j));
                val += v;
            }
            return val * factor;
        }
    };

    typedef FreeCarrierGainSolver3D::ActiveRegionParams ActiveRegionParams;

    FreeCarrierGainSolver3D* solver;                ///< Solver
    shared_ptr<const MeshD<3>> dest_mesh;           ///< Destination mesh
    InterpolationFlags interpolation_flags;         ///< Interpolation flags
    std::vector<CompressedSetOfNumbers<>> regions;  ///< Map from destination mesh to active region

    DataBase(FreeCarrierGainSolver3D* solver, const shared_ptr<const MeshD<3>>& dst_mesh)
        : solver(solver), dest_mesh(dst_mesh), interpolation_flags(solver->geometry), regions(solver->regions.size()) {
        InterpolationFlags flags(solver->geometry);
        std::map<std::string, size_t> region_roles;
        region_roles["active"] = 0;
        for (size_t a = 0; a != solver->regions.size(); ++a) {
            region_roles["active" + boost::lexical_cast<std::string>(a)] = a;
        }
        for (size_t i = 0; i < dst_mesh->size(); ++i) {
            auto point = dst_mesh->at(i);
            auto p = flags.wrap(point);
            auto roles = this->solver->geometry->getRolesAt(p);
            for (auto role : roles) {
                if (region_roles.find(role) != region_roles.end()) {
                    size_t reg = region_roles[role];
                    regions[reg].push_back(i);
                    break;
                }
            }
        }
        for (auto& region : regions) region.shrink_to_fit();
    }

    size_t size() const override { return dest_mesh->size(); }
};

struct FreeCarrierGainSolver3D::ComputedData : public FreeCarrierGainSolver3D::DataBaseTensor2 {
    using typename DataBaseTensor2::AveragedData;

    /// Computed interpolations in each active region
    std::vector<DataVector<Tensor2<double>>> data;

    ComputedData(FreeCarrierGainSolver3D* solver, const shared_ptr<const MeshD<3>>& dst_mesh)
        : DataBaseTensor2(solver, dst_mesh), data(solver->regions.size()) {}

    void compute(double wavelength, InterpolationMethod interp) {
        // Compute gains on mesh for each active region
        OmpLockGuard lock(gain_omp_lock);
        for (size_t reg = 0; reg != this->solver->regions.size(); ++reg) {
            if (this->regions[reg].size() == 0) continue;
            AveragedData temps(this->solver, "temperature", make_shared<ActiveRegionMesh>(this, reg), this->solver->regions[reg]);
            AveragedData concs(temps);
            concs.name = "carriers concentration";
            temps.data = this->solver->inTemperature(temps.mesh, interp);
            concs.data = this->solver->inCarriersConcentration(CarriersConcentration::PAIRS, concs.mesh, interp);
            this->data[reg] = getValues(wavelength, interp, reg, concs, temps);
        }
    }

    virtual DataVector<Tensor2<double>> getValues(double wavelength,
                                                  InterpolationMethod interp,
                                                  size_t reg,
                                                  const AveragedData& concs,
                                                  const AveragedData& temps) = 0;

    Tensor2<double> at(size_t i) const override {
        for (size_t reg = 0; reg != this->regions.size(); ++reg) {
            auto idx = this->regions[reg].indexOf(i);
            if (idx != CompressedSetOfNumbers<>::NOT_INCLUDED) return this->data[reg][idx];
        }
        return Tensor2<double>(0., 0.);
    }
};

struct FreeCarrierGainSolver3D::GainData : public FreeCarrierGainSolver3D::ComputedData {
    using typename DataBaseTensor2::AveragedData;

    template <typename... Args> GainData(Args... args) : ComputedData(args...) {}

    DataVector<Tensor2<double>> getValues(double wavelength,
                                          InterpolationMethod interp,
                                          size_t reg,
                                          const AveragedData& concs,
                                          const AveragedData& temps) override {
        double hw = phys::h_eVc1e9 / wavelength;
        DataVector<Tensor2<double>> values(this->regions[reg].size());
        std::exception_ptr error;

        if (this->solver->inFermiLevels.hasProvider()) {
            AveragedData Fcs(temps);
            Fcs.name = "quasi Fermi level for electrons";
            AveragedData Fvs(temps);
            Fvs.name = "quasi Fermi level for holes";
            Fcs.data = this->solver->inFermiLevels(FermiLevels::ELECTRONS, temps.mesh, interp);
            Fvs.data = this->solver->inFermiLevels(FermiLevels::HOLES, temps.mesh, interp);
PLASK_OMP_PARALLEL_FOR
            for (plask::openmp_size_t i = 0; i < this->regions[reg].size(); ++i) {
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
PLASK_OMP_PARALLEL_FOR
            for (plask::openmp_size_t i = 0; i < this->regions[reg].size(); ++i) {
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

struct FreeCarrierGainSolver3D::DgdnData : public FreeCarrierGainSolver3D::ComputedData {
    using typename DataBaseTensor2::AveragedData;

    template <typename... Args> DgdnData(Args... args) : ComputedData(args...) {}

    DataVector<Tensor2<double>> getValues(double wavelength,
                                          InterpolationMethod /*interp*/,
                                          size_t reg,
                                          const AveragedData& concs,
                                          const AveragedData& temps) override {
        double hw = phys::h_eVc1e9 / wavelength;
        const double h = 0.5 * DIFF_STEP;
        DataVector<Tensor2<double>> values(this->regions[reg].size());
        std::exception_ptr error;
PLASK_OMP_PARALLEL_FOR
        for (plask::openmp_size_t i = 0; i < this->regions[reg].size(); ++i) {
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

const LazyData<Tensor2<double>> FreeCarrierGainSolver3D::getGainData(Gain::EnumType what,
                                                                     const shared_ptr<const MeshD<3>>& dst_mesh,
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
        throw BadInput(this->getId(), "wrong gain type requested");
    }
}

struct FreeCarrierGainSolver3D::EnergyLevelsData : public FreeCarrierGainSolver3D::DataBaseVector {
    using typename DataBaseVector::AveragedData;

    size_t which;
    std::vector<AveragedData> temps;
    mutable bool quiet = false;

    EnergyLevelsData(EnergyLevels::EnumType which,
                     FreeCarrierGainSolver3D* solver,
                     const shared_ptr<const MeshD<3>>& dst_mesh,
                     InterpolationMethod interp)
        : DataBaseVector(solver, dst_mesh), which(size_t(which)) {
        temps.reserve(solver->regions.size());
        for (size_t reg = 0; reg != solver->regions.size(); ++reg) {
            temps.emplace_back(this->solver, "temperature", make_shared<ActiveRegionMesh>(this, reg), this->solver->regions[reg]);
            temps.back().data = this->solver->inTemperature(temps.back().mesh, interp);
        }
    }

    std::vector<double> at(size_t i) const override {
        for (size_t reg = 0; reg != this->solver->regions.size(); ++reg) {
            auto idx = this->regions[reg].indexOf(i);
            if (idx != CompressedSetOfNumbers<>::NOT_INCLUDED) {
                double T = temps[reg][idx];
                ActiveRegionParams params(this->solver, this->solver->params0[reg], T, quiet);
                quiet = true;
                std::vector<double> result;
                result.reserve(params.levels[which].size());
                for (const auto& level : params.levels[which]) result.push_back(level.E);
                return result;
            }
        }
        return std::vector<double>();
    }
};

const LazyData<std::vector<double>> FreeCarrierGainSolver3D::getEnergyLevels(EnergyLevels::EnumType which,
                                                                             const shared_ptr<const MeshD<3>>& dst_mesh,
                                                                             InterpolationMethod interp) {
    this->initCalculation();
    EnergyLevelsData* data = new EnergyLevelsData(which, this, dst_mesh, getInterpolationMethod<INTERPOLATION_LINEAR>(interp));
    return LazyData<std::vector<double>>(data);
}

std::string FreeCarrierGainSolver3D::getClassName() const { return "gain.FreeCarrier3D"; }

}}}  // namespace plask::gain::freecarrier
