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
#include "freecarrier.hpp"
#include "fd.hpp"
#include "gauss_matrix.hpp"

namespace plask { namespace gain { namespace freecarrier {

constexpr double DIFF_STEP = 0.001;

OmpNestLock gain_omp_lock;

template <typename BaseT>
FreeCarrierGainSolver<BaseT>::Level::Level(double E, const Tensor2<double>& M, WhichLevel which, const ActiveRegionParams& params)
    : E(E), M(M) {
    thickness = 0.;
    if (which == EL) {
        for (size_t i = 0; i < params.U[EL].size(); ++i)
            if (params.U[EL][i] < E) thickness += params.region.thicknesses[i];
    } else {
        for (size_t i = 0; i < params.U[which].size(); ++i)
            if (params.U[which][i] > E) thickness += params.region.thicknesses[i];
    }
}

template <typename BaseT>
FreeCarrierGainSolver<BaseT>::FreeCarrierGainSolver(const std::string& name)
    : BaseT(name),
      outGain(this, &FreeCarrierGainSolver<BaseT>::getGainData),
      outEnergyLevels(this, &FreeCarrierGainSolver<BaseT>::getEnergyLevels),
      lifetime(0.1),
      matrixelem(0.),
      T0(300.),
      levelsep(0.0001),
      strained(false),
      quick_levels(true) {
    inTemperature = 300.;
    inTemperature.changedConnectMethod(this, &FreeCarrierGainSolver<BaseT>::onInputChange);
    inCarriersConcentration.changedConnectMethod(this, &FreeCarrierGainSolver<BaseT>::onInputChange);
}

template <typename BaseT> FreeCarrierGainSolver<BaseT>::~FreeCarrierGainSolver() {
    inTemperature.changedDisconnectMethod(this, &FreeCarrierGainSolver<BaseT>::onInputChange);
    inCarriersConcentration.changedDisconnectMethod(this, &FreeCarrierGainSolver<BaseT>::onInputChange);
}

template <typename BaseT> void FreeCarrierGainSolver<BaseT>::loadConfiguration(XMLReader& reader, Manager& manager) {
    // Load a configuration parameter from XML.
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "config") {
            lifetime = reader.getAttribute<double>("lifetime", lifetime);
            matrixelem = reader.getAttribute<double>("matrix-elem", matrixelem);
            T0 = reader.getAttribute<double>("T0", T0);
            strained = reader.getAttribute<bool>("strained", strained);
            //             quick_levels = reader.getAttribute<bool>("quick-levels", quick_levels);
            reader.requireTagEnd();
        } else
            this->parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <levels>, or <config>");
    }
}

template <typename BaseT> void FreeCarrierGainSolver<BaseT>::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    detectActiveRegions();
    estimateLevels();
    outGain.fireChanged();
}

template <typename BaseT> void FreeCarrierGainSolver<BaseT>::onInvalidate() {
    params0.clear();
    regions.clear();
    substrateMaterial.reset();
}

// template <typename BaseT>
// void FreeCarrierGainSolver<BaseT>::compute()
//{
//     this->initCalculation(); // This must be called before any calculation!
// }

template <typename BaseT>
void FreeCarrierGainSolver<BaseT>::ActiveRegionInfo::summarize(const FreeCarrierGainSolver<BaseT>* solver) {
    holes = BOTH_HOLES;
    auto bbox = layers->getBoundingBox();
    total = bbox.upper.vert() - bbox.lower.vert() - bottom - top;
    materials.clear();
    materials.reserve(layers->children.size());
    thicknesses.clear();
    thicknesses.reserve(layers->children.size());
    for (const auto& layer : layers->children) {
        auto block =
            static_cast<Block<GeometryType::DIM>*>(static_cast<Translation<GeometryType::DIM>*>(layer.get())->getChild().get());
        auto material = block->singleMaterial();
        if (!material) throw plask::Exception("{}: Active region can consist only of solid layers", solver->getId());
        auto bbox = static_cast<GeometryObjectD<GeometryType::DIM>*>(layer.get())->getBoundingBox();
        double thck = bbox.upper.vert() - bbox.lower.vert();
        materials.push_back(material);
        thicknesses.push_back(thck);
    }
    double substra = solver->strained ? solver->substrateMaterial->lattC(solver->T0, 'a') : 0.;
    if (materials.size() > 2) {
        Material* material = materials[0].get();
        double e;
        if (solver->strained) {
            double latt = material->lattC(solver->T0, 'a');
            e = (substra - latt) / latt;
        } else
            e = 0.;
        double el0 = material->CB(solver->T0, e, 'G'), hh0 = material->VB(solver->T0, e, 'G', 'H'),
               lh0 = material->VB(solver->T0, e, 'G', 'L');
        material = materials[1].get();
        if (solver->strained) {
            double latt = material->lattC(solver->T0, 'a');
            e = (substra - latt) / latt;
        } else
            e = 0.;
        double el1 = material->CB(solver->T0, e, 'G'), hh1 = material->VB(solver->T0, e, 'G', 'H'),
               lh1 = material->VB(solver->T0, e, 'G', 'L');
        for (size_t i = 2; i < materials.size(); ++i) {
            material = materials[i].get();
            if (solver->strained) {
                double latt = material->lattC(solver->T0, 'a');
                e = (substra - latt) / latt;
            } else
                e = 0.;
            double el2 = material->CB(solver->T0, e, 'G');
            double hh2 = material->VB(solver->T0, e, 'G', 'H');
            double lh2 = material->VB(solver->T0, e, 'G', 'L');
            if ((el0 < el1 && el1 > el2) || (hh0 > hh1 && hh1 < hh2) || (lh0 > lh1 && lh1 < lh2)) {
                if (i != 2 && i != materials.size() - 1) {
                    bool eb = (el0 < el1 && el1 > el2);
                    if (eb != (hh0 > hh1 && hh1 < hh2)) holes = ConsideredHoles(holes & ~HEAVY_HOLES);
                    if (eb != (lh0 > lh1 && lh1 < lh2)) holes = ConsideredHoles(holes & ~LIGHT_HOLES);
                }
                if (holes == NO_HOLES)
                    throw Exception("{0}: Quantum wells in conduction band do not coincide with wells is valence band",
                                    solver->getId());
                if ((el0 < el1 && el1 > el2) || (hh0 > hh1 && hh1 < hh2 && holes & HEAVY_HOLES) ||
                    (lh0 > lh1 && lh1 < lh2 && holes & LIGHT_HOLES))
                    wells.push_back(i - 1);
            } else if (i == 2)
                wells.push_back(0);
            if (el2 != el1) {
                el0 = el1;
                el1 = el2;
            }
            if (hh2 != hh1) {
                hh0 = hh1;
                hh1 = hh2;
            }
            if (lh2 != lh1) {
                lh0 = lh1;
                lh1 = lh2;
            }
        }
    }
    if (wells.back() < materials.size() - 2) wells.push_back(materials.size() - 1);
    totalqw = 0.;
    for (size_t i = 0; i < thicknesses.size(); ++i)
        if (isQW(i)) totalqw += thicknesses[i];

#ifndef NDEBUG
    solver->writelog(LOG_DEBUG, "Active region @ {1}  ({2}/{3})", origin, bottom, top);
    assert(materials.size() == thicknesses.size());
    std::string ws = "-";
    for (size_t i = 0; i != materials.size(); ++i) {
        auto w = std::find(wells.begin(), wells.end(), i);
        if (w != wells.end()) ws = format("{:d}", size_t(w - wells.begin()));
        solver->writelog(LOG_DEBUG, "[{4}]  {1:.2f}nm {2}{3}", thicknesses[i] * 1e3, materials[i]->name(),
                         isQW(i) ? "  (QW)" : "", ws);
    }
#endif
}

template <typename BaseT>
FreeCarrierGainSolver<BaseT>::ActiveRegionParams::ActiveRegionParams(const FreeCarrierGainSolver* solver,
                                                                     const ActiveRegionInfo& region,
                                                                     double T,
                                                                     bool quiet,
                                                                     double mt)
    : region(region) {
    size_t n = region.materials.size();
    U[EL].reserve(n);
    U[HH].reserve(n);
    U[LH].reserve(n);
    M[EL].reserve(n);
    M[HH].reserve(n);
    M[LH].reserve(n);
    double substra = solver->strained ? solver->substrateMaterial->lattC(T, 'a') : 0.;
    Eg = std::numeric_limits<double>::max();

    size_t mi = 0;
    double me = 0;

    if (!solver->inBandEdges.hasProvider()) {
        if (!quiet) solver->writelog(LOG_DETAIL, "Band edges taken from material database");
        size_t i = 0;
        for (auto material : region.materials) {
            OmpLockGuard<OmpNestLock> lockq = material->lock();
            double e;
            if (solver->strained) {
                double latt = material->lattC(T, 'a');
                e = (substra - latt) / latt;
            } else
                e = 0.;
            double uel = material->CB(T, e, 'G'), uhh = material->VB(T, e, 'G', 'H');
            U[EL].push_back(uel);
            U[HH].push_back(uhh);
            double eg = uel - uhh;
            if (eg < Eg) {
                Eg = eg;
                mi = i;
                me = e;
            }
            U[LH].push_back(material->VB(T, e, 'G', 'L'));
            M[EL].push_back(material->Me(T, e));
            M[HH].push_back(material->Mhh(T, e));
            M[LH].push_back(material->Mlh(T, e));
            ++i;
        }
    } else {
        if (!quiet) solver->writelog(LOG_DETAIL, "Band edges taken from inBandEdges receiver");
        shared_ptr<Translation<GeometryType::DIM>> shifted(new Translation<GeometryType::DIM>(region.layers, region.origin));
        auto mesh = makeGeometryGrid(shifted)->getElementMesh();
        assert(mesh->size() == mesh->vert()->size());
        auto CB = solver->inBandEdges(BandEdges::CONDUCTION, mesh);
        auto VB_H = solver->inBandEdges(BandEdges::VALENCE_HEAVY, mesh);
        auto VB_L = solver->inBandEdges(BandEdges::VALENCE_LIGHT, mesh);
        for (size_t i = 0; i != mesh->vert()->size(); ++i) {
            auto material = region.materials[i];
            OmpLockGuard<OmpNestLock> lockq = material->lock();
            double e;
            if (solver->strained) {
                double latt = material->lattC(T, 'a');
                e = (substra - latt) / latt;
            } else
                e = 0.;
            double uel = CB[i];
            double uhh = VB_H[i];
            U[EL].push_back(uel);
            U[HH].push_back(uhh);
            double eg = uel - uhh;
            if (eg < Eg) {
                Eg = eg;
                mi = i;
                me = e;
            }
            U[LH].push_back(VB_L[i]);
            M[EL].push_back(material->Me(T, e));
            M[HH].push_back(material->Mhh(T, e));
            M[LH].push_back(material->Mlh(T, e));
        }
    }
    if (mt == 0.) {
        if (solver->matrixelem != 0.) {
            Mt = solver->matrixelem;
        } else {
            double deltaSO = region.materials[mi]->Dso(T, me);
            Mt = (1. / M[EL][mi].c11 - 1.) * (Eg + deltaSO) * Eg / (Eg + 0.666666666666667 * deltaSO) / 2.;
            if (!quiet) solver->writelog(LOG_DETAIL, "Estimated momentum matrix element to {:.2f} eV m0", Mt);
        }
    } else {
        Mt = mt;
    }
}

template <typename BaseT>
double FreeCarrierGainSolver<BaseT>::level(WhichLevel which, double E, const ActiveRegionParams& params, size_t start, size_t stop)
    const {
    size_t nA = 2 * (stop - start + 1);

    DgbMatrix A(nA);

    constexpr double fact = 2e-12 * phys::me / (phys::hb_eV * phys::hb_J);

    double m1 = params.M[which][start].c11;
    double k1_2 = fact * m1 * (E - params.U[which][start]);
    if (which != EL) k1_2 = -k1_2;
    double k1 = sqrt(abs(k1_2));

    // Wave functions are confined, so we can assume exponentially decreasing relation in the outer layers
    A(0, 0) = A(nA - 1, nA - 1) = 1.;
    A(0, 1) = A(nA - 1, nA - 2) = 0.;

    for (size_t i = start, o = 1; i < stop; i++, o += 2) {
        double k0_2 = k1_2, k0 = k1, m0 = m1;
        double d = (o == 1) ? 0. : params.region.thicknesses[i];
        if (k0_2 >= 0.) {
            double coskd = cos(k0 * d), sinkd = sin(k0 * d);
            A(o, o - 1) = coskd;
            A(o + 1, o - 1) = -sinkd;
            A(o, o) = sinkd;
            A(o + 1, o) = coskd;
        } else {
            double phi = exp(-k0 * d);
            A(o, o - 1) = phi;
            A(o + 1, o - 1) = -phi;
            A(o, o) = 1. / phi;
            A(o + 1, o) = 1. / phi;
        }

        A(o + 2, o) = 0.;
        A(o - 1, o + 1) = 0.;

        m1 = params.M[which][i + 1].c11;
        k1_2 = fact * m1 * (E - params.U[which][i + 1]);
        if (which != EL) k1_2 = -k1_2;
        if (k1_2 >= 0.) {
            k1 = sqrt(k1_2);
            A(o, o + 1) = -1.;
            A(o + 1, o + 1) = 0.;
            A(o, o + 2) = 0.;
            A(o + 1, o + 2) = -(k1 * m0) / (k0 * m1);
        } else {
            k1 = sqrt(-k1_2);
            double f = (k1 * m0) / (k0 * m1);
            A(o, o + 1) = -1.;
            A(o + 1, o + 1) = f;
            A(o, o + 2) = -1.;
            A(o + 1, o + 2) = -f;
        }
    }

    return A.determinant();
}

template <typename BaseT>
void FreeCarrierGainSolver<BaseT>::estimateWellLevels(WhichLevel which, ActiveRegionParams& params, size_t qw) const {
    if (params.U[which].size() < 3) return;

    size_t start = params.region.wells[qw], stop = params.region.wells[qw + 1];
    double umin = std::numeric_limits<double>::max(), umax = std::numeric_limits<double>::lowest();
    double num = 0.;
    double ustart, ustop;
    Tensor2<double> M;
    for (size_t i = start; i <= stop; ++i) {
        double ub = params.U[which][i];
        if (i == start) ustart = ub;
        if (i == stop) ustop = ub;
        auto m = params.M[which][i];
        if (which == EL) {
            if (ub < umin) {
                umin = ub;
                M = m;
            }
        } else {
            if (ub > umax) {
                umax = ub;
                M = m;
            }
        }
        if (i != start && i != stop) {
            double no = 1e-6 / PI * params.region.thicknesses[i] * sqrt(2. * phys::me / (phys::hb_eV * phys::hb_J) * m.c11);
            num = max(no, num);
        }
    }
    if (which == EL)
        umax = min(ustart, ustop);
    else
        umin = max(ustart, ustop);
    if (umax < umin)
        throw Exception("{}: Outer layers of active region have wrong band offset", this->getId());  // TODO make clearer
    num = 2. * ceil(sqrt(umax - umin) * num);  // 2.* is the simplest way to ensure that all levels are found
    umin += 0.5 * levelsep;
    umax -= 0.5 * levelsep;
    double step = (umax - umin) / num;
    size_t n = size_t(num);
    double a, b = umin;
    double fa, fb = level(which, b, params, qw);
    if (fb == 0.) {
        params.levels[which].emplace_back(fb, M, which, params);
        b += levelsep;
        fb = level(which, b, params, qw);
    }
    for (size_t i = 0; i < n; ++i) {
        a = b;
        fa = fb;
        b = a + step;
        fb = level(which, b, params, qw);
        if (fb == 0.) {
            params.levels[which].emplace_back(fb, M, which, params);
            continue;
        }
        if ((fa < 0.) != (fb < 0.)) {
            boost::uintmax_t iters = 1000;
            double xa, xb;
            std::tie(xa, xb) = toms748_solve([&](double x) { return level(which, x, params, qw); }, a, b, fa, fb,
                                             [this](double l, double r) { return r - l < levelsep; }, iters);
            if (xb - xa > levelsep) throw ComputationError(this->getId(), "Could not find level estimate in quantum well");
            params.levels[which].emplace_back(0.5 * (xa + xb), M, which, params);
        }
    }
}

template <typename BaseT>
void FreeCarrierGainSolver<BaseT>::estimateAboveLevels(WhichLevel which, ActiveRegionParams& params) const {
    if (params.U[which].size() < 5) return;  // This makes sense with at least two quantum wells

    /// Detect range above the wells
    size_t N = params.U[EL].size() - 1;
    double umin = std::numeric_limits<double>::max(), umax = std::numeric_limits<double>::lowest();
    if (which == EL)
        umax = min(params.U[EL][0], params.U[EL][N]);
    else
        umin = max(params.U[which][0], params.U[which][params.U[which].size() - 1]);
    Tensor2<double> M;
    for (size_t i : params.region.wells) {
        if (i == 0 || i == N) continue;
        double ub = params.U[which][i];
        if (which == EL) {
            if (ub < umin) {
                umin = ub;
                M = params.M[which][i];
            }
        } else {
            if (ub > umax) {
                umax = ub;
                M = params.M[which][i];
            }
        }
    }

    if (umax <= umin) return;

    double num =
        2. * ceil(1e-6 / PI * params.region.total * sqrt(2. * (umax - umin) * phys::me / (phys::hb_eV * phys::hb_J) * M.c11));
    umin += 0.5 * levelsep;
    umax -= 0.5 * levelsep;
    double step = (umax - umin) / num;
    size_t n = size_t(num);
    double a, b = umin;
    double fa, fb = level(which, b, params);
    if (fb == 0.) {
        params.levels[which].emplace_back(fb, M, which, params);
        b += levelsep;
        fb = level(which, b, params);
    }
    for (size_t i = 0; i < n; ++i) {
        a = b;
        fa = fb;
        b = a + step;
        fb = level(which, b, params);
        if (fb == 0.) {
            params.levels[which].emplace_back(fb, M, which, params);
            continue;
        }
        if ((fa < 0.) != (fb < 0.)) {
            boost::uintmax_t iters = 1000;
            double xa, xb;
            std::tie(xa, xb) = toms748_solve([&](double x) { return level(which, x, params); }, a, b, fa, fb,
                                             [this](double l, double r) { return r - l < levelsep; }, iters);
            if (xb - xa > levelsep) throw ComputationError(this->getId(), "Could not find level estimate above quantum wells");
            params.levels[which].emplace_back(0.5 * (xa + xb), M, which, params);
        }
    }
}

template <typename BaseT> void FreeCarrierGainSolver<BaseT>::estimateLevels() {
    params0.clear();
    params0.reserve(regions.size());

    size_t reg = 0;
    for (const ActiveRegionInfo& region : regions) {
        params0.emplace_back(this, region);
        ActiveRegionParams& params = params0.back();
        for (size_t qw = 0; qw < region.wells.size() - 1; ++qw) {
            estimateWellLevels(EL, params, qw);
            if (region.holes & ActiveRegionInfo::HEAVY_HOLES)
                estimateWellLevels(HH, params, qw);
            else
                params.levels[HH].clear();
            if (region.holes & ActiveRegionInfo::LIGHT_HOLES)
                estimateWellLevels(LH, params, qw);
            else
                params.levels[LH].clear();
        }
        std::sort(params.levels[EL].begin(), params.levels[EL].end(), [](const Level& a, const Level& b) { return a.E < b.E; });
        std::sort(params.levels[HH].begin(), params.levels[HH].end(), [](const Level& a, const Level& b) { return a.E > b.E; });
        std::sort(params.levels[LH].begin(), params.levels[LH].end(), [](const Level& a, const Level& b) { return a.E > b.E; });
        params.nhh = std::min(params.levels[EL].size(), params.levels[HH].size());
        params.nlh = std::min(params.levels[EL].size(), params.levels[LH].size());
        estimateAboveLevels(EL, params);
        estimateAboveLevels(HH, params);
        estimateAboveLevels(LH, params);

        if (maxLoglevel > LOG_DETAIL) {
            {
                std::stringstream str;
                std::string sep = "";
                for (auto l : params.levels[EL]) {
                    str << sep << format("{:.4f}", l.E);
                    sep = ", ";
                }
                this->writelog(LOG_DETAIL, "Estimated electron levels for active region {:d} (eV): {}", reg++, str.str());
            }
            {
                std::stringstream str;
                std::string sep = "";
                for (auto l : params.levels[HH]) {
                    str << sep << format("{:.4f}", l.E);
                    sep = ", ";
                }
                this->writelog(LOG_DETAIL, "Estimated heavy hole levels for active region {:d} (eV): {}", reg - 1, str.str());
            }
            {
                std::stringstream str;
                std::string sep = "";
                for (auto l : params.levels[LH]) {
                    str << sep << format("{:.4f}", l.E);
                    sep = ", ";
                }
                this->writelog(LOG_DETAIL, "Estimated light hole levels for active region {:d} (eV): {}", reg - 1, str.str());
            }
        }

        if (params.levels[EL].empty()) throw Exception("{}: No electron levels found", this->getId());
        if (params.levels[HH].empty() && params.levels[LH].empty()) throw Exception("{}: No hole levels found", this->getId());
    }
}

template <typename BaseT> double FreeCarrierGainSolver<BaseT>::getN(double F, double T, const ActiveRegionParams& params) const {
    size_t n = params.levels[EL].size();
    const double kT = phys::kB_eV * T;
    constexpr double fact = phys::me * phys::kB_eV / (2. * PI * phys::hb_eV * phys::hb_J);  // 1/µm (1e6) -> 1/cm³ (1e-6)

    double N = 2e-6 * pow(fact * T * params.sideM(EL).c00, 1.5) * fermiDiracHalf((F - params.sideU(EL)) / kT);

    for (size_t i = 0; i < n; ++i) {
        double M = params.levels[EL][i].M.c00;
        N += 2. * fact * T * M / params.levels[EL][i].thickness * log(1 + exp((F - params.levels[EL][i].E) / kT));
    }

    return N;
}

template <typename BaseT> double FreeCarrierGainSolver<BaseT>::getP(double F, double T, const ActiveRegionParams& params) const {
    size_t nh = params.levels[HH].size(), nl = params.levels[LH].size();
    const double kT = phys::kB_eV * T;
    constexpr double fact = phys::me * phys::kB_eV / (2. * PI * phys::hb_eV * phys::hb_J);  // 1/µm (1e6) -> 1/cm³ (1e-6)

    // Get parameters for outer layers
    double N = 2e-6 * (pow(fact * T * params.sideM(HH).c00, 1.5) * fermiDiracHalf((params.sideU(HH) - F) / kT) +
                       pow(fact * T * params.sideM(LH).c00, 1.5) * fermiDiracHalf((params.sideU(LH) - F) / kT));

    for (size_t i = 0; i < nh; ++i) {
        double M = params.levels[HH][i].M.c00;
        N += 2. * fact * T * M / params.levels[HH][i].thickness * log(1 + exp((params.levels[HH][i].E - F) / kT));
    }

    for (size_t i = 0; i < nl; ++i) {
        double M = params.levels[LH][i].M.c00;
        N += 2. * fact * T * M / params.levels[LH][i].thickness * log(1 + exp((params.levels[LH][i].E - F) / kT));
    }

    return N;
}

template <typename BaseT>
void FreeCarrierGainSolver<BaseT>::findFermiLevels(double& Fc, double& Fv, double n, double T, const ActiveRegionParams& params)
    const {
    double Ue = params.sideU(EL), Uh = params.sideU(HH);
    double fs = 0.05 * abs(Ue - Uh);
    if (fs <= levelsep) fs = 2. * levelsep;
    if (isnan(Fc)) Fc = Ue;
    if (isnan(Fv)) Fv = Uh;
    boost::uintmax_t iters;
    double xa, xb;

    iters = 1000;
    std::tie(xa, xb) = fermi_bracket_and_solve([this, T, n, &params](double x) { return getN(x, T, params) - n; }, Fc, fs, iters);
    if (xb - xa > levelsep) throw ComputationError(this->getId(), "Could not find quasi-Fermi level for electrons");
    Fc = 0.5 * (xa + xb);

    iters = 1000;
    std::tie(xa, xb) = fermi_bracket_and_solve([this, T, n, &params](double x) { return getP(x, T, params) - n; }, Fv, fs, iters);
    if (xb - xa > levelsep) throw ComputationError(this->getId(), "Could not find quasi-Fermi level for holes");
    Fv = 0.5 * (xa + xb);
}

template <typename BaseT>
Tensor2<double> FreeCarrierGainSolver<BaseT>::getGain0(double hw,
                                                       double Fc,
                                                       double Fv,
                                                       double T,
                                                       double nr,
                                                       const ActiveRegionParams& params) const {
    constexpr double fac = 1e4 * phys::qe * phys::qe / (2. * phys::c * phys::epsilon0 * phys::hb_J);  // 1e4: 1/µm -> 1/cm
    const double ikT = (1. / phys::kB_eV) / T;
    const double Dhw = hw - params.Eg;

    Tensor2<double> g(0., 0.);

    for (size_t i = 0; i < params.nhh; ++i) {
        const double Ec = params.levels[EL][i].E, Ev = params.levels[HH][i].E;
        const double Ep = hw - (Ec - Ev);
        if (Ep < 0.) continue;
        const double sin2 = (Dhw > 0.) ? Ep / Dhw : 0.;
        const Tensor2<double> pp(1. - 0.5 * sin2, sin2);
        const double mu = 1. / (1. / params.levels[EL][i].M.c00 + 1. / params.levels[HH][i].M.c00);
        const double Ecp = Ec + Ep * mu / params.levels[EL][i].M.c00, Evp = Ev - Ep * mu / params.levels[HH][i].M.c00;
        g += mu * (1. / (exp(ikT * (Ecp - Fc)) + 1) - 1. / (exp(ikT * (Evp - Fv)) + 1)) * pp;
    }

    for (size_t i = 0; i < params.nlh; ++i) {
        const double Ec = params.levels[EL][i].E, Ev = params.levels[LH][i].E;
        const double Ep = hw - (Ec - Ev);
        if (Ep < 0.) continue;
        const double sin2 = (Dhw > 0.) ? Ep / Dhw : 0.;
        const Tensor2<double> pp(0.3333333333333333333333 + 0.5 * sin2, 1.3333333333333333333333 - sin2);
        const double mu = 1. / (1. / params.levels[EL][i].M.c00 + 1. / params.levels[LH][i].M.c00);
        const double Ecp = Ec + Ep * mu / params.levels[EL][i].M.c00, Evp = Ev - Ep * mu / params.levels[LH][i].M.c00;
        g += mu * (1. / (exp(ikT * (Ecp - Fc)) + 1) - 1. / (exp(ikT * (Evp - Fv)) + 1)) * pp;
    }
    return fac / (hw * nr * params.region.totalqw) * params.Mt * g;
}

template <typename BaseT>
Tensor2<double> FreeCarrierGainSolver<BaseT>::getGain(double hw,
                                                      double Fc,
                                                      double Fv,
                                                      double T,
                                                      double nr,
                                                      const ActiveRegionParams& params) const {
    if (lifetime == 0) return getGain0(hw, Fc, Fv, T, nr, params);

    const double E0 = params.levels[EL][0].E - ((params.region.holes == ActiveRegionInfo::BOTH_HOLES)
                                                    ? std::max(params.levels[HH][0].E, params.levels[LH][0].E)
                                                : (params.region.holes == ActiveRegionInfo::HEAVY_HOLES) ? params.levels[HH][0].E
                                                                                                         : params.levels[LH][0].E);

    const double b = 1e12 * phys::hb_eV / lifetime;
    const double tmax = 32. * b;
    const double tmin = std::max(-tmax, E0 - hw);
    double dt = (tmax - tmin) / 1024.;  // TODO Estimate integral precision and maybe chose better integration

    Tensor2<double> g = Tensor2<double>(0., 0.);
    for (double t = tmin; t <= tmax; t += dt) {
        // L(t) = b / (π (t²+b²)),
        g += getGain0(hw + t, Fc, Fv, T, nr, params) / (t * t + b * b);
    }
    g *= b * dt / PI;

    return g;
}

template struct PLASK_SOLVER_API FreeCarrierGainSolver<SolverWithMesh<Geometry2DCartesian, MeshAxis>>;
template struct PLASK_SOLVER_API FreeCarrierGainSolver<SolverWithMesh<Geometry2DCylindrical, MeshAxis>>;
template struct PLASK_SOLVER_API FreeCarrierGainSolver<SolverOver<Geometry3D>>;

}}}  // namespace plask::gain::freecarrier
