#include "gauss_matrix.h"
#include "freecarrier.h"
#include "fd.h"

namespace plask { namespace gain { namespace freecarrier {

constexpr double DIFF_STEP = 0.001;

template <typename GeometryType>
FreeCarrierGainSolver<GeometryType>::Level::Level(double E, const Tensor2<double>& M,
                                                  WhichLevel which, const ActiveRegionParams& params):
    E(E), M(M)
{
    thickness = 0.;
    if (which == EL) {
        for (size_t i = 0; i < params.U[EL].size(); ++i)
            if (params.U[EL][i] < E) thickness += params.region.thicknesses[i];
    } else {
        for (size_t i = 0; i < params.U[which].size(); ++i)
            if (params.U[which][i] > E) thickness += params.region.thicknesses[i];
    }
}



template <typename GeometryType>
FreeCarrierGainSolver<GeometryType>::FreeCarrierGainSolver(const std::string& name): SolverWithMesh<GeometryType, MeshAxis>(name),
    outGain(this, &FreeCarrierGainSolver<GeometryType>::getGainData),
//     outEnergyLevels(this, &FreeCarrierGainSolver<GeometryType>::getEnergyLevels),
    lifetime(0.1),
    matrixelem(0.),
    T0(300.),
    levelsep(0.0001),
    strained(false),
    quick_levels(true)
{
    inTemperature = 300.;
    inTemperature.changedConnectMethod(this, &FreeCarrierGainSolver<GeometryType>::onInputChange);
    inCarriersConcentration.changedConnectMethod(this, &FreeCarrierGainSolver<GeometryType>::onInputChange);
}

template <typename GeometryType>
FreeCarrierGainSolver<GeometryType>::~FreeCarrierGainSolver() {
    inTemperature.changedDisconnectMethod(this, &FreeCarrierGainSolver<GeometryType>::onInputChange);
    inCarriersConcentration.changedDisconnectMethod(this, &FreeCarrierGainSolver<GeometryType>::onInputChange);
}


template <typename GeometryType>
void FreeCarrierGainSolver<GeometryType>::loadConfiguration(XMLReader& reader, Manager& manager)
{
    // Load a configuration parameter from XML.
    while (reader.requireTagOrEnd())
    {
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

template <typename GeometryType>
void FreeCarrierGainSolver<GeometryType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    detectActiveRegions();
    estimateLevels();
    outGain.fireChanged();
}


template <typename GeometryType>
void FreeCarrierGainSolver<GeometryType>::onInvalidate()
{
    params0.clear();
    regions.clear();
    materialSubstrate.reset();
}

//template <typename GeometryType>
//void FreeCarrierGainSolver<GeometryType>::compute()
//{
//    this->initCalculation(); // This must be called before any calculation!
//}


template <typename GeometryType>
void FreeCarrierGainSolver<GeometryType>::detectActiveRegions()
{
    regions.clear();

    shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(this->geometry->getChild());
    shared_ptr<RectangularMesh<2>> points = mesh->getMidpointsMesh();

    size_t ileft = 0, iright = points->axis0->size();
    bool in_active = false;

    bool added_bottom_cladding = false;
    bool added_top_cladding = false;

    for (size_t r = 0; r < points->axis1->size(); ++r) {
        bool had_active = false; // indicates if we had active region in this layer
        shared_ptr<Material> layer_material;
        bool layer_QW = false;

        for (size_t c = 0; c < points->axis0->size(); ++c)
        { // In the (possible) active region
            auto point = points->at(c,r);
            auto tags = this->geometry->getRolesAt(point);
            bool active = false; for (const auto& tag: tags) if (tag.substr(0,6) == "active") { active = true; break; }
            bool QW = tags.find("QW") != tags.end()/* || tags.find("QD") != tags.end()*/;
            bool substrate = tags.find("substrate") != tags.end();

            if (substrate) {
                if (!materialSubstrate)
                    materialSubstrate = this->geometry->getMaterial(point);
                else if (*materialSubstrate != *this->geometry->getMaterial(point))
                    throw Exception("{0}: Non-uniform substrate layer.", this->getId());
            }

            if (QW && !active)
                throw Exception("{0}: All marked quantum wells must belong to marked active region.", this->getId());

            if (c < ileft) {
                if (active)
                    throw Exception("{0}: Left edge of the active region not aligned.", this->getId());
            } else if (c >= iright) {
                if (active)
                    throw Exception("{0}: Right edge of the active region not aligned.", this->getId());
            } else {
                // Here we are inside potential active region
                if (active) {
                    if (!had_active) {
                        if (!in_active)
                        { // active region is starting set-up new region info
                            regions.emplace_back(mesh->at(c,r));
                            ileft = c;
                        }
                        layer_material = this->geometry->getMaterial(point);
                        layer_QW = QW;
                    } else {
                        if (*layer_material != *this->geometry->getMaterial(point))
                            throw Exception("{0}: Non-uniform active region layer.", this->getId());
                        if (layer_QW != QW)
                            throw Exception("{0}: Quantum-well role of the active region layer not consistent.", this->getId());
                    }
                } else if (had_active) {
                    if (!in_active) {
                        iright = c;

                        // add layer below active region (cladding) LUKASZ
                        /*auto bottom_material = this->geometry->getMaterial(points->at(ileft,r-1));
                        for (size_t cc = ileft; cc < iright; ++cc)
                            if (*this->geometry->getMaterial(points->at(cc,r-1)) != *bottom_material)
                                throw Exception("{0}: Material below quantum well not uniform.", this->getId());
                        auto& region = regions.back();
                        double w = mesh->axis0->at(iright) - mesh->axis0->at(ileft);
                        double h = mesh->axis1->at(r) - mesh->axis1->at(r-1);
                        region.origin += Vec<2>(0., -h);
                        this->writelog(LOG_DETAIL, "Adding bottom cladding; h = {0}",h);
                        region.layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w, h), bottom_material));*/
                    } else
                        throw Exception("{0}: Right edge of the active region not aligned.", this->getId());
                }
                had_active |= active;
            }
        }
        in_active = had_active;

        // Now fill-in the layer info
        ActiveRegionInfo* region = regions.empty()? nullptr : &regions.back();
        if (region) {
            if (!added_bottom_cladding) {
                // add layer below active region (cladding) LUKASZ
                auto bottom_material = this->geometry->getMaterial(points->at(ileft,r-1));
                for (size_t cc = ileft; cc < iright; ++cc)
                    if (*this->geometry->getMaterial(points->at(cc,r-1)) != *bottom_material)
                        throw Exception("{0}: Material below active region not uniform.", this->getId());
                auto& region = regions.back();
                double w = mesh->axis0->at(iright) - mesh->axis0->at(ileft);
                double h = mesh->axis1->at(r) - mesh->axis1->at(r-1);
                region.origin += Vec<2>(0., -h);
                //this->writelog(LOG_DETAIL, "Adding bottom cladding; h = {0}",h);
                region.layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w, h), bottom_material));
                region.bottom = h;
                added_bottom_cladding = true;
            }

            double h = mesh->axis1->at(r+1) - mesh->axis1->at(r);
            double w = mesh->axis0->at(iright) - mesh->axis0->at(ileft);
            if (in_active) {
                size_t n = region->layers->getChildrenCount();
                shared_ptr<Block<2>> last;
                if (n > 0) last = static_pointer_cast<Block<2>>(static_pointer_cast<Translation<2>>(region->layers->getChildNo(n-1))->getChild());
                assert(!last || last->size.c0 == w);
                if (last && layer_material == last->getRepresentativeMaterial() && layer_QW == region->isQW(region->size()-1)) {
                    //TODO check if usage of getRepresentativeMaterial is fine here (was material)
                    last->setSize(w, last->size.c1 + h);
                } else {
                    auto layer = plask::make_shared<Block<2>>(Vec<2>(w,h), layer_material);
                    if (layer_QW) layer->addRole("QW");
                    region->layers->push_back(layer);
                }
            } else {
                if (!added_top_cladding) {

                    // add layer above active region (top cladding)
                    auto top_material = this->geometry->getMaterial(points->at(ileft,r));
                    for (size_t cc = ileft; cc < iright; ++cc)
                        if (*this->geometry->getMaterial(points->at(cc,r)) != *top_material)
                            throw Exception("{0}: Material above quantum well not uniform.", this->getId());
                    region->layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w,h), top_material));
                    //this->writelog(LOG_DETAIL, "Adding top cladding; h = {0}",h);

                    ileft = 0;
                    iright = points->axis0->size();
                    region->top = h;
                    added_top_cladding = true;
                }
            }
        }
    }
    if (!regions.empty() && regions.back().isQW(regions.back().size()-1))
        throw Exception("{0}: Quantum-well at the edge of the structure.", this->getId());

    if (strained && !materialSubstrate)
        throw BadInput(this->getId(), "Strained quantum wells requested but no layer with substrate role set");

    this->writelog(LOG_DETAIL, "Found {0} active region{1}", regions.size(), (regions.size()==1)?"":"s");
    for (auto& region: regions) region.summarize(this);
}

template <typename GeometryType>
void FreeCarrierGainSolver<GeometryType>::ActiveRegionInfo::summarize(const FreeCarrierGainSolver<GeometryType>* solver)
{
    holes = BOTH_HOLES;
    auto bbox = layers->getBoundingBox();
    total = bbox.upper[1] - bbox.lower[1] - bottom - top;
    materials.clear(); materials.reserve(layers->children.size());
    thicknesses.clear(); thicknesses.reserve(layers->children.size());
    for (const auto& layer: layers->children) {
        auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layer.get())->getChild().get());
        auto material = block->singleMaterial();
        if (!material) throw plask::Exception("{}: Active region can consist only of solid layers", solver->getId());
        auto bbox = static_cast<GeometryObjectD<2>*>(layer.get())->getBoundingBox();
        double thck = bbox.upper[1] - bbox.lower[1];
        materials.push_back(material);
        thicknesses.push_back(thck);
    }
    double substra = solver->strained? solver->materialSubstrate->lattC(solver->T0, 'a') : 0.;
    if (materials.size() > 2) {
        Material* material = materials[0].get();
        double e;
        if (solver->strained) { double latt = material->lattC(solver->T0, 'a'); e = (substra - latt) / latt; } else e = 0.;
        double el0 = material->CB(solver->T0, e, 'G'),
                hh0 = material->VB(solver->T0, e, 'G',  'H'),
                lh0 = material->VB(solver->T0, e, 'G',  'L');
        material = materials[1].get();
        if (solver->strained) { double latt = material->lattC(solver->T0, 'a'); e = (substra - latt) / latt; } else e = 0.;
        double el1 = material->CB(solver->T0, e, 'G'),
                hh1 = material->VB(solver->T0, e, 'G',  'H'),
                lh1 = material->VB(solver->T0, e, 'G',  'L');
        for (size_t i = 2; i < materials.size(); ++i) {
            material = materials[i].get();
            if (solver->strained) { double latt = material->lattC(solver->T0, 'a'); e = (substra - latt) / latt; } else e = 0.;
            double el2 = material->CB(solver->T0, e, 'G');
            double hh2 = material->VB(solver->T0, e, 'G',  'H');
            double lh2 = material->VB(solver->T0, e, 'G',  'L');
            if ((el0 < el1 && el1 > el2) || (hh0 > hh1 && hh1 < hh2) || (lh0 > lh1 && lh1 < lh2)) {
                if (i != 2 && i != materials.size()-1) {
                    bool eb = (el0 < el1 && el1 > el2);
                    if (eb != (hh0 > hh1 && hh1 < hh2)) holes = ConsideredHoles(holes & ~HEAVY_HOLES);
                    if (eb != (lh0 > lh1 && lh1 < lh2)) holes = ConsideredHoles(holes & ~LIGHT_HOLES);
                }
                if (holes == NO_HOLES)
                    throw Exception("{0}: Quantum wells in conduction band do not coincide with wells is valence band", solver->getId());
                if ((el0 < el1 && el1 > el2) || (hh0 > hh1 && hh1 < hh2 && holes & HEAVY_HOLES) || (lh0 > lh1 && lh1 < lh2 && holes & LIGHT_HOLES))
                    wells.push_back(i-1);
            } else if (i == 2) wells.push_back(0);
            if (el2 != el1) { el0 = el1; el1 = el2; }
            if (hh2 != hh1) { hh0 = hh1; hh1 = hh2; }
            if (lh2 != lh1) { lh0 = lh1; lh1 = lh2; }
        }
    }
    if (wells.back() < materials.size()-2) wells.push_back(materials.size()-1);
    totalqw = 0.;
    for (size_t i = 0; i < thicknesses.size(); ++i)
        if (isQW(i)) totalqw += thicknesses[i];
}


template <typename GeometryType>
FreeCarrierGainSolver<GeometryType>::ActiveRegionParams::ActiveRegionParams(const FreeCarrierGainSolver* solver, const ActiveRegionInfo& region, double T, bool quiet): region(region) {
    size_t n = region.materials.size();
    U[EL].reserve(n); U[HH].reserve(n); U[LH].reserve(n);
    M[EL].reserve(n); M[HH].reserve(n); M[LH].reserve(n);
    double substra = solver->strained? solver->materialSubstrate->lattC(T, 'a') : 0.;
    Eg = std::numeric_limits<double>::max();

    if (!solver->inBandEdges.hasProvider()) {
        solver->writelog(LOG_DETAIL, "Band edges taken from material database");
        size_t i = 0, mi;
        double me;
        for (auto material: region.materials) {
            OmpLockGuard<OmpNestLock> lockq = material->lock();
            double e; if (solver->strained) { double latt = material->lattC(T, 'a'); e = (substra - latt) / latt; } else e = 0.;
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
        if (solver->matrixelem != 0.) {
            Mt = solver->matrixelem;
        } else {
            double deltaSO = region.materials[mi]->Dso(T, me);
            Mt = (1./M[EL][mi].c11 - 1.) * (Eg + deltaSO) * Eg / (Eg + 0.666666666666667*deltaSO) / 2.;
            if (!quiet) solver->writelog(LOG_DETAIL, "Estimated momentum matrix element to {:.2f} eV m0", Mt);
        }
    } else {
        solver->writelog(LOG_DETAIL, "Band edges taken from inBandEdges receiver");
        //TODO
    }
}


template <typename GeometryType>
double FreeCarrierGainSolver<GeometryType>::level(WhichLevel which, double E, const ActiveRegionParams& params,
                                                  size_t start, size_t stop) const
{
    size_t nA = 2 * (stop - start + 1);

    DgbMatrix A(nA);

    constexpr double fact = 2e-12 * phys::me / (phys::hb_eV*phys::hb_J);

    double m1 = params.M[which][start].c11;
    double k1_2 = fact * m1 * (E - params.U[which][start]); if (which != EL) k1_2 = -k1_2;
    double k1 = sqrt(abs(k1_2));

    // Wave functions are confined, so we can assume exponentially decreasing relation in the outer layers
    A(0, 0) = A(nA-1, nA-1) = 1.;
    A(0, 1) = A(nA-1, nA-2) = 0.;

    for (size_t i = start, o = 1; i < stop; i++, o+=2) {
        double k0_2 = k1_2, k0 = k1, m0 = m1;
        double d = (o == 1)? 0. : params.region.thicknesses[i];
        if (k0_2 >= 0.) {
            double coskd = cos(k0*d), sinkd = sin(k0*d);
            A(o,   o-1) =  coskd;
            A(o+1, o-1) = -sinkd;
            A(o,   o  ) =  sinkd;
            A(o+1, o  ) =  coskd;
        } else {
            double phi = exp(-k0*d);
            A(o,   o-1) =  phi;
            A(o+1, o-1) = -phi;
            A(o,   o  ) = 1./phi;
            A(o+1, o  ) = 1./phi;
        }

        A(o+2, o  ) = 0.;
        A(o-1, o+1) = 0.;

        m1 = params.M[which][i+1].c11;
        k1_2 = fact * m1 * (E - params.U[which][i+1]); if (which != EL) k1_2 = -k1_2;
        if (k1_2 >= 0.) {
            k1 = sqrt(k1_2);
            A(o,   o+1) = -1.;
            A(o+1, o+1) =  0.;
            A(o,   o+2) =  0.;
            A(o+1, o+2) = -(k1 * m0) / (k0 * m1);
        } else {
            k1 = sqrt(-k1_2);
            double f = (k1 * m0) / (k0 * m1);
            A(o,   o+1) = -1.;
            A(o+1, o+1) =  f;
            A(o,   o+2) = -1.;
            A(o+1, o+2) = -f;
        }
    }

    return A.determinant();
}


template <typename GeometryType>
void FreeCarrierGainSolver<GeometryType>::estimateWellLevels(WhichLevel which, ActiveRegionParams& params, size_t qw) const
{
    if (params.U[which].size() < 3) return;

    size_t start = params.region.wells[qw], stop = params.region.wells[qw+1];
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
            if (ub < umin) { umin = ub; M = m; }
        } else {
            if (ub > umax) { umax = ub; M = m; }
        }
        if (i != start && i != stop) {
            double no = 1e-6 / M_PI * params.region.thicknesses[i] * sqrt(2. * phys::me / (phys::hb_eV*phys::hb_J) * m.c11);
            num = max(no, num);
        }
    }
    if (which == EL) umax = min(ustart, ustop);
    else umin = max(ustart, ustop);
    if (umax < umin)
        throw Exception("{}: Outer layers of active region have wrong band offset", this->getId()); //TODO make clearer
    num = 2. * ceil(sqrt(umax-umin)*num); // 2.* is the simplest way to ensure that all levels are found
    umin += 0.5 * levelsep;
    umax -= 0.5 * levelsep;
    double step = (umax-umin) / num;
    size_t n = size_t(num);
    double a, b = umin;
    double fa, fb = level(which, b, params, qw);
    if (fb == 0.) {
        params.levels[which].emplace_back(fb, M, which, params);
        b += levelsep; fb = level(which, b, params, qw);
    }
    for (size_t i = 0; i < n; ++i) {
        a = b; fa = fb;
        b = a + step; fb = level(which, b, params, qw);
        if (fb == 0.) {
            params.levels[which].emplace_back(fb, M, which, params);
            continue;
        }
        if ((fa < 0.) != (fb < 0.)) {
            boost::uintmax_t iters = 1000;
            double xa, xb;
            std::tie(xa, xb) = toms748_solve(
                [&](double x){ return level(which, x, params, qw); },
                a, b, fa, fb, [this](double l, double r){ return r-l < levelsep; }, iters)
            ;
            if (xb - xa > levelsep)
                throw ComputationError(this->getId(), "Could not find level estimate in quantum well");
            params.levels[which].emplace_back(0.5*(xa+xb), M, which, params);
        }
    }
}


template <typename GeometryType>
void FreeCarrierGainSolver<GeometryType>::estimateAboveLevels(WhichLevel which, ActiveRegionParams& params) const
{
    if (params.U[which].size() < 5) return; // This makes sense with at least two quantum wells

    /// Detect range above the wells
    size_t N = params.U[EL].size()-1;
    double umin = std::numeric_limits<double>::max(), umax = std::numeric_limits<double>::lowest();
    if (which == EL)
        umax = min(params.U[EL][0], params.U[EL][N]);
    else
        umin = max(params.U[which][0], params.U[which][params.U[which].size()-1]);
    Tensor2<double> M;
    for (size_t i: params.region.wells) {
        if (i == 0 || i == N) continue;
        double ub = params.U[which][i];
        if (which == EL) {
            if (ub < umin) { umin = ub; M = params.M[which][i]; }
        } else {
            if (ub > umax) { umax = ub; M = params.M[which][i]; }
        }
    }

    if (umax <= umin) return;

    double num = 2. * ceil(1e-6 / M_PI * params.region.total * sqrt(2. * (umax-umin) * phys::me / (phys::hb_eV*phys::hb_J) * M.c11));
    umin += 0.5 * levelsep;
    umax -= 0.5 * levelsep;
    double step = (umax-umin) / num;
    size_t n = size_t(num);
    double a, b = umin;
    double fa, fb = level(which, b, params);
    if (fb == 0.) {
        params.levels[which].emplace_back(fb, M, which, params);
        b += levelsep; fb = level(which, b, params);
    }
    for (size_t i = 0; i < n; ++i) {
        a = b; fa = fb;
        b = a + step; fb = level(which, b, params);
        if (fb == 0.) {
            params.levels[which].emplace_back(fb, M, which, params);
            continue;
        }
        if ((fa < 0.) != (fb < 0.)) {
            boost::uintmax_t iters = 1000;
            double xa, xb;
            std::tie(xa, xb) = toms748_solve(
                [&](double x){ return level(which, x, params); },
                a, b, fa, fb, [this](double l, double r){ return r-l < levelsep; }, iters)
            ;
            if (xb - xa > levelsep)
                throw ComputationError(this->getId(), "Could not find level estimate above quantum wells");
            params.levels[which].emplace_back(0.5*(xa+xb), M, which, params);
        }
    }
}

template <typename GeometryType>
void FreeCarrierGainSolver<GeometryType>::estimateLevels()
{
    params0.clear(); params0.reserve(regions.size());

    size_t reg = 0;
    for (const ActiveRegionInfo& region: regions) {
        params0.emplace_back(this, region);
        ActiveRegionParams& params = params0.back();
        for (size_t qw = 0; qw < region.wells.size()-1; ++qw) {
            estimateWellLevels(EL, params, qw);
            if (region.holes & ActiveRegionInfo::HEAVY_HOLES) estimateWellLevels(HH, params, qw); else params.levels[HH].clear();
            if (region.holes & ActiveRegionInfo::LIGHT_HOLES) estimateWellLevels(LH, params, qw); else params.levels[LH].clear();
        }
        std::sort(params.levels[EL].begin(), params.levels[EL].end(), [](const Level& a, const Level& b){return a.E < b.E;});
        std::sort(params.levels[HH].begin(), params.levels[HH].end(), [](const Level& a, const Level& b){return a.E > b.E;});
        std::sort(params.levels[LH].begin(), params.levels[LH].end(), [](const Level& a, const Level& b){return a.E > b.E;});
        params.nhh = std::min(params.levels[EL].size(), params.levels[HH].size());
        params.nlh = std::min(params.levels[EL].size(), params.levels[LH].size());
        estimateAboveLevels(EL, params);
        estimateAboveLevels(HH, params);
        estimateAboveLevels(LH, params);

        if (maxLoglevel > LOG_DETAIL) {
            {
                std::stringstream str; std::string sep = "";
                for (auto l: params.levels[EL]) { str << sep << format("{:.4f}", l.E); sep = ", "; }
                this->writelog(LOG_DETAIL, "Estimated electron levels for active region {:d} [eV]: {}", ++reg, str.str());
            }{
                std::stringstream str; std::string sep = "";
                for (auto l: params.levels[HH]) { str << sep << format("{:.4f}", l.E); sep = ", "; }
                this->writelog(LOG_DETAIL, "Estimated heavy hole levels for active region {:d} [eV]: {}", reg, str.str());
            }{
                std::stringstream str; std::string sep = "";
                for (auto l: params.levels[LH]) { str << sep << format("{:.4f}", l.E); sep = ", "; }
                this->writelog(LOG_DETAIL, "Estimated light hole levels for active region {:d} [eV]: {}", reg, str.str());
            }
        }
    }
}

template <typename GeometryT>
double FreeCarrierGainSolver<GeometryT>::getN(double F, double T, const ActiveRegionParams& params) const
{
    size_t n = params.levels[EL].size();
    const double kT = phys::kB_eV * T;
    constexpr double fact = phys::me * phys::kB_eV / (2.*M_PI * phys::hb_eV * phys::hb_J); // 1/µm (1e6) -> 1/cm³ (1e-6)

    double N = 2e-6 * pow(fact * T * params.sideM(EL).c00, 1.5) * fermiDiracHalf((F-params.sideU(EL))/kT);

    for (size_t i = 0; i < n; ++i) {
        double M = params.levels[EL][i].M.c00;
        N += 2. * fact * T * M / params.levels[EL][i].thickness * log(1 + exp((F-params.levels[EL][i].E)/kT));
    }

    return N;
}

template <typename GeometryT>
double FreeCarrierGainSolver<GeometryT>::getP(double F, double T, const ActiveRegionParams& params) const
{
    size_t nh = params.levels[HH].size(), nl = params.levels[LH].size();
    const double kT = phys::kB_eV * T;
    constexpr double fact = phys::me * phys::kB_eV / (2.*M_PI * phys::hb_eV * phys::hb_J); // 1/µm (1e6) -> 1/cm³ (1e-6)

    // Get parameters for outer layers
    double N = 2e-6 * (pow(fact * T * params.sideM(HH).c00, 1.5) * fermiDiracHalf((params.sideU(HH)-F)/kT) +
                       pow(fact * T * params.sideM(LH).c00, 1.5) * fermiDiracHalf((params.sideU(LH)-F)/kT));

    for (size_t i = 0; i < nh; ++i) {
        double M = params.levels[HH][i].M.c00;
        N += 2. * fact * T * M / params.levels[HH][i].thickness * log(1 + exp((params.levels[HH][i].E-F)/kT));
    }

    for (size_t i = 0; i < nl; ++i) {
        double M = params.levels[LH][i].M.c00;
        N += 2. * fact * T * M / params.levels[LH][i].thickness * log(1 + exp((params.levels[LH][i].E-F)/kT));
    }

    return N;
}

template <typename GeometryT>
void FreeCarrierGainSolver<GeometryT>::findFermiLevels(double& Fc, double& Fv, double n, double T,
                                                       const ActiveRegionParams& params) const
{
    double Ue = params.sideU(EL), Uh = params.sideU(HH);
    double fs = 0.05 * abs(Ue - Uh); if (fs <= levelsep) fs = 2. * levelsep;
    if (isnan(Fc)) Fc = Ue;
    if (isnan(Fv)) Fv = Uh;
    boost::uintmax_t iters;
    double xa, xb;

    iters = 1000;
    std::tie(xa, xb) = fermi_bracket_and_solve(
        [this,T,n,&params](double x){ return getN(x, T, params) - n; },
        Fc, fs, iters)
    ;
    if (xb - xa > levelsep)
        throw ComputationError(this->getId(), "Could not find quasi-Fermi level for electrons");
    Fc = 0.5 * (xa + xb);

    iters = 1000;
    std::tie(xa, xb) = fermi_bracket_and_solve(
        [this,T,n,&params](double x){ return getP(x, T, params) - n; },
        Fv, fs, iters)
    ;
    if (xb - xa > levelsep)
        throw ComputationError(this->getId(), "Could not find quasi-Fermi level for holes");
    Fv = 0.5 * (xa + xb);
}


template <typename GeometryT>
double FreeCarrierGainSolver<GeometryT>::getGain0(double hw, double Fc, double Fv, double T, double nr,
                                                  const ActiveRegionParams& params) const
{
    constexpr double fac = 1e4 * phys::qe*phys::qe / (2. * phys::c * phys::epsilon0 * phys::hb_J); // 1e4: 1/µm -> 1/cm
    const double ikT = (1./phys::kB_eV) / T;
    const double Dlt = 2. * (hw - params.Eg);

    double g = 0.;

    for (size_t i = 0; i < params.nhh; ++i) {
        const double Ec = params.levels[EL][i].E, Ev = params.levels[HH][i].E;
        const double Ep = hw - (Ec - Ev);
        if (Ep < 0.) continue;
        const double pp = 1. - ((Dlt > 0.)? Ep / Dlt : 0.);
        const double mu = 1. / (1. / params.levels[EL][i].M.c00 + 1. / params.levels[HH][i].M.c00);
        const double Ecp = Ec + Ep * mu / params.levels[EL][i].M.c00, Evp = Ev - Ep * mu / params.levels[HH][i].M.c00;
        g += mu * pp * (1. / (exp(ikT*(Ecp-Fc)) + 1) - 1. / (exp(ikT*(Evp-Fv)) + 1));
    }

    for (size_t i = 0; i < params.nlh; ++i) {
        const double Ec = params.levels[EL][i].E, Ev = params.levels[LH][i].E;
        const double Ep = hw - (Ec - Ev);
        if (Ep < 0.) continue;
        const double pp = 0.3333333333333333333333 + ((Dlt > 0.)? Ep / Dlt : 0.);
        const double mu = 1. / (1. / params.levels[EL][i].M.c00 + 1. / params.levels[LH][i].M.c00);
        const double Ecp = Ec + Ep * mu / params.levels[EL][i].M.c00, Evp = Ev - Ep * mu / params.levels[LH][i].M.c00;
        g += mu * pp * (1. / (exp(ikT*(Ecp-Fc)) + 1) - 1. / (exp(ikT*(Evp-Fv)) + 1));
    }
    return fac / (hw * nr * params.region.totalqw) * params.Mt * g;
}

template <typename GeometryT>
double FreeCarrierGainSolver<GeometryT>::getGain(double hw, double Fc, double Fv, double T, double nr,
                                                 const ActiveRegionParams& params) const
{
    if (lifetime == 0)
        return getGain0(hw, Fc, Fv, T, nr, params);

    const double E0 = params.levels[EL][0].E - ((params.region.holes == ActiveRegionInfo::BOTH_HOLES)? std::max(params.levels[HH][0].E, params.levels[LH][0].E) :
                                                (params.region.holes == ActiveRegionInfo::HEAVY_HOLES)? params.levels[HH][0].E : params.levels[LH][0].E);

    const double b = 1e12*phys::hb_eV / lifetime;
    const double tmax = 32. * b;
    const double tmin = std::max(-tmax, E0-hw);
    double dt = (tmax-tmin) / 1024.; //TODO Estimate integral precision and maybe chose better integration

    double g = 0.;
    for (double t = tmin; t <= tmax; t += dt) {
        // L(t) = b / (π (x²+b²)),
        g += getGain0(hw+t, Fc, Fv, T, nr, params) / (t*t + b*b);
    }
    g *= b * dt / M_PI;

    return g;
}

static const shared_ptr<OrderedAxis> zero_axis(new OrderedAxis({0.}));

/// Base for lazy data implementation
template <typename GeometryT>
struct FreeCarrierGainSolver<GeometryT>::DataBase: public LazyDataImpl<double>
{
    struct AveragedData {
        shared_ptr<const RectangularMesh<2>> mesh;
        LazyData<double> data;
        double factor;
        const FreeCarrierGainSolver<GeometryT>* solver;
        const char* name;
        AveragedData(const FreeCarrierGainSolver<GeometryT>* solver, const char* name,
                     const shared_ptr<const MeshAxis>& haxis, const ActiveRegionInfo& region):
            solver(solver), name(name)
        {
            auto vaxis = plask::make_shared<OrderedAxis>();
            OrderedAxis::WarningOff vaxiswoff(vaxis);
            for(size_t n = 0; n != region.size(); ++n) {
                if (region.isQW(n)) {
                    auto box = region.getLayerBox(n);
                    vaxis->addPoint(0.5 * (box.lower.c1 + box.upper.c1));
                }
            }
            mesh = plask::make_shared<const RectangularMesh<2>>(const_pointer_cast<MeshAxis>(haxis),
                                                                vaxis, RectangularMesh<2>::ORDER_01);
            factor = 1. / vaxis->size();
        }
        size_t size() const { return mesh->axis0->size(); }
        double operator[](size_t i) const {
            double val = 0.;
            for (size_t j = 0; j != mesh->axis1->size(); ++j) {
                double v = data[mesh->index(i,j)];
                if (isnan(v))
                    throw ComputationError(solver->getId(), "Wrong {0} ({1}) at {2}", name, v, mesh->at(i,j));
                v = max(v, 1e-6); // To avoid hangs
                val += v;
            }
            return val * factor;
        }
    };

    typedef typename FreeCarrierGainSolver<GeometryT>::ActiveRegionParams ActiveRegionParams;

    FreeCarrierGainSolver<GeometryT>* solver;           ///< Solver
    std::vector<shared_ptr<MeshAxis>> regpoints; ///< Points in each active region
    std::vector<LazyData<double>> data;                 ///< Computed interpolations in each active region
    shared_ptr<const MeshD<2>> dest_mesh;               ///< Destination mesh

    void setupFromAxis(const shared_ptr<MeshAxis>& axis) {
        regpoints.reserve(solver->regions.size());
        InterpolationFlags flags(solver->geometry);
        for (size_t r = 0; r != solver->regions.size(); ++r) {
            std::set<double> pts;
            auto box = solver->regions[r].getBoundingBox();
            double y = 0.5 * (box.lower.c1 + box.upper.c1);
            for (double x: *axis) {
                auto p = flags.wrap(vec(x,y));
                if (solver->regions[r].contains(p)) pts.insert(p.c0);
            }
            auto msh = plask::make_shared<OrderedAxis>();
            OrderedAxis::WarningOff mshw(msh);            ;
            msh->addOrderedPoints(pts.begin(), pts.end(), pts.size());
            regpoints.emplace_back(std::move(msh));
        }
    }

    DataBase(FreeCarrierGainSolver<GeometryT>* solver, const shared_ptr<const MeshD<2>>& dst_mesh):
        solver(solver), dest_mesh(dst_mesh)
    {
        // Create horizontal points lists
        if (solver->mesh) {
            setupFromAxis(solver->mesh);
        } else if (auto rect_mesh = dynamic_pointer_cast<const RectangularMesh<2>>(dst_mesh)) {
            setupFromAxis(rect_mesh->axis0);
        } else {
            regpoints.reserve(solver->regions.size());
            InterpolationFlags flags(solver->geometry);
            for (size_t r = 0; r != solver->regions.size(); ++r) {
                std::set<double> pts;
                for (auto point: *dest_mesh) {
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

    void compute(double wavelength, InterpolationMethod interp)
    {
        double hw = phys::h_eVc1e9 / wavelength;
        // Compute gains on mesh for each active region
        data.resize(solver->regions.size());
        for (size_t reg = 0; reg != solver->regions.size(); ++reg) {
            if (regpoints[reg]->size() == 0) {
                data[reg] = LazyData<double>(dest_mesh->size(), 0.);
                continue;
            }
            DataVector<double> values(regpoints[reg]->size());
            AveragedData temps(solver, "temperature", regpoints[reg], solver->regions[reg]);
            AveragedData concs(temps); concs.name = "carriers concentration";
            temps.data = solver->inTemperature(temps.mesh, interp);
            concs.data = solver->inCarriersConcentration(CarriersConcentration::PAIRS, temps.mesh, interp);
            std::exception_ptr error;
            #pragma omp parallel for
            for (plask::openmp_size_t i = 0; i < regpoints[reg]->size(); ++i) {
                if (error) continue;
                try {
                    double T = temps[i];
                    double nr = solver->regions[reg].averageNr(wavelength, T, concs[i]);
                    ActiveRegionParams params(solver, solver->params0[reg], T, bool(i));
                    values[i] = getValue(hw, concs[i], T, nr, params);
                } catch(...) {
                    #pragma omp critical
                    error = std::current_exception();
                }
            }
            if (error) std::rethrow_exception(error);
            data[reg] = interpolate(plask::make_shared<RectangularMesh<2>>(regpoints[reg], zero_axis),
                                    values, dest_mesh, interp, solver->geometry);
        }
    }

    virtual double getValue(double hw, double conc, double T, double nr, const ActiveRegionParams& params) = 0;

    size_t size() const override { return dest_mesh->size(); }

    double at(size_t i) const override {
        for (size_t reg = 0; reg != solver->regions.size(); ++reg)
            if (solver->regions[reg].inQW(dest_mesh->at(i)))
                return data[reg][i];
        return 0.;
    }
};

template <typename GeometryT>
struct FreeCarrierGainSolver<GeometryT>::GainData: public FreeCarrierGainSolver<GeometryT>::DataBase
{
    template <typename... Args>
    GainData(Args... args): DataBase(args...) {}

    double getValue(double hw, double conc, double T, double nr, const ActiveRegionParams& params) override
    {
        double Fc = NAN, Fv = NAN;
        this->solver->findFermiLevels(Fc, Fv, conc, T, params);
        return this->solver->getGain(hw, Fc, Fv, T, nr, params);
    }
};

template <typename GeometryT>
struct FreeCarrierGainSolver<GeometryT>::DgdnData: public FreeCarrierGainSolver<GeometryT>::DataBase
{
    template <typename... Args>
    DgdnData(Args... args): DataBase(args...) {}

    double getValue(double hw, double conc, double T, double nr, const ActiveRegionParams& params) override
    {
        const double h = 0.5 * DIFF_STEP;
        double Fc = NAN, Fv = NAN;
        this->solver->findFermiLevels(Fc, Fv, (1.-h)*conc, T, params);
        double gain1 = this->solver->getGain(hw, Fc, Fv, T, nr, params);
        this->solver->findFermiLevels(Fc, Fv, (1.+h)*conc, T, params);
        double gain2 = this->solver->getGain(hw, Fc, Fv, T, nr, params);
        return (gain2 - gain1) / (2.*h*conc);
    }
};



template <typename GeometryType>
const LazyData<double> FreeCarrierGainSolver<GeometryType>::getGainData(Gain::EnumType what, const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    if (what == Gain::GAIN) {
        this->writelog(LOG_DETAIL, "Calculating gain");
        this->initCalculation(); // This must be called before any calculation!
        GainData* data = new GainData(this, dst_mesh);
        data->compute(wavelength, getInterpolationMethod<INTERPOLATION_SPLINE>(interp));
        return LazyData<double>(data);
    } else if (what == Gain::DGDN) {
        this->writelog(LOG_DETAIL, "Calculating gain over carriers concentration derivative");
        this->initCalculation(); // This must be called before any calculation!
        DgdnData* data = new DgdnData(this, dst_mesh);
        data->compute(wavelength, getInterpolationMethod<INTERPOLATION_SPLINE>(interp));
        return LazyData<double>(data);
    } else {
        throw BadInput(this->getId(), "Wrong gain type requested");
    }
}


template <typename GeometryType>
shared_ptr<GainSpectrum<GeometryType>> FreeCarrierGainSolver<GeometryType>::getGainSpectrum(const Vec<2>& point)
{
    this->initCalculation();
    return make_shared<GainSpectrum<GeometryType>>(this, point);
}

// // template <typename GeometryType>
// // const EnergyLevels FreeCarrierGainSolver<GeometryType>::getEnergyLevels(size_t num)
// // {
// //     this->initCalculation();
// // }


template <> std::string FreeCarrierGainSolver<Geometry2DCartesian>::getClassName() const { return "gain.FreeCarrier2D"; }
template <> std::string FreeCarrierGainSolver<Geometry2DCylindrical>::getClassName() const { return "gain.FreeCarrierCyl"; }

template struct PLASK_SOLVER_API FreeCarrierGainSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API FreeCarrierGainSolver<Geometry2DCylindrical>;

}}} // namespace
