#include "gauss_matrix.h"
#include "fermigolden.h"
#include "fd.h"

#include <boost/math/tools/roots.hpp>
using boost::math::tools::toms748_solve;
using boost::math::tools::bracket_and_solve_root;


namespace plask { namespace gain { namespace fermigolden {

constexpr double DIFF_STEP = 0.001;


template <typename GeometryType>
FermiGoldenGainSolver<GeometryType>::FermiGoldenGainSolver(const std::string& name): SolverWithMesh<GeometryType, RectangularMesh<1>>(name),
    outGain(this, &FermiGoldenGainSolver<GeometryType>::getGain, []{return 2;}),
    lifetime(0.1),
    matrixelem(0.),
    T0(300.),
    levelsep(0.0001),
    strained(false),
    quick_levels(true)
{
    inTemperature = 300.;
    inTemperature.changedConnectMethod(this, &FermiGoldenGainSolver<GeometryType>::onInputChange);
    inCarriersConcentration.changedConnectMethod(this, &FermiGoldenGainSolver<GeometryType>::onInputChange);
}

template <typename GeometryType>
FermiGoldenGainSolver<GeometryType>::~FermiGoldenGainSolver() {
    inTemperature.changedDisconnectMethod(this, &FermiGoldenGainSolver<GeometryType>::onInputChange);
    inCarriersConcentration.changedDisconnectMethod(this, &FermiGoldenGainSolver<GeometryType>::onInputChange);
}


template <typename GeometryType>
void FermiGoldenGainSolver<GeometryType>::loadConfiguration(XMLReader& reader, Manager& manager)
{
    // Load a configuration parameter from XML.
    while (reader.requireTagOrEnd())
    {
        std::string param = reader.getNodeName();
        if (param == "config") {
            lifetime = reader.getAttribute<double>("lifetime", lifetime);
            matrixelem = reader.getAttribute<double>("matrix-elem", matrixelem);
            strained = reader.getAttribute<bool>("strained", strained);
            quick_levels = reader.getAttribute<bool>("quick-levels", quick_levels);
            reader.requireTagEnd();
        } else
            this->parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <levels>, or <config>");
    }
}


template <typename GeometryType>
void FermiGoldenGainSolver<GeometryType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    detectActiveRegions();
    estimateLevels();
    outGain.fireChanged();
}


template <typename GeometryType>
void FermiGoldenGainSolver<GeometryType>::onInvalidate()
{
    levels[EL].clear();
    levels[HH].clear();
    levels[LH].clear();
    regions.clear();
    materialSubstrate.reset();
}

//template <typename GeometryType>
//void FermiGoldenGainSolver<GeometryType>::compute()
//{
//    this->initCalculation(); // This must be called before any calculation!
//}


template <typename GeometryType>
void FermiGoldenGainSolver<GeometryType>::detectActiveRegions()
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
                    throw Exception("%1%: Non-uniform substrate layer.", this->getId());
            }

            if (QW && !active)
                throw Exception("%1%: All marked quantum wells must belong to marked active region.", this->getId());

            if (c < ileft) {
                if (active)
                    throw Exception("%1%: Left edge of the active region not aligned.", this->getId());
            } else if (c >= iright) {
                if (active)
                    throw Exception("%1%: Right edge of the active region not aligned.", this->getId());
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
                            throw Exception("%1%: Non-uniform active region layer.", this->getId());
                        if (layer_QW != QW)
                            throw Exception("%1%: Quantum-well role of the active region layer not consistent.", this->getId());
                    }
                } else if (had_active) {
                    if (!in_active) {
                        iright = c;

                        // add layer below active region (cladding) LUKASZ
                        /*auto bottom_material = this->geometry->getMaterial(points->at(ileft,r-1));
                        for (size_t cc = ileft; cc < iright; ++cc)
                            if (*this->geometry->getMaterial(points->at(cc,r-1)) != *bottom_material)
                                throw Exception("%1%: Material below quantum well not uniform.", this->getId());
                        auto& region = regions.back();
                        double w = mesh->axis0->at(iright) - mesh->axis0->at(ileft);
                        double h = mesh->axis1->at(r) - mesh->axis1->at(r-1);
                        region.origin += Vec<2>(0., -h);
                        this->writelog(LOG_DETAIL, "Adding bottom cladding; h = %1%",h);
                        region.layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w, h), bottom_material));*/
                    } else
                        throw Exception("%1%: Right edge of the active region not aligned.", this->getId());
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
                        throw Exception("%1%: Material below active region not uniform.", this->getId());
                auto& region = regions.back();
                double w = mesh->axis0->at(iright) - mesh->axis0->at(ileft);
                double h = mesh->axis1->at(r) - mesh->axis1->at(r-1);
                region.origin += Vec<2>(0., -h);
                //this->writelog(LOG_DETAIL, "Adding bottom cladding; h = %1%",h);
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
                            throw Exception("%1%: Material above quantum well not uniform.", this->getId());
                    region->layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w,h), top_material));
                    //this->writelog(LOG_DETAIL, "Adding top cladding; h = %1%",h);

                    ileft = 0;
                    iright = points->axis0->size();
                    region->top = h;
                    added_top_cladding = true;
                }
            }
        }
    }
    if (!regions.empty() && regions.back().isQW(regions.back().size()-1))
        throw Exception("%1%: Quantum-well at the edge of the structure.", this->getId());

    this->writelog(LOG_DETAIL, "Found %1% active region%2%", regions.size(), (regions.size()==1)?"":"s");
    for (auto& region: regions) region.summarize(this);

    if (strained && !materialSubstrate)
        throw BadInput(this->getId(), "Strained quantum wells requested but no layer with substrate role set");

}


template <typename GeometryType>
double FermiGoldenGainSolver<GeometryType>::level(WhichLevel which, double E, const ActiveRegionParams& params,
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
void FermiGoldenGainSolver<GeometryType>::estimateWellLevels(WhichLevel which, std::vector<Level>& levels, const ActiveRegionParams& params, size_t qw) const
{
    if (params.U[which].size() < 3) return;

    size_t start = params.region.wells[qw], stop = params.region.wells[qw+1];
    double umin = std::numeric_limits<double>::max(), umax = -std::numeric_limits<double>::max();
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
        throw Exception("%s: Outer layers of active region have wrong band offset", this->getId()); //TODO make clearer
    num = 2. * ceil(sqrt(umax-umin)*num); // 2.* is the simplest way to ensure that all levels are found
    umin += 0.5 * levelsep;
    umax -= 0.5 * levelsep;
    double step = (umax-umin) / num;
    size_t n = size_t(num);
    double a, b = umin;
    double fa, fb = level(which, b, params, qw);
    if (fb == 0.) {
        levels.emplace_back(fb, M, which, params);
        b += levelsep; fb = level(which, b, params, qw);
    }
    for (size_t i = 0; i < n; ++i) {
        a = b; fa = fb;
        b = a + step; fb = level(which, b, params, qw);
        if (fb == 0.) {
            levels.emplace_back(fb, M, which, params);
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
            levels.emplace_back(0.5*(xa+xb), M, which, params);
        }
    }
}


template <typename GeometryType>
void FermiGoldenGainSolver<GeometryType>::estimateAboveLevels(WhichLevel which, std::vector<Level>& levels, const ActiveRegionParams& params) const
{
    if (params.U[which].size() < 5) return; // This makes sense with at least two quantum wells

    /// Detect range above the wells
    size_t N = params.U[EL].size()-1;
    double umin = std::numeric_limits<double>::max(), umax = -std::numeric_limits<double>::max();
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
        levels.emplace_back(fb, M, which, params);
        b += levelsep; fb = level(which, b, params);
    }
    for (size_t i = 0; i < n; ++i) {
        a = b; fa = fb;
        b = a + step; fb = level(which, b, params);
        if (fb == 0.) {
            levels.emplace_back(fb, M, which, params);
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
            levels.emplace_back(0.5*(xa+xb), M, which, params);
        }
    }
}

template <typename GeometryType>
void FermiGoldenGainSolver<GeometryType>::estimateLevels()
{
    for (size_t i = 0; i < 3; ++i) {
        levels[i].clear(); levels[i].reserve(regions.size());
    }
    nlevels.clear(); nlevels.reserve(regions.size());
    params0.clear(); params0.reserve(regions.size());

    size_t reg = 0;
    for (const ActiveRegionInfo& region: regions) {
        std::vector<Level> lel, lhh, llh;
        params0.emplace_back(this, region, T0);
        for (size_t qw = 0; qw < region.wells.size()-1; ++qw) {
            estimateWellLevels(EL, lel, params0.back(), qw);
            estimateWellLevels(HH, lhh, params0.back(), qw);
            estimateWellLevels(LH, llh, params0.back(), qw);
        }
        std::sort(lel.begin(), lel.end(), [](const Level& a, const Level& b){return a.E < b.E;});
        std::sort(lhh.begin(), lhh.end(), [](const Level& a, const Level& b){return a.E > b.E;});
        std::sort(llh.begin(), llh.end(), [](const Level& a, const Level& b){return a.E > b.E;});
        nlevels.push_back(std::min(lel.size(), std::min(lhh.size(), llh.size())));
        estimateAboveLevels(EL, lel, params0.back());
        estimateAboveLevels(HH, lhh, params0.back());
        estimateAboveLevels(LH, llh, params0.back());

        levels[EL].emplace_back(std::move(lel));
        levels[HH].emplace_back(std::move(lhh));
        levels[LH].emplace_back(std::move(llh));

        if (maxLoglevel > LOG_DETAIL) {
            {
                std::stringstream str; std::string sep = "";
                for (auto l: levels[EL].back()) { str << sep << format("%.4f", l.E); sep = ", "; }
                this->writelog(LOG_DETAIL, "Estimated electron levels for active region %d [eV]: %s", ++reg, str.str());
            }{
                std::stringstream str; std::string sep = "";
                for (auto l: levels[HH].back()) { str << sep << format("%.4f", l.E); sep = ", "; }
                this->writelog(LOG_DETAIL, "Estimated heavy hole levels for active region %d [eV]: %s", reg, str.str());
            }{
                std::stringstream str; std::string sep = "";
                for (auto l: levels[LH].back()) { str << sep << format("%.4f", l.E); sep = ", "; }
                this->writelog(LOG_DETAIL, "Estimated light hole levels for active region %d [eV]: %s", reg, str.str());
            }
        }
    }
}

template <typename GeometryT>
double FermiGoldenGainSolver<GeometryT>::getN(double F, double T, size_t reg, const ActiveRegionParams& params) const {
    size_t n = levels[EL][reg].size();
    const double kT = phys::kB_eV * T;
    constexpr double fact = phys::me * phys::kB_eV / (2.*M_PI * phys::hb_eV * phys::hb_J);

    double N = 2e-6 * pow(fact * T * params.sideM(EL).c00, 1.5) * fermiDiracHalf((F-params.sideU(EL))/kT);

    for (size_t i = 0; i < n; ++i) {
        double M = levels[EL][reg][i].M.c00;
        N += 2. * fact * T * M / levels[EL][reg][i].thickness * log(1 + exp((F-levels[EL][reg][i].E)/kT)); // 1/µm (1e6) -> 1/cm³ (1e-6)
    }

    return N;
}

template <typename GeometryT>
double FermiGoldenGainSolver<GeometryT>::getP(double F, double T, size_t reg, const ActiveRegionParams& params) const {
    size_t nh = levels[HH][reg].size(), nl = levels[LH][reg].size();
    const double kT = phys::kB_eV * T;
    constexpr double fact = phys::me * phys::kB_eV / (2.*M_PI * phys::hb_eV * phys::hb_J);

    // Get parameters for outer layers
    double N = 2e-6 * (pow(fact * T * params.sideM(HH).c00, 1.5) * fermiDiracHalf((params.sideU(HH)-F)/kT) +
                       pow(fact * T * params.sideM(LH).c00, 1.5) * fermiDiracHalf((params.sideU(LH)-F)/kT));

    for (size_t i = 0; i < nh; ++i) {
        double M = levels[HH][reg][i].M.c00;
        N += 2. * fact * T * M / levels[HH][reg][i].thickness * log(1 + exp((levels[HH][reg][i].E-F)/kT)); // 1/µm (1e6) -> 1/cm³ (1e-6)
    }

    for (size_t i = 0; i < nl; ++i) {
        double M = levels[LH][reg][i].M.c00;
        N += 2. * fact * T * M / levels[LH][reg][i].thickness * log(1 + exp((levels[LH][reg][i].E-F)/kT)); // 1/µm (1e6) -> 1/cm³ (1e-6)
    }

    return N;
}

template <typename GeometryT>
void FermiGoldenGainSolver<GeometryT>::findFermiLevels(double& Fc, double& Fv, double n, double T, size_t reg) const
{
    ActiveRegionParams params(this, regions[reg], T);

    if (isnan(Fc)) Fc = params.sideU(EL);
    if (isnan(Fv)) Fv = params.sideU(HH);

    boost::uintmax_t iters;
    double xa, xb;

    iters = 1000;
    std::tie(xa, xb) = bracket_and_solve_root(
        [this,T,reg,n,&params](double x){ return getN(x, T, reg, params) - n; },
        Fc, 1.5, true,
        [this](double l, double r){ return r-l < levelsep; }, iters)
    ;
    if (xb - xa > levelsep)
        throw ComputationError(this->getId(), "Could not find quasi-Fermi for electrons");
    Fc = 0.5 * (xa + xb);

    iters = 1000;
    std::tie(xa, xb) = bracket_and_solve_root(
        [this,T,reg,n,&params](double x){ return getP(x, T, reg, params) - n; },
        Fv, 1.5, false,
        [this](double l, double r){ return r-l < levelsep; }, iters)
    ;
    if (xb - xa > levelsep)
        throw ComputationError(this->getId(), "Could not find quasi-Fermi for electrons");
    Fv = 0.5 * (xa + xb);
}




static const shared_ptr<OrderedAxis> zero_axis(new OrderedAxis({0.}));

/// Base for lazy data implementation
template <typename GeometryT>
struct FermiGoldenGainSolver<GeometryT>::DataBase: public LazyDataImpl<double>
{
    struct AveragedData {
        shared_ptr<const RectangularMesh<2>> mesh;
        LazyData<double> data;
        double factor;
        const FermiGoldenGainSolver<GeometryT>* solver;
        const char* name;
        AveragedData(const FermiGoldenGainSolver<GeometryT>* solver, const char* name,
                     const shared_ptr<const RectangularAxis>& haxis, const ActiveRegionInfo& region):
            solver(solver), name(name)
        {
            auto vaxis = plask::make_shared<OrderedAxis>();
            for(size_t n = 0; n != region.size(); ++n) {
                if (region.isQW(n)) {
                    auto box = region.getLayerBox(n);
                    vaxis->addPoint(0.5 * (box.lower.c1 + box.upper.c1));
                }
            }
            mesh = plask::make_shared<const RectangularMesh<2>>(const_pointer_cast<RectangularAxis>(haxis),
                                                         vaxis, RectangularMesh<2>::ORDER_01);
            factor = 1. / vaxis->size();
        }
        size_t size() const { return mesh->axis0->size(); }
        double operator[](size_t i) const {
            double val = 0.;
            for (size_t j = 0; j != mesh->axis1->size(); ++j) {
                double v = data[mesh->index(i,j)];
                if (isnan(v))
                    throw ComputationError(solver->getId(), "Wrong %1% (%2%) at %3%", name, v, mesh->at(i,j));
                v = max(v, 1e-6); // To avoid hangs
                val += v;
            }
            return val * factor;
        }
    };

    FermiGoldenGainSolver<GeometryT>* solver;           ///< Solver
    std::vector<shared_ptr<RectangularAxis>> regpoints; ///< Points in each active region
    std::vector<LazyData<double>> data;                 ///< Computed interpolations in each active region
    shared_ptr<const MeshD<2>> dest_mesh;               ///< Destination mesh

    void setupFromAxis(const shared_ptr<RectangularAxis>& axis) {
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
            msh->addOrderedPoints(pts.begin(), pts.end(), pts.size());
            regpoints.emplace_back(std::move(msh));
        }
    }

    DataBase(FermiGoldenGainSolver<GeometryT>* solver, const shared_ptr<const MeshD<2>>& dst_mesh):
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
                msh->addOrderedPoints(pts.begin(), pts.end(), pts.size());
                regpoints.emplace_back(std::move(msh));
            }
        }
    }

    void compute(double wavelength, InterpolationMethod interp)
    {
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
            concs.data = solver->inCarriersConcentration(temps.mesh, interp);
            std::exception_ptr error;
            #pragma omp parallel for
            for (size_t i = 0; i < regpoints[reg]->size(); ++i) {
                if (error) continue;
                try {
                    // Make concentration non-zero
                    values[i] = getValue(wavelength, temps[i], max(concs[i], 1e-9), solver->regions[reg]);
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

    virtual double getValue(double wavelength, double temp, double conc, const ActiveRegionInfo& region) = 0;

    size_t size() const override { return dest_mesh->size(); }

    double at(size_t i) const override {
        for (size_t reg = 0; reg != solver->regions.size(); ++reg)
            if (solver->regions[reg].inQW(dest_mesh->at(i)))
                return data[reg][i];
        return 0.;
    }
};

template <typename GeometryT>
struct FermiGoldenGainSolver<GeometryT>::GainData: public FermiGoldenGainSolver<GeometryT>::DataBase
{
    template <typename... Args>
    GainData(Args... args): DataBase(args...) {}

    double getValue(double wavelength, double temp, double conc, const ActiveRegionInfo& region) override
    {
//         QW::gain gainModule = this->solver->getGainModule(wavelength, temp, conc, region);
//         double thck = (this->solver->extern_levels)? region.qwtotal : region.qw;
//         return gainModule.Get_gain_at_n(this->solver->nm_to_eV(wavelength), thck); // earlier: qwtotal
    }
};

template <typename GeometryT>
struct FermiGoldenGainSolver<GeometryT>::DgdnData: public FermiGoldenGainSolver<GeometryT>::DataBase
{
    template <typename... Args>
    DgdnData(Args... args): DataBase(args...) {}

    double getValue(double wavelength, double temp, double conc, const ActiveRegionInfo& region) override
    {
//         double thck = region.qw;
//         if (this->solver->extern_levels) thck = region.qwtotal;
//         double h = 0.5 * DIFF_STEP;
//         double conc1, conc2;
//         conc1 = (1.-h) * conc;
//         conc2 = (1.+h) * conc;
//         double gain1 =
//             this->solver->getGainModule(wavelength, temp, conc1, region)
//                 .Get_gain_at_n(this->solver->nm_to_eV(wavelength), thck); // earlier: qwtotal
//         double gain2 =
//             this->solver->getGainModule(wavelength, temp, conc2, region)
//                 .Get_gain_at_n(this->solver->nm_to_eV(wavelength), thck); // earlier: qwtotal
//         return (gain2 - gain1) / (2.*h*conc);
    }
};



template <typename GeometryType>
const LazyData<double> FermiGoldenGainSolver<GeometryType>::getGain(Gain::EnumType what, const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    if (what == Gain::DGDN) {
        this->writelog(LOG_DETAIL, "Calculating gain over carriers concentration derivative");
        this->initCalculation(); // This must be called before any calculation!
        DgdnData* data = new DgdnData(this, dst_mesh);
        data->compute(wavelength, getInterpolationMethod<INTERPOLATION_SPLINE>(interp));
        return LazyData<double>(data);
    } else {
        this->writelog(LOG_DETAIL, "Calculating gain");
        this->initCalculation(); // This must be called before any calculation!
        GainData* data = new GainData(this, dst_mesh);
        data->compute(wavelength, getInterpolationMethod<INTERPOLATION_SPLINE>(interp));
        return LazyData<double>(data);
    }
}


template <typename GeometryType>
GainSpectrum<GeometryType> FermiGoldenGainSolver<GeometryType>::getGainSpectrum(const Vec<2>& point)
{
    this->initCalculation();
    return GainSpectrum<GeometryType>(this, point);
}


template <> std::string FermiGoldenGainSolver<Geometry2DCartesian>::getClassName() const { return "gain.Fermi2D"; }
template <> std::string FermiGoldenGainSolver<Geometry2DCylindrical>::getClassName() const { return "gain.FermiCyl"; }

template struct PLASK_SOLVER_API FermiGoldenGainSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API FermiGoldenGainSolver<Geometry2DCylindrical>;

}}} // namespace
