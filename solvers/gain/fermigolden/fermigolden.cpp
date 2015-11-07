#include "gauss_matrix.h"
#include "fermigolden.h"

namespace plask { namespace gain { namespace fermigolden {

constexpr double DIFF_STEP = 0.001;


template <typename GeometryType>
FermiGoldenGainSolver<GeometryType>::FermiGoldenGainSolver(const std::string& name): SolverWithMesh<GeometryType, RectangularMesh<1>>(name),
    outGain(this, &FermiGoldenGainSolver<GeometryType>::getGain, []{return 2;}),
    lifetime(0.1),
    matrixelem(0.),
    T0(300.),
    strained(false)
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
            reader.requireTagEnd();
        } else if (param == "levels") {
            std::string els, hhs, lhs;
            if (reader.hasAttribute("el") || reader.hasAttribute("hh") || reader.hasAttribute("lh")) {
                els = reader.requireAttribute("el");
                hhs = reader.requireAttribute("hh");
                lhs = reader.requireAttribute("lh");
                reader.requireTagEnd();
            } else {
                while (reader.requireTagOrEnd()) {
                    if (reader.getNodeName() == "el") els = reader.requireTextInCurrentTag();
                    else if (reader.getNodeName() == "hh") hhs = reader.requireTextInCurrentTag();
                    else if (reader.getNodeName() == "lh") lhs = reader.requireTextInCurrentTag();
                    else throw XMLUnexpectedElementException(reader, "<el>, <hh>, or <lh>");
                }
                if (els == "") throw XMLUnexpectedElementException(reader, "<el>");
                if (hhs == "") throw XMLUnexpectedElementException(reader, "<hh>");
                if (lhs == "") throw XMLUnexpectedElementException(reader, "<lh>");
            }
            boost::char_separator<char> sep(", ");
            boost::tokenizer<boost::char_separator<char>> elt(els, sep), hht(hhs, sep), lht(lhs, sep);
            levels_el.assign(1, std::vector<double>()); levels_el[0].reserve(std::distance(elt.begin(), elt.end()));
            for (const auto& i: elt) levels_el[0].push_back(boost::lexical_cast<double>(i));
            levels_hh.assign(1, std::vector<double>()); levels_hh[0].reserve(std::distance(hht.begin(), hht.end()));
            for (const auto& i: hht) levels_hh[0].push_back(boost::lexical_cast<double>(i));
            levels_lh.assign(1, std::vector<double>()); levels_lh[0].reserve(std::distance(lht.begin(), lht.end()));
            for (const auto& i: lht) levels_lh[0].push_back(boost::lexical_cast<double>(i));
            extern_levels = true;
        } else
            this->parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <levels>, or <config>");
    }
}


template <typename GeometryType>
void FermiGoldenGainSolver<GeometryType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    detectActiveRegions();
    if (!extern_levels) estimateLevels();
    outGain.fireChanged();
}


template <typename GeometryType>
void FermiGoldenGainSolver<GeometryType>::onInvalidate()
{
    levels_el.clear();
    levels_hh.clear();
    levels_lh.clear();
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
                        region.layers->push_back(make_shared<Block<2>>(Vec<2>(w, h), bottom_material));*/
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
                region.layers->push_back(make_shared<Block<2>>(Vec<2>(w, h), bottom_material));
                region.bottomlen = h;
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
                    auto layer = make_shared<Block<2>>(Vec<2>(w,h), layer_material);
                    if (layer_QW) layer->addRole("QW");
                    region->layers->push_back(layer);
                    //if (layer_QW) this->writelog(LOG_DETAIL, "Adding qw; h = %1%",h);
                    //else this->writelog(LOG_DETAIL, "Adding barrier; h = %1%",h);
                }
            } else {
                if (!added_top_cladding) {

                    // add layer above active region (top cladding)
                    auto top_material = this->geometry->getMaterial(points->at(ileft,r));
                    for (size_t cc = ileft; cc < iright; ++cc)
                        if (*this->geometry->getMaterial(points->at(cc,r)) != *top_material)
                            throw Exception("%1%: Material above quantum well not uniform.", this->getId());
                    region->layers->push_back(make_shared<Block<2>>(Vec<2>(w,h), top_material));
                    //this->writelog(LOG_DETAIL, "Adding top cladding; h = %1%",h);

                    ileft = 0;
                    iright = points->axis0->size();
                    region->toplen = h;
                    added_top_cladding = true;
                }
            }
        }
    }
    if (!regions.empty() && regions.back().isQW(regions.back().size()-1))
        throw Exception("%1%: Quantum-well at the edge of the structure.", this->getId());

    this->writelog(LOG_INFO, "Found %1% active region%2%", regions.size(), (regions.size()==1)?"":"s");
    size_t n = 0;
    for (auto& region: regions) {
        region.summarize(this);
        this->writelog(LOG_DETAIL, "Active region %d: %g nm QWs, %g nm total", n++, 0.1*region.qwtotallen, 0.1*region.totallen);
    }

    // energy levels for active region with two identcal QWs won't be calculated so QW widths must be changed
    this->writelog(LOG_DETAIL, "Updating QW widths");
    n = 0;
    for (auto& region: regions) {
        region.lens.clear();
        int N = region.size(); // number of all layers in the active region (QW, barr, external)
        int noOfQWs = 0; // number of QWs counter
        for (int i=0; i<N; ++i) {
            if (region.isQW(i)) noOfQWs++;
            region.lens.push_back(region.getLayerBox(i).height()*1e4); // in [A]
            this->writelog(LOG_DEBUG, "Layer %1% thickness: %2% nm", i+1, 0.1*region.lens[i]);
        }
        this->writelog(LOG_DEBUG, "Number of QWs in the above active region: %1%", noOfQWs);

        // if (adjust_widths) {
        //     double hstep = region.qwlen*roughness/qw_width_mod;
        //     if ( !(noOfQWs%2) ) {
        //         double h0 = region.qwlen-(int(noOfQWs/2))*hstep+0.5*hstep;
        //         for (int i=0; i<n; ++i) {
        //             if (region.isQW(i)) {
        //                 region.lens[i] = h0;
        //                 this->writelog(LOG_DEBUG, "Layer %1% thickness: %2% nm", i+1, 0.1*region.lens[i]);
        //                 h0 += hstep;
        //             }
        //         }
        //     } else {
        //         double h0 = region.qwlen-(int(noOfQWs/2))*hstep;
        //         for (int i=0; i<n; ++i) {
        //             if (region.isQW(i)) {
        //                 region.lens[i] = h0;
        //                 this->writelog(LOG_DETAIL, "Layer %1% modified thickness: %2% nm", i+1, 0.1*region.lens[i]);
        //                 h0 += hstep;
        //             }
        //         }
        //     }
        //     this->writelog(LOG_DETAIL, "QW thickness step: %1% nm", 0.1*hstep);
        // }
        n++;
    }
    
    if (strained && !materialSubstrate)
        throw BadInput(this->getId(), "Strained quantum wells requested but no layer with substrate role set");
}


template <typename GeometryType> template <typename FermiGoldenGainSolver<GeometryType>::WhichLevel which>
double FermiGoldenGainSolver<GeometryType>::level(const ActiveRegionInfo& region, double T, double E)
{
    size_t N = region.materials.size();
    size_t last = 2*N - 1;
    double substra = materialSubstrate->lattC(T, 'a');

    DgbMatrix A(2*N);

    double k1_2 = layerk2<which>(region, substra, T, E, 0);
    double k1 = sqrt(abs(k1_2));

    // Wave functions are confined, so we can assume exponentially decreasing relation in the outer layers
    A(0, 0) = A(last, last) = 1.;
    A(0, 1) = A(last, last-1) = 0.;

    for (size_t i = 0; i < N-1; ++i) {
        size_t o = 2*i + 1;

        double k0_2 = k1_2, k0 = k1;
        double d = region.lens[i];
        if (k0_2 >= 0.) {
            double coskd = cos(k0*d), sinkd = sin(k0*d);
            A(o,   o-1) = coskd;
            A(o+1, o-1) = -k0 * sinkd;
            A(o,   o  ) = sinkd;
            A(o+1, o  ) = k0 * coskd;
        } else {
            double phi = exp(-k0*d);
            A(o,   o-1) = phi;
            A(o+1, o-1) = -k0 * phi;
            A(o,   o  ) = 1. / phi;
            A(o+1, o  ) = k0 / phi;
        }

        A(o+2, o  ) = 0.;
        A(o-1, o+1) = 0.;

        k1_2 = layerk2<which>(region, substra, T, E, i+1);
        if (k1_2 >= 0.) {
            k1 = sqrt(k1_2);
            A(o,   o-1) = -1.;
            A(o+1, o-1) =  0.;
            A(o,   o  ) =  0;
            A(o+1, o  ) = -k1;
        } else {
            k1 = sqrt(-k1_2);
            A(o,   o-1) = -1.;
            A(o+1, o-1) =  k1;
            A(o,   o  ) = -1.;
            A(o+1, o  ) = -k1;
        }
    }

    return A.determinant();
}


template <typename GeometryType>
void FermiGoldenGainSolver<GeometryType>::estimateLevels()
{
    if (regions.size() == 1)
        this->writelog(LOG_DETAIL, "Found 1 active region");
    else
        this->writelog(LOG_DETAIL, "Found %1% active regions", regions.size());

    double substra = strained? materialSubstrate->lattC(T0, 'a') : 0.;

    levels_el.clear(); levels_el.reserve(regions.size());
    levels_hh.clear(); levels_hh.reserve(regions.size());
    levels_lh.clear(); levels_lh.reserve(regions.size());
    
    for (const ActiveRegionInfo& region: regions) {
        size_t N = region.materials.size();
        double cbmin = std::numeric_limits<double>::max(), cbmax = std::numeric_limits<double>::min();
        double vhmin = std::numeric_limits<double>::max(), vhmax = std::numeric_limits<double>::min();
        double vlmin = std::numeric_limits<double>::max(), vlmax = std::numeric_limits<double>::min();
        for (size_t i = 0; i < N; ++i) {
            double cb = layerUB<LEVEL_EC>(region, substra, T0, i);
            cbmax = max(cb, cbmax); cbmin = min(cb, cbmin);
            double vh = layerUB<LEVEL_HH>(region, substra, T0, i);
            vhmax = max(vh, vhmax); vhmin = min(vh, vhmin);
            double vl = layerUB<LEVEL_LH>(region, substra, T0, i);
            vlmax = max(vl, vlmax); vlmin = min(vl, vlmin);
#ifndef NDEBUG
            this->writelog(LOG_DEBUG, "Electron levels between %g and %g", cbmin, cbmax);
            this->writelog(LOG_DEBUG, "Heavy holes levels between %g and %g", vhmin, vhmax);
            this->writelog(LOG_DEBUG, "Light holes levels between %g and %g", vlmin, vlmax);
#endif
        }
    }
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
            auto vaxis = make_shared<OrderedAxis>();
            for(size_t n = 0; n != region.size(); ++n) {
                if (region.isQW(n)) {
                    auto box = region.getLayerBox(n);
                    vaxis->addPoint(0.5 * (box.lower.c1 + box.upper.c1));
                }
            }
            mesh = make_shared<const RectangularMesh<2>>(const_pointer_cast<RectangularAxis>(haxis),
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

    FermiGoldenGainSolver<GeometryT>* solver;                 ///< Solver
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
            auto msh = make_shared<OrderedAxis>();
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
                auto msh = make_shared<OrderedAxis>();
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
            data[reg] = interpolate(make_shared<RectangularMesh<2>>(regpoints[reg], zero_axis),
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
//         double len = (this->solver->extern_levels)? region.qwtotallen : region.qwlen;
//         return gainModule.Get_gain_at_n(this->solver->nm_to_eV(wavelength), len); // earlier: qwtotallen
    }
};

template <typename GeometryT>
struct FermiGoldenGainSolver<GeometryT>::DgdnData: public FermiGoldenGainSolver<GeometryT>::DataBase
{
    template <typename... Args>
    DgdnData(Args... args): DataBase(args...) {}

    double getValue(double wavelength, double temp, double conc, const ActiveRegionInfo& region) override
    {
//         double len = region.qwlen;
//         if (this->solver->extern_levels) len = region.qwtotallen;
//         double h = 0.5 * DIFF_STEP;
//         double conc1, conc2;
//         conc1 = (1.-h) * conc;
//         conc2 = (1.+h) * conc;
//         double gain1 =
//             this->solver->getGainModule(wavelength, temp, conc1, region)
//                 .Get_gain_at_n(this->solver->nm_to_eV(wavelength), len); // earlier: qwtotallen
//         double gain2 =
//             this->solver->getGainModule(wavelength, temp, conc2, region)
//                 .Get_gain_at_n(this->solver->nm_to_eV(wavelength), len); // earlier: qwtotallen
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
