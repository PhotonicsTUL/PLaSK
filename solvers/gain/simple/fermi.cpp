#include "fermi.h"

namespace plask { namespace solvers { namespace fermi {

template <typename GeometryType>
FermiGainSolver<GeometryType>::FermiGainSolver(const std::string& name): SolverWithMesh<GeometryType,RectilinearMesh1D>(name),
    outGain(this, &FermiGainSolver<GeometryType>::getGain),
    outGainOverCarriersConcentration(this, &FermiGainSolver<GeometryType>::getdGdn)// getDelegated will be called whether provider value is requested
{
    inTemperature = 300.; // temperature receiver has some sensible value
    lifetime = 0.1; // [ps]
    matrixelem = 10.0;
    cond_waveguide_depth = 0.26; // [eV]
    vale_waveguide_depth = 0.13; // [eV]
    differenceQuotient = 0.01;  // [%]
    if_strain = false;
    inTemperature.changedConnectMethod(this, &FermiGainSolver<GeometryType>::onInputChange);
    inCarriersConcentration.changedConnectMethod(this, &FermiGainSolver<GeometryType>::onInputChange);
}

template <typename GeometryType>
FermiGainSolver<GeometryType>::~FermiGainSolver() {
    inTemperature.changedDisconnectMethod(this, &FermiGainSolver<GeometryType>::onInputChange);
    inCarriersConcentration.changedDisconnectMethod(this, &FermiGainSolver<GeometryType>::onInputChange);
    if (extern_levels) {
        delete[] extern_levels->el;
        delete[] extern_levels->hh;
        delete[] extern_levels->lh;
    }
}


template <typename GeometryType>
void FermiGainSolver<GeometryType>::loadConfiguration(XMLReader& reader, Manager& manager)
{
    // Load a configuration parameter from XML.
    while (reader.requireTagOrEnd())
    {
        std::string param = reader.getNodeName();
        if (param == "config") {
            lifetime = reader.getAttribute<double>("lifetime", lifetime);
            matrixelem = reader.getAttribute<double>("matrix-elem", matrixelem);
            if_strain = reader.getAttribute<bool>("strained", if_strain);
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
            double *el = nullptr, *hh = nullptr, *lh = nullptr;
            try {
                el = new double[std::distance(elt.begin(), elt.end())+1];
                hh = new double[std::distance(hht.begin(), hht.end())+1];
                lh = new double[std::distance(lht.begin(), lht.end())+1];
                double* e = el; for (const auto& i: elt) *(e++) = - boost::lexical_cast<double>(i); *e = 1.;
                double* h = hh; for (const auto& i: hht) *(h++) = - boost::lexical_cast<double>(i); *h = 1.;
                double* l = lh; for (const auto& i: lht) *(l++) = - boost::lexical_cast<double>(i); *l = 1.;
            } catch(...) {
                delete[] el; delete[] hh; delete[] lh;
            }
            if (extern_levels) {
                delete[] extern_levels->el; delete[] extern_levels->hh; delete[] extern_levels->lh;
            }
            extern_levels.reset(QW::ExternalLevels(el, hh, lh));
        } else
            this->parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <levels>, or <config>");
    }
}


template <typename GeometryType>
void FermiGainSolver<GeometryType>::onInitialize() // In this function check if geometry and mesh are set
{
    if (!this->geometry) throw NoGeometryException(this->getId());

    detectActiveRegions();

    //TODO

    outGain.fireChanged();
}


template <typename GeometryType>
void FermiGainSolver<GeometryType>::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
{
    //TODO (if needed)
}

//template <typename GeometryType>
//void FermiGainSolver<GeometryType>::compute()
//{
//    this->initCalculation(); // This must be called before any calculation!
//}




template <typename GeometryType>
void FermiGainSolver<GeometryType>::detectActiveRegions()
{
    regions.clear();

    shared_ptr<RectilinearMesh2D> mesh = makeGeometryGrid(this->geometry->getChild());
    shared_ptr<RectilinearMesh2D> points = mesh->getMidpointsMesh();

    size_t ileft = 0, iright = points->axis0.size();
    bool in_active = false;

    for (size_t r = 0; r < points->axis1.size(); ++r)
    {
        bool had_active = false; // indicates if we had active region in this layer
        shared_ptr<Material> layer_material;
        bool layer_QW = false;

        for (size_t c = 0; c < points->axis0.size(); ++c)
        { // In the (possible) active region
            auto point = points->at(c,r);
            auto tags = this->geometry->getRolesAt(point);
            bool active = tags.find("active") != tags.end();
            bool QW = tags.find("QW") != tags.end()/* || tags.find("QD") != tags.end()*/;
            bool substrate = tags.find("substrate") != tags.end();

            if (substrate)
            {
                if (!materialSubstrate)
                    materialSubstrate = this->geometry->getMaterial(point);
                else if (*materialSubstrate != *this->geometry->getMaterial(point))
                    throw Exception("%1%: Non-uniform substrate layer.", this->getId());
            }

            if (QW && !active)
                throw Exception("%1%: All marked quantum wells must belong to marked active region.", this->getId());

            if (c < ileft)
            {
                if (active)
                    throw Exception("%1%: Left edge of the active region not aligned.", this->getId());
            }
            else if (c >= iright)
            {
                if (active)
                    throw Exception("%1%: Right edge of the active region not aligned.", this->getId());
            }
            else
            {
                // Here we are inside potential active region
                if (active)
                {
                    if (!had_active)
                    {
                        if (!in_active)
                        { // active region is starting set-up new region info
                            regions.emplace_back(mesh->at(c,r));
                            ileft = c;
                        }
                        layer_material = this->geometry->getMaterial(point);
                        layer_QW = QW;
                    }
                    else
                    {
                        if (*layer_material != *this->geometry->getMaterial(point))
                            throw Exception("%1%: Non-uniform active region layer.", this->getId());
                        if (layer_QW != QW)
                            throw Exception("%1%: Quantum-well role of the active region layer not consistent.", this->getId());
                    }
                }
                else if (had_active)
                {
                    if (!in_active)
                    {
                        iright = c;
                        if (layer_QW)
                        { // quantum well is at the egde of the active region, add one row below it
                            if (r == 0)
                                throw Exception("%1%: Quantum-well at the edge of the structure.", this->getId());
                            auto bottom_material = this->geometry->getMaterial(points->at(ileft,r-1));
                            for (size_t cc = ileft; cc < iright; ++cc)
                                if (*this->geometry->getMaterial(points->at(cc,r-1)) != *bottom_material)
                                    throw Exception("%1%: Material below quantum well not uniform.", this->getId());
                            auto& region = regions.back();
                            double w = mesh->axis0[iright] - mesh->axis0[ileft];
                            double h = mesh->axis1[r] - mesh->axis1[r-1];
                            region.origin += Vec<2>(0., -h);
                            region.layers->push_back(make_shared<Block<2>>(Vec<2>(w, h), bottom_material));
                        }
                    }
                    else
                        throw Exception("%1%: Right edge of the active region not aligned.", this->getId());
                }
                had_active |= active;
            }
        }
        in_active = had_active;

        // Now fill-in the layer info
        ActiveRegionInfo* region = regions.empty()? nullptr : &regions.back();
        if (region)
        {
            double h = mesh->axis1[r+1] - mesh->axis1[r];
            double w = mesh->axis0[iright] - mesh->axis0[ileft];
            if (in_active)
            {
                size_t n = region->layers->getChildrenCount();
                shared_ptr<Block<2>> last;
                if (n > 0) last = static_pointer_cast<Block<2>>(static_pointer_cast<Translation<2>>(region->layers->getChildNo(n-1))->getChild());
                assert(!last || last->size.c0 == w);
                if (last && layer_material == last->getRepresentativeMaterial() && layer_QW == region->isQW(region->size()-1))  //TODO check if usage of getRepresentativeMaterial is fine here (was material)
                {
                    last->setSize(w, last->size.c1 + h);
                }
                else
                {
                    auto layer = make_shared<Block<2>>(Vec<2>(w,h), layer_material);
                    if (layer_QW) layer->addRole("QW");
                    region->layers->push_back(layer);
                }
            }
            else
            {
                if (region->isQW(region->size()-1))
                { // top layer of the active region is quantum well, add the next layer
                    auto top_material = this->geometry->getMaterial(points->at(ileft,r));
                    for (size_t cc = ileft; cc < iright; ++cc)
                        if (*this->geometry->getMaterial(points->at(cc,r)) != *top_material)
                            throw Exception("%1%: Material above quantum well not uniform.", this->getId());
                    region->layers->push_back(make_shared<Block<2>>(Vec<2>(w,h), top_material));
                }
                ileft = 0;
                iright = points->axis0.size();
            }
        }
    }
    if (!regions.empty() && regions.back().isQW(regions.back().size()-1))
        throw Exception("%1%: Quantum-well at the edge of the structure.", this->getId());

    this->writelog(LOG_DETAIL, "Found %1% active region%2%", regions.size(), (regions.size()==1)?"":"s");
    size_t n = 0;
    for (auto& region: regions) {
        region.summarize(this);
        this->writelog(LOG_DETAIL, "Active region %1%: %2%nm single QW, %3%nm all QW, %4%nm total",
                       n++, 0.1*region.qwlen, 0.1*region.qwtotallen, 0.1*region.totallen);
    }
}


template <typename GeometryType>
QW::gain FermiGainSolver<GeometryType>::getGainModule(double wavelength, double T, double n, const ActiveRegionInfo& region)
{
    QW::gain gainModule;

    if (isnan(n) || n < 0) throw ComputationError(this->getId(), "Wrong carriers concentration (%1%/cm3)", n);

    gainModule.Set_temperature(T);
    gainModule.Set_koncentr(n);

    double qstrain = 0.; // strain in well
    double bstrain = 0.; // strain in barrier

    if (if_strain == true)
    {
        //qstrain = (region.materialQW->lattC(T,'a') - this->materialSubstrate->lattC(T,'a')) / region.materialQW->lattC(T,'a');
        //bstrain = (region.materialBarrier->lattC(T,'a') - this->materialSubstrate->lattC(T,'a')) / region.materialBarrier->lattC(T,'a');
        qstrain = bstrain = 0.;
        writelog(LOG_RESULT, "Strain in QW: %1%", qstrain);
        writelog(LOG_RESULT, "Strain in B: %1%", bstrain);
    }
    //write_debug("a_QW: %1%", region.materialQW->lattC(T,'a'));
    //write_debug("a_sub: %1%", materialSubstrate->lattC(T,'a'));
    //write_debug("strain: %1%", qstrain);
    //writelog(LOG_RESULT, "latt const for QW: %1%", region.materialQW->lattC(T,'a'));
    //writelog(LOG_RESULT, "latt const for subs: %1%", materialSubstrate->lattC(T,'a'));
    //writelog(LOG_RESULT, "latt const for barr: %1%", region.materialBarrier->lattC(T,'a'));

    Tensor2<double> qme, qmhh, qmlh, bme, bmhh, bmlh;
    double qEc, qEvhh, qEvlh, bEc, bEvhh, bEvlh, qEg, vhhdepth, vlhdepth, cdepth, vdepth;

    #pragma omp critical // necessary as the material may be defined in Python
    {
        qme = region.materialQW->Me(T,qstrain);
        qmhh = region.materialQW->Mhh(T,qstrain);
        qmlh = region.materialQW->Mlh(T,qstrain);
        bme = region.materialBarrier->Me(T,bstrain);
        bmhh = region.materialBarrier->Mhh(T,bstrain);
        bmlh = region.materialBarrier->Mlh(T,bstrain);

        gainModule.Set_refr_index(region.materialQW->nr(wavelength, T));
        gainModule.Set_split_off(region.materialQW->Dso(T,qstrain));

        qEc = region.materialQW->CB(T,qstrain);
        qEvhh = region.materialQW->VB(T,qstrain,'G','H');
        qEvlh = region.materialQW->VB(T,qstrain,'G','L');
        bEc = region.materialBarrier->CB(T,bstrain);
        bEvhh = region.materialBarrier->VB(T,bstrain,'G','H');
        bEvlh = region.materialBarrier->VB(T,bstrain,'G','L');
        // to be continued...
    }

    gainModule.Set_electron_mass_in_plain(qme.c00);
    gainModule.Set_electron_mass_transverse(qme.c11);
    gainModule.Set_heavy_hole_mass_in_plain(qmhh.c00);
    gainModule.Set_heavy_hole_mass_transverse(qmhh.c11);
    gainModule.Set_light_hole_mass_in_plain(qmlh.c00);
    gainModule.Set_light_hole_mass_transverse(qmlh.c11);
    gainModule.Set_electron_mass_in_barrier(bme.c00);
    gainModule.Set_heavy_hole_mass_in_barrier(bmhh.c00);
    gainModule.Set_light_hole_mass_in_barrier(bmlh.c00);
    gainModule.Set_well_width(region.qwlen);
    gainModule.Set_waveguide_width(region.totallen);
    gainModule.Set_lifetime(lifetime);

    gainModule.Set_cond_waveguide_depth(cond_waveguide_depth);
    gainModule.Set_vale_waveguide_depth(vale_waveguide_depth);

    qEg = qEc-qEvhh;
    cdepth = bEc - qEc;
    vhhdepth = qEvhh-bEvhh;
    vlhdepth = qEvlh-bEvlh;
    vdepth = vhhdepth; // it will be changed

    if ((vhhdepth < 0.)&&(vlhdepth < 0.)) {
        std::string qname, bname;
        #pragma omp critical
        {
            qname = region.materialQW->name();
            bname = region.materialBarrier->name();
        }
        throw BadInput(this->getId(), "Valence QW depth negative both for hh and lh, check VB values of materials %1% and %2%", qname, bname);
    }

    if (cdepth < 0.) {
        std::string qname, bname;
        #pragma omp critical
        {
            qname = region.materialQW->name();
            bname = region.materialBarrier->name();
        }
        throw BadInput(this->getId(), "Conduction QW depth negative, check CB values of materials %1% and %2%", qname, bname);
    }

    gainModule.Set_conduction_depth(cdepth);

    if (if_strain == true)
    {
        // compute levels
        if (extern_levels)
            gainModule.przygoblALL(*extern_levels, gainModule.przel_dlug_z_angstr(region.qwtotallen));
        else
        {
            gainModule.przygoblE();
            if (vhhdepth > 0.)
            {
                gainModule.Set_valence_depth(vhhdepth);
                gainModule.przygoblHH();
            }
            if (vlhdepth > 0.)
            {
                gainModule.Set_valence_depth(vlhdepth);
                gainModule.przygoblLH();
            }
        }
        if ( (qstrain==0.) && (bstrain==0.) )
        {
            qEg = qEc-qEvhh;
            vdepth = vhhdepth;
        }
        else if ( (qstrain<0.) && (bstrain==0.) )
        {
            qEg = qEc-qEvhh;
            vdepth = vhhdepth;
        }
        else if ( (qstrain>0.) && (bstrain==0.) )
        {
            qEg = qEc-qEvlh;
            vdepth = vlhdepth;
        }
    }

    gainModule.Set_bandgap(qEg);
    gainModule.Set_valence_depth(vdepth);
    gainModule.Set_momentum_matrix_element(matrixelem);

    if (if_strain == true)
    {
         //gainModule.przygoblQFL(gainModule.przel_dlug_z_angstr(region.qwtotallen));
         gainModule.przygoblQFL(region.qwtotallen);
         return gainModule;
    }
    else
    {
        // compute levels
        if (extern_levels)
            gainModule.przygobl_n(*extern_levels, gainModule.przel_dlug_z_angstr(region.qwtotallen));
        else
            gainModule.przygobl_n(gainModule.przel_dlug_z_angstr(region.qwtotallen));
        return gainModule;
    }
}

//  TODO: it should return computed levels
template <typename GeometryType>
/// Function computing energy levels
std::deque<std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,double,double>>
FermiGainSolver<GeometryType>::determineLevels(double T, double n)
{
    this->initCalculation(); // This must be called before any calculation!

    std::deque<std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,double,double>> result;

    if (regions.size() == 1)
        this->writelog(LOG_DETAIL, "Found 1 active region");
    else
        this->writelog(LOG_DETAIL, "Found %1% active regions", regions.size());

    for (int act=0; act<regions.size(); act++)
    {
        double qFlc, qFlv;
        std::vector<double> el, hh, lh;

        this->writelog(LOG_DETAIL, "Evaluating energy levels for active region nr %1%:", act+1);

        QW::gain gainModule = getGainModule(0.0, T, n, regions[act]); //wavelength=0.0 - no refractive index needs to be calculated (any will do)

        writelog(LOG_RESULT, "Conduction band quasi-Fermi level (from the band edge) = %1% eV", qFlc = gainModule.Get_qFlc());
        writelog(LOG_RESULT, "Valence band quasi-Fermi level (from the band edge) = %1% eV", qFlv = gainModule.Get_qFlv());

        int j = 0;
        double level;

        std::string levelsstr = "Electron energy levels (from the conduction band edge) [eV]: ";
        do
        {
            level = gainModule.Get_electron_level_depth(j);
            if (level > 0) {
                el.push_back(level);
                levelsstr += format("%1%, ", level);
            }
            j++;
        }
        while(level>0);
        writelog(LOG_RESULT, levelsstr.substr(0, levelsstr.length()-2));

        levelsstr = "Heavy hole energy levels (from the valence band edge) [eV]: ";
        j=0;
        do
        {
            level = gainModule.Get_heavy_hole_level_depth(j);
            if (level > 0) {
                hh.push_back(level);
                levelsstr += format("%1%, ", level);
            }
            j++;
        }
        while(level>0);
        writelog(LOG_RESULT, levelsstr.substr(0, levelsstr.length()-2));

        levelsstr = "Light hole energy levels (from the valence band edge) [eV]: ";
        j=0;
        do
        {
            level = gainModule.Get_light_hole_level_depth(j);
            if (level > 0) {
                lh.push_back(level);
                levelsstr += format("%1%, ", level);
            }
            j++;
        }
        while(level>0);
        writelog(LOG_RESULT, levelsstr.substr(0, levelsstr.length()-2));

        result.push_back(std::make_tuple(el, hh, lh, qFlc, qFlv));
    }
    return result;
}

template <typename GeometryType>
const DataVector<double> FermiGainSolver<GeometryType>::getGain(const MeshD<2>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    if (interp == INTERPOLATION_DEFAULT) interp = INTERPOLATION_SPLINE;

    this->writelog(LOG_INFO, "Calculating gain");
    this->initCalculation(); // This must be called before any calculation!

    RectilinearMesh2D mesh2;
    if (this->mesh) {
        RectilinearAxis verts;
        for (auto p: dst_mesh) verts.addPoint(p.vert());
        mesh2.axis0 = this->mesh->axis; mesh2.axis1 = verts;
    }
    const MeshD<2>& src_mesh = (this->mesh)? (const MeshD<2>&)mesh2 : dst_mesh;

    WrappedMesh<2> geo_mesh(src_mesh, this->geometry);

    DataVector<const double> nOnMesh = inCarriersConcentration(geo_mesh, interp); // carriers concentration on the mesh
    DataVector<const double> TOnMesh = inTemperature(geo_mesh, interp); // temperature on the mesh
    DataVector<double> gainOnMesh(geo_mesh.size(), 0.);

    std::vector<std::pair<size_t,size_t>> points;
    for (size_t i = 0; i != geo_mesh.size(); i++)
        for (size_t r = 0; r != regions.size(); ++r)
            if (regions[r].contains(geo_mesh[i]) && nOnMesh[i] > 0.)
                points.push_back(std::make_pair(i,r));

    #pragma omp parallel for
    for (int j = 0; j < points.size(); j++)
    {
        size_t i = points[j].first;
        const ActiveRegionInfo& region = regions[points[j].second];
        QW::gain gainModule = getGainModule(wavelength, TOnMesh[i], nOnMesh[i], region);
        gainOnMesh[i] = gainModule.Get_gain_at_n(nm_to_eV(wavelength), region.qwtotallen);
    }

    if (this->mesh) {
        WrappedMesh<2> geo_dst_mesh(dst_mesh, this->geometry);
        return interpolate(mesh2, gainOnMesh, geo_dst_mesh, interp);
    } else {
        return gainOnMesh;
    }
}


template <typename GeometryType>
const DataVector<double> FermiGainSolver<GeometryType>::getdGdn(const MeshD<2>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    if (interp == INTERPOLATION_DEFAULT) interp = INTERPOLATION_SPLINE;

    this->writelog(LOG_INFO, "Calculating gain over carriers concentration first derivative");
    this->initCalculation(); // This must be called before any calculation!

    RectilinearMesh2D mesh2;
    if (this->mesh) {
        RectilinearAxis verts;
        for (auto p: dst_mesh) verts.addPoint(p.vert());
        mesh2.axis0 = this->mesh->axis; mesh2.axis1 = verts;
    }
    const MeshD<2>& src_mesh = (this->mesh)? (const MeshD<2>&)mesh2 : dst_mesh;

    WrappedMesh<2> geo_mesh(src_mesh, this->geometry);

    DataVector<const double> nOnMesh = inCarriersConcentration(geo_mesh, interp); // carriers concentration on the mesh
    DataVector<const double> TOnMesh = inTemperature(geo_mesh, interp); // temperature on the mesh
    DataVector<double> dGdn(geo_mesh.size(), 0.);

    std::vector<std::pair<size_t,size_t>> points;
    for (size_t i = 0; i != geo_mesh.size(); i++)
        for (size_t r = 0; r != regions.size(); ++r)
            if (regions[r].contains(geo_mesh[i]) && nOnMesh[i] > 0.)
                points.push_back(std::make_pair(i,r));

    #pragma omp parallel for
    for (int j = 0; j < points.size(); j++)
    {
        size_t i = points[j].first;
        const ActiveRegionInfo& region = regions[points[j].second];
        double gainOnMesh1 = getGainModule(wavelength, TOnMesh[i], (1.-0.5*differenceQuotient) * nOnMesh[i], region)
            .Get_gain_at_n(nm_to_eV(wavelength), region.qwtotallen);
        double gainOnMesh2 = getGainModule(wavelength, TOnMesh[i], (1.+0.5*differenceQuotient) * nOnMesh[i], region)
            .Get_gain_at_n(nm_to_eV(wavelength), region.qwtotallen);
        dGdn[i] = (gainOnMesh2 - gainOnMesh1) / (differenceQuotient*nOnMesh[i]);
    }

    if (this->mesh) {
        WrappedMesh<2> geo_dst_mesh(dst_mesh, this->geometry);
        return interpolate(mesh2, dGdn, geo_dst_mesh, interp);
    } else {
        return dGdn;
    }
}


template <typename GeometryType>
GainSpectrum<GeometryType> FermiGainSolver<GeometryType>::getGainSpectrum(const Vec<2>& point)
{
    this->initCalculation();
    return GainSpectrum<GeometryType>(this, point);
}


template <> std::string FermiGainSolver<Geometry2DCartesian>::getClassName() const { return "gain.Fermi2D"; }
template <> std::string FermiGainSolver<Geometry2DCylindrical>::getClassName() const { return "gain.FermiCyl"; }

template struct FermiGainSolver<Geometry2DCartesian>;
template struct FermiGainSolver<Geometry2DCylindrical>;

}}} // namespace
