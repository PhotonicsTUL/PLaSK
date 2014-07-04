#include "fermi.h"

namespace plask { namespace solvers { namespace fermi {

template <typename GeometryType>
FermiGainSolver<GeometryType>::FermiGainSolver(const std::string& name): SolverWithMesh<GeometryType, RectangularMesh<1>>(name),
    outGain(this, &FermiGainSolver<GeometryType>::getGain),
    outGainOverCarriersConcentration(this, &FermiGainSolver<GeometryType>::getdGdn)// getDelegated will be called whether provider value is requested
{
    inTemperature = 300.; // temperature receiver has some sensible value
    lifetime = 0.1; // [ps]
    matrixelem = 0.0;
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
            std::unique_ptr<double[]> el(new double[std::distance(elt.begin(), elt.end())+1]);
            std::unique_ptr<double[]> hh(new double[std::distance(hht.begin(), hht.end())+1]);
            std::unique_ptr<double[]> lh(new double[std::distance(lht.begin(), lht.end())+1]);
            double* e = el.get(); for (const auto& i: elt) *(e++) = - boost::lexical_cast<double>(i); *e = 1.;
            double* h = hh.get(); for (const auto& i: hht) *(h++) = - boost::lexical_cast<double>(i); *h = 1.;
            double* l = lh.get(); for (const auto& i: lht) *(l++) = - boost::lexical_cast<double>(i); *l = 1.;
            if (extern_levels) {
                delete[] extern_levels->el; delete[] extern_levels->hh; delete[] extern_levels->lh;
            }
            extern_levels.reset(QW::ExternalLevels(el.release(), hh.release(), lh.release()));
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

    shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(this->geometry->getChild());
    shared_ptr<RectangularMesh<2>> points = mesh->getMidpointsMesh();

    size_t ileft = 0, iright = points->axis0->size();
    bool in_active = false;

    for (size_t r = 0; r < points->axis1->size(); ++r)
    {
        bool had_active = false; // indicates if we had active region in this layer
        shared_ptr<Material> layer_material;
        bool layer_QW = false;

        for (size_t c = 0; c < points->axis0->size(); ++c)
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
                            double w = mesh->axis0->at(iright) - mesh->axis0->at(ileft);
                            double h = mesh->axis1->at(r) - mesh->axis1->at(r-1);
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
            double h = mesh->axis1->at(r+1) - mesh->axis1->at(r);
            double w = mesh->axis0->at(iright) - mesh->axis0->at(ileft);
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
                iright = points->axis0->size();
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

    if (n == 0) n = 1e-6; // To avoid hangs

    gainModule.Set_temperature(T);
    gainModule.Set_koncentr(n);

    double qstrain = 0.; // strain in well
    double bstrain = 0.; // strain in barrier

    if (if_strain)
    {
        if (!this->materialSubstrate) throw ComputationError(this->getId(), "No layer with role 'substrate' has been found");

        qstrain = (this->materialSubstrate->lattC(T,'a') - region.materialQW->lattC(T,'a')) / region.materialQW->lattC(T,'a');
        bstrain = (this->materialSubstrate->lattC(T,'a') - region.materialBarrier->lattC(T,'a')) / region.materialBarrier->lattC(T,'a');
        qstrain *= 1.;
        bstrain *= 1.;
        //writelog(LOG_RESULT, "Strain in QW: %1%", qstrain);
        //writelog(LOG_RESULT, "Strain in B: %1%", bstrain);
    }

    //writelog(LOG_RESULT, "latt const for QW: %1%", region.materialQW->lattC(T,'a'));
    //writelog(LOG_RESULT, "latt const for subs: %1%", materialSubstrate->lattC(T,'a'));
    //writelog(LOG_RESULT, "latt const for barr: %1%", region.materialBarrier->lattC(T,'a'));

    Tensor2<double> qme, qmhh, qmlh, bme, bmhh, bmlh;
    double qEc, qEvhh, qEvlh, bEc, bEvhh, bEvlh, qEg, vhhdepth, vlhdepth, cdepth, vdepth;
    double qEgG, qEgX, qEgL, bEgG, bEgX, bEgL; // qw and barrier energy gaps

    std::exception_ptr error;
    #pragma omp critical // necessary as the material may be defined in Python
    {
        try {
            qme = region.materialQW->Me(T,qstrain);
            qmhh = region.materialQW->Mhh(T,qstrain);
            qmlh = region.materialQW->Mlh(T,qstrain);
            bme = region.materialBarrier->Me(T,bstrain);
            bmhh = region.materialBarrier->Mhh(T,bstrain);
            bmlh = region.materialBarrier->Mlh(T,bstrain);

            gainModule.Set_refr_index(region.materialQW->nr(wavelength, T));
            gainModule.Set_split_off(region.materialQW->Dso(T,qstrain));

            qEgG = region.materialQW->Eg(T,0.,'G');
            qEgX = region.materialQW->Eg(T,0.,'X');
            qEgL = region.materialQW->Eg(T,0.,'L');
            bEgG = region.materialBarrier->Eg(T,0.,'G');
            bEgX = region.materialBarrier->Eg(T,0.,'X');
            bEgL = region.materialBarrier->Eg(T,0.,'L');

            qEc = region.materialQW->CB(T,qstrain);
            qEvhh = region.materialQW->VB(T,qstrain,'G','H');
            qEvlh = region.materialQW->VB(T,qstrain,'G','L');
            bEc = region.materialBarrier->CB(T,bstrain);
            bEvhh = region.materialBarrier->VB(T,bstrain,'G','H');
            bEvlh = region.materialBarrier->VB(T,bstrain,'G','L');
        } catch (...) {
            error = std::current_exception();
        }
    }
    if (error) std::rethrow_exception(error);

    // TODO co robic w ponizszych przypadkach? - poprawic jak bedzie wiadomo
    if ((qEgG>=qEgX) && (qEgX)) this->writelog(LOG_DETAIL, "indirect Eg for QW: Eg,G = %1% eV, Eg,X = %2% eV; using Eg,G in calculations", qEgG, qEgX);
    if ((qEgG>=qEgL) && (qEgL)) this->writelog(LOG_DETAIL, "indirect Eg for QW: Eg,G = %1% eV, Eg,L = %2% eV; using Eg,G in calculations", qEgG, qEgL);
    if ((bEgG>=bEgX) && (bEgX)) this->writelog(LOG_DETAIL, "indirect Eg for barrier: Eg,G = %1% eV, Eg,X = %2% eV; using Eg,G in calculations", bEgG, bEgX);
    if ((bEgG>=bEgL) && (bEgL)) this->writelog(LOG_DETAIL, "indirect Eg for barrier: Eg,G = %1% eV, Eg,L = %2% eV; using Eg,G in calculations", bEgG, bEgL);

    if (qEc<qEvhh) throw ComputationError(this->getId(), "QW CB = %1% eV is below VB for heavy holes = %2% eV", qEc, qEvhh);
    if (qEc<qEvlh) throw ComputationError(this->getId(), "QW CB = %1% eV is below VB for light holes = %2% eV", qEc, qEvlh);
    if (bEc<bEvhh) throw ComputationError(this->getId(), "Barrier CB = %1% eV is below VB for heavy holes = %2% eV", bEc, bEvhh);
    if (bEc<bEvlh) throw ComputationError(this->getId(), "Barrier CB = %1% eV is below VB for light holes = %2% eV", bEc, bEvlh);

    gainModule.Set_electron_mass_in_plain(qme.c00);
    gainModule.Set_electron_mass_transverse(qme.c11);
    gainModule.Set_heavy_hole_mass_in_plain(qmhh.c00);
    gainModule.Set_heavy_hole_mass_transverse(qmhh.c11);
    gainModule.Set_light_hole_mass_in_plain(qmlh.c00);
    gainModule.Set_light_hole_mass_transverse(qmlh.c11);
    gainModule.Set_electron_mass_in_barrier(bme.c00);
    gainModule.Set_heavy_hole_mass_in_barrier(bmhh.c00);
    gainModule.Set_light_hole_mass_in_barrier(bmlh.c00);
    gainModule.Set_well_width(region.qwlen); //powinno byc - szerokosc pojedynczej studni
    gainModule.Set_waveguide_width(region.totallen);
    gainModule.Set_lifetime(lifetime);

    gainModule.Set_cond_waveguide_depth(cond_waveguide_depth);
    gainModule.Set_vale_waveguide_depth(vale_waveguide_depth);

    //writelog(LOG_RESULT, "qwlen: %1%", region.qwlen); // TEST
    //writelog(LOG_RESULT, "totallen: %1%", region.totallen); // TEST

    qEg = qEc-qEvhh;
    cdepth = bEc - qEc;
    vhhdepth = qEvhh-bEvhh;
    vlhdepth = qEvlh-bEvlh;
    vdepth = vhhdepth;

    if ((vhhdepth < 0.)&&(vlhdepth < 0.)) {
        std::string qname, bname;
        #pragma omp critical
        {
            try {
                qname = region.materialQW->name();
                bname = region.materialBarrier->name();
            } catch (...) {
                error = std::current_exception();
            }
        }
        if (error) std::rethrow_exception(error);
        throw BadInput(this->getId(), "Valence QW depth negative both for hh and lh, check VB values of materials %1% and %2%", qname, bname);
    }

    if (cdepth < 0.) {
        std::string qname, bname;
        #pragma omp critical
        {
            try {
                qname = region.materialQW->name();
                bname = region.materialBarrier->name();
            } catch (...) {
                error = std::current_exception();
            }
        }
        if (error) std::rethrow_exception(error);
        throw BadInput(this->getId(), "Conduction QW depth negative, check CB values of materials %1% and %2%", qname, bname);
    }

    gainModule.Set_conduction_depth(cdepth);

    if (if_strain == true)
    {
        // compute levels
        if (extern_levels)
            gainModule.przygoblALL(*extern_levels, gainModule.przel_dlug_z_angstr(region.qwtotallen));// earlier: qwtotallen
        else
        {
            gainModule.przygoblE();
            /*for (int i=0; i<gainModule.Get_number_of_electron_levels(); ++i) // TEST
                writelog(LOG_RESULT, "el_lev: %1%", gainModule.Get_electron_level_depth(i));*/ // TEST

            gainModule.Set_valence_depth(vhhdepth);
            gainModule.przygoblHH();
            /*for (int i=0; i<gainModule.Get_number_of_heavy_hole_levels(); ++i) // TEST
                writelog(LOG_RESULT, "hh_lev: %1%", gainModule.Get_heavy_hole_level_depth(i));*/ // TEST
            if (bstrain<0.)
            {
                std::vector<double> tLevHH;
                tLevHH.clear();
                double tDelEv = bEvhh-bEvlh;
                for (int i=0; i<gainModule.Get_number_of_heavy_hole_levels(); ++i)
                    tLevHH.push_back(gainModule.Get_heavy_hole_level_depth(i)+tDelEv);
                gainModule.przygoblHHc(tLevHH);
                /*for (int i=0; i<gainModule.Get_number_of_heavy_hole_levels(); ++i) // TEST
                    writelog(LOG_RESULT, "hh_lev_corr: %1%", gainModule.Get_heavy_hole_level_depth(i));*/ // TEST
            }

            gainModule.Set_valence_depth(vlhdepth);
            gainModule.przygoblLH();
            /*for (int i=0; i<gainModule.Get_number_of_light_hole_levels(); ++i) // TEST
                    writelog(LOG_RESULT, "lh_lev: %1%", gainModule.Get_light_hole_level_depth(i));*/ // TEST
            if (bstrain>0.)
            {
                std::vector<double> tLevLH;
                tLevLH.clear();
                double tDelEv = bEvhh-bEvlh;
                for (int i=0; i<gainModule.Get_number_of_light_hole_levels(); ++i)
                    tLevLH.push_back(gainModule.Get_light_hole_level_depth(i)-tDelEv);
                gainModule.przygoblLHc(tLevLH);
                /*for (int i=0; i<gainModule.Get_number_of_light_hole_levels(); ++i) // TEST
                        writelog(LOG_RESULT, "lh_lev_corr: %1%", gainModule.Get_light_hole_level_depth(i));*/ // TEST

            }
        }

        if (qstrain<=0.)
            qEg = qEc-qEvhh;
        else
            qEg = qEc-qEvlh;

        if ( (qstrain==0.) && (bstrain==0.) )
            vdepth = vhhdepth;
        else if ( (qstrain<0.) && (bstrain==0.) )
            vdepth = vhhdepth;
        else if ( (qstrain>0.) && (bstrain==0.) )
            vdepth = vlhdepth;
        else if ( (qstrain==0.) && (bstrain<0.) )
            vdepth = vlhdepth;
        else if ( (qstrain<0.) && (bstrain<0.) )
            vdepth = qEvhh-bEvlh;
        else if ( (qstrain>0.) && (bstrain<0.) )
            vdepth = vlhdepth;
        else if ( (qstrain==0.) && (bstrain>0.) )
            vdepth = vhhdepth;
        else if ( (qstrain<0.) && (bstrain>0.) )
            vdepth = vhhdepth;
        else if ( (qstrain>0.) && (bstrain>0.) )
            vdepth = qEvlh-bEvhh;
    }

    gainModule.Set_bandgap(qEg);
    gainModule.Set_valence_depth(vdepth);

    if (if_strain == true)
    {
         gainModule.przygoblQFL(region.qwlen); // earlier: qwtotallen
         //gainModule.przygoblQFL(region.qwtotallen); old line - nice way to change quasi-Fermi levels when another well is added - do not follow this way in old-gain model;-)
    }
    else
    {
        // compute levels
        if (extern_levels)
            gainModule.przygobl_n(*extern_levels, gainModule.przel_dlug_z_angstr(region.qwtotallen)); // earlier: qwtotallen
        else
            gainModule.przygobl_n(gainModule.przel_dlug_z_angstr(region.qwlen)); // earlier: qwtotallen
    }

    //writelog(LOG_RESULT, "matrix element: %1%", gainModule.Get_momentum_matrix_element()); // TEST;

    return gainModule;
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


/// Base for lazy data implementation
template <typename GeometryT>
struct FermiGainSolver<GeometryT>::DataBase: public LazyDataImpl<double>
{
    struct AveragedData {
        shared_ptr<const RectangularMesh<2>> mesh;
        LazyData<double> data;
        double factor;
        const FermiGainSolver<GeometryT>* solver;
        const char* name;
        AveragedData(const FermiGainSolver<GeometryT>* solver, const char* name,
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
                auto v = data[mesh->index(i,j)];
                if (isnan(v) || v < 0)
                    throw ComputationError(solver->getId(), "Wrong %1% (%2%) at %3%", name, v, mesh->at(i,j));
                val += v;
            }
            return val * factor;
        }
    };

    FermiGainSolver<GeometryT>* solver;                 ///< Solver
    std::vector<shared_ptr<RectangularAxis>> regpoints; ///< Points in each active region
    std::vector<LazyData<double>> data;                 ///< Computed interpolations in each active region
    shared_ptr<const MeshD<2>> dst_mesh;                ///< Destination mesh

    DataBase(FermiGainSolver<GeometryT>* solver, const shared_ptr<const MeshD<2>>& dst_mesh):
        solver(solver), dst_mesh(dst_mesh)
    {
        // Create horizontal points lists
        if (solver->mesh) {
            regpoints.assign(solver->regions.size(), solver->mesh);
        } else if (auto rect_mesh = dynamic_pointer_cast<const RectangularMesh<2>>(dst_mesh)) {
            regpoints.reserve(solver->regions.size());
            for (size_t r = 0; r != solver->regions.size(); ++r) {
                std::set<double> pts;
                auto box = solver->regions[r].getBoundingBox();
                double y = 0.5 * (box.lower.c1 + box.upper.c1);
                for (auto x: *rect_mesh->axis0) {
                    if (solver->regions[r].contains(vec(x,y))) pts.insert(x);
                }
                auto msh = make_shared<OrderedAxis>();
                msh->addOrderedPoints(pts.begin(), pts.end(), pts.size());
                regpoints.emplace_back(std::move(msh));
            }
        } else {
            regpoints.reserve(solver->regions.size());
            for (size_t r = 0; r != solver->regions.size(); ++r) {
                std::set<double> pts;
                for (auto point: *dst_mesh) {
                    if (solver->regions[r].contains(point)) pts.insert(point.c0);
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
        shared_ptr<OrderedAxis> zero(new OrderedAxis({0}));
        for (size_t reg = 0; reg != solver->regions.size(); ++reg)
        {
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
                    values[i] = getValue(wavelength, temps[i], concs[i], solver->regions[reg]);
                } catch(...) {
                    #pragma omp critical
                    error = std::current_exception();
                }
            }
            if (error) std::rethrow_exception(error);
            data[reg] = interpolate(make_shared<RectangularMesh<2>>(regpoints[reg], zero),
                                    values, dst_mesh, interp);
        }
    }

    virtual double getValue(double wavelength, double temp, double conc, const ActiveRegionInfo& region) = 0;

    size_t size() const override { return dst_mesh->size(); }

    double at(size_t i) const override {
        for (size_t reg = 0; reg != solver->regions.size(); ++reg)
            if (solver->regions[reg].inQW(dst_mesh->at(i)))
                return data[reg][i];
        return 0.;
    }
};

template <typename GeometryT>
struct FermiGainSolver<GeometryT>::GainData: public FermiGainSolver<GeometryT>::DataBase
{
    template <typename... Args>
    GainData(Args... args): DataBase(args...) {}

    double getValue(double wavelength, double temp, double conc, const ActiveRegionInfo& region) override
    {
        //if (isnan(conc) || (conc<0.)) return 0.; // odkomentowac poki sa problemy z liczeniem dyfuzji obok obszaru czynnego
        QW::gain gainModule = this->solver->getGainModule(wavelength, temp, conc, region);
        double len = (this->solver->extern_levels)? region.qwtotallen : region.qwlen;
        return gainModule.Get_gain_at_n(this->solver->nm_to_eV(wavelength), len); // earlier: qwtotallen
    }
};

template <typename GeometryT>
struct FermiGainSolver<GeometryT>::DgdnData: public FermiGainSolver<GeometryT>::DataBase
{
    template <typename... Args>
    DgdnData(Args... args): DataBase(args...) {}

    double getValue(double wavelength, double temp, double conc, const ActiveRegionInfo& region) override
    {
        double len = region.qwlen;
        if (this->solver->extern_levels) len = region.qwtotallen;
        double h = 0.5*this->solver->differenceQuotient;
        double gain1 =
            this->solver->getGainModule(wavelength, temp, (1.-h)*conc, region)
                .Get_gain_at_n(this->solver->nm_to_eV(wavelength), len); // earlier: qwtotallen
        double gain2 =
            this->solver->getGainModule(wavelength, temp, (1.+h)*conc, region)
                .Get_gain_at_n(this->solver->nm_to_eV(wavelength), len); // earlier: qwtotallen
        return (gain2 - gain1) / (2.*h*conc);
    }
};



template <typename GeometryType>
const LazyData<double> FermiGainSolver<GeometryType>::getGain(const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    this->writelog(LOG_DETAIL, "Calculating gain");
    this->initCalculation(); // This must be called before any calculation!

    GainData* data = new GainData(this, dst_mesh);
    data->compute(wavelength, defInterpolation<INTERPOLATION_SPLINE>(interp));

    return LazyData<double>(data);
}


template <typename GeometryType>
const LazyData<double> FermiGainSolver<GeometryType>::getdGdn(const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    this->writelog(LOG_DETAIL, "Calculating gain over carriers concentration first derivative");
    this->initCalculation(); // This must be called before any calculation!

    DgdnData* data = new DgdnData(this, dst_mesh);
    data->compute(wavelength, defInterpolation<INTERPOLATION_SPLINE>(interp));

    return LazyData<double>(data);
}


template <typename GeometryType>
GainSpectrum<GeometryType> FermiGainSolver<GeometryType>::getGainSpectrum(const Vec<2>& point)
{
    this->initCalculation();
    return GainSpectrum<GeometryType>(this, point);
}


template <> std::string FermiGainSolver<Geometry2DCartesian>::getClassName() const { return "gain.Fermi2D"; }
template <> std::string FermiGainSolver<Geometry2DCylindrical>::getClassName() const { return "gain.FermiCyl"; }

template struct PLASK_SOLVER_API FermiGainSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API FermiGainSolver<Geometry2DCylindrical>;

}}} // namespace
