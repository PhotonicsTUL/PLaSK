#include "ferminew.h"

namespace plask { namespace solvers { namespace FermiNew {

template <typename GeometryType>
FermiNewGainSolver<GeometryType>::FermiNewGainSolver(const std::string& name): SolverWithMesh<GeometryType,OrderedMesh1D>(name),
    outGain(this, &FermiNewGainSolver<GeometryType>::getGain),
    outLuminescence(this, &FermiNewGainSolver<GeometryType>::getLuminescence),
    outGainOverCarriersConcentration(this, &FermiNewGainSolver<GeometryType>::getDgDn)
{
    Tref = 300.; // [K], only for this temperature energy levels are calculated
    inTemperature = 300.; // temperature receiver has some sensible value
    cond_qw_shift = 0.; // [eV]
    vale_qw_shift = 0.; // [eV]
    qw_width_mod = 40.; // [-] (if equal to 10 - differences in QW widths are to big)
    roughness = 0.05; // [-]
    lifetime = 0.1; // [ps]
    matrix_elem = 0.; // [m0*eV]
    matrix_elem_sc_fact = 1.; // [-] change it when numerical value is different from the experimental one
    differenceQuotient = 0.01;  // [%]
    strains = false;
    fixed_qw_widths = false;
    build_struct_once = true;
    inTemperature.changedConnectMethod(this, &FermiNewGainSolver<GeometryType>::onInputChange);
    inCarriersConcentration.changedConnectMethod(this, &FermiNewGainSolver<GeometryType>::onInputChange);
}

template <typename GeometryType>
FermiNewGainSolver<GeometryType>::~FermiNewGainSolver() {
    inTemperature.changedDisconnectMethod(this, &FermiNewGainSolver<GeometryType>::onInputChange);
    inCarriersConcentration.changedDisconnectMethod(this, &FermiNewGainSolver<GeometryType>::onInputChange);
}


template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::loadConfiguration(XMLReader& reader, Manager& manager)
{
    // Load a configuration parameter from XML.
    while (reader.requireTagOrEnd())
    {
        std::string param = reader.getNodeName();
        if (param == "config") {
            roughness = reader.getAttribute<double>("roughness", roughness);
            lifetime = reader.getAttribute<double>("lifetime", lifetime);
            matrix_elem = reader.getAttribute<double>("matrix-elem", matrix_elem);
            matrix_elem_sc_fact = reader.getAttribute<double>("matrix-elem-sc-fact", matrix_elem_sc_fact);
            cond_qw_shift = reader.getAttribute<double>("cond-qw-shift", cond_qw_shift);
            vale_qw_shift = reader.getAttribute<double>("vale-qw-shift", vale_qw_shift);
            Tref = reader.getAttribute<double>("Tref", Tref);
            strains = reader.getAttribute<bool>("strains", strains);
            fixed_qw_widths = reader.getAttribute<bool>("fixed-qw-widths", fixed_qw_widths);
            build_struct_once = reader.getAttribute<bool>("fast-levels", build_struct_once);
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
            /*double *el = nullptr, *hh = nullptr, *lh = nullptr;
            try {
                el = new double[std::distance(elt.begin(), elt.end())+1];
                hh = new double[std::distance(hht.begin(), hht.end())+1];
                lh = new double[std::distance(lht.begin(), lht.end())+1];
                double* e = el; for (const auto& i: elt) *(e++) = - boost::lexical_cast<double>(i); *e = 1.;
                double* h = hh; for (const auto& i: hht) *(h++) = - boost::lexical_cast<double>(i); *h = 1.;
                double* l = lh; for (const auto& i: lht) *(l++) = - boost::lexical_cast<double>(i); *l = 1.;
            } catch(...) {
                delete[] el; delete[] hh; delete[] lh;
            }*/
            std::unique_ptr<double[]> el(new double[std::distance(elt.begin(), elt.end())+1]);
            std::unique_ptr<double[]> hh(new double[std::distance(hht.begin(), hht.end())+1]);
            std::unique_ptr<double[]> lh(new double[std::distance(lht.begin(), lht.end())+1]);
            double* e = el.get(); for (const auto& i: elt) *(e++) = - boost::lexical_cast<double>(i); *e = 1.;
            double* h = hh.get(); for (const auto& i: hht) *(h++) = - boost::lexical_cast<double>(i); *h = 1.;
            double* l = lh.get(); for (const auto& i: lht) *(l++) = - boost::lexical_cast<double>(i); *l = 1.;
            /*if (extern_levels) {
                delete[] extern_levels->el; delete[] extern_levels->hh; delete[] extern_levels->lh;
            }
            extern_levels.reset(QW::ExternalLevels(el.release(), hh.release(), lh.release()));*/
        } else
            this->parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <levels>, or <config>");
    }
}


template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::onInitialize() // In this function check if geometry and mesh are set
{
    if (!this->geometry) throw NoGeometryException(this->getId());

    detectActiveRegions();
    if (build_struct_once) {
        region_levels.resize(regions.size());
    }

    outGain.fireChanged();
    outLuminescence.fireChanged();
}


template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
{
    region_levels.clear();
}

/*template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::compute()
{
    this->initCalculation(); // This must be called before any calculation!
}*/

template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::detectActiveRegions()
{
    regions.clear();

    shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(this->geometry->getChild());
    shared_ptr<RectangularMesh<2>> points = mesh->getMidpointsMesh();

    size_t ileft = 0, iright = points->axis0->size();
    bool in_active = false;

    bool added_bottom_cladding = false;
    bool added_top_cladding = false;

    for (size_t r = 0; r < points->axis1->size(); ++r)
    {
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
            if (!added_bottom_cladding)
            {
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
                    //if (layer_QW) this->writelog(LOG_DETAIL, "Adding qw; h = %1%",h);
                    //else this->writelog(LOG_DETAIL, "Adding barrier; h = %1%",h);
                }
            }
            else
            {
                if (!added_top_cladding)
                {

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
        this->writelog(LOG_INFO, "Active region %1%: %2% nm single QW, %3% nm all QW, %4% nm total",
                       n++, 0.1*region.qwlen, 0.1*region.qwtotallen, 0.1*region.totallen);
    }

    // energy levels for active region with two identcal QWs won't be calculated so QW widths must be changed
    this->writelog(LOG_DETAIL, "Updating QW widths");
    n = 0;
    for (auto& region: regions) {
        region.lens.clear();
        int tN = region.size(); // number of all layers in the active region (QW, barr, external)
        int tNoOfQWs = 0; // number of QWs counter
        for (int i=0; i<tN; ++i) {
            if (region.isQW(i)) tNoOfQWs++;
            region.lens.push_back(region.getLayerBox(i).height()*1e4); // in [A]
            this->writelog(LOG_DEBUG, "Layer %1% thickness: %2% nm", i+1, 0.1*region.lens[i]);
        }
        this->writelog(LOG_DEBUG, "Number of QWs in the above active region: %1%", tNoOfQWs);
        this->writelog(LOG_DEBUG, "QW initial thickness: %1% nm", 0.1*region.qwlen);

        if (!fixed_qw_widths)
        {
            double tHstep = region.qwlen*roughness/qw_width_mod;
            if ( !(tNoOfQWs%2) )
            {
                double tH0 = region.qwlen-(int(tNoOfQWs/2))*tHstep+0.5*tHstep;
                for (int i=0; i<tN; ++i) {
                    if (region.isQW(i))
                    {
                        region.lens[i] = tH0;
                        this->writelog(LOG_DEBUG, "Layer %1% thickness: %2% nm", i+1, 0.1*region.lens[i]);
                        tH0 += tHstep;
                    }
                }
            }
            else
            {
                double tH0 = region.qwlen-(int(tNoOfQWs/2))*tHstep;
                for (int i=0; i<tN; ++i) {
                    if (region.isQW(i))
                    {
                        region.lens[i] = tH0;
                        this->writelog(LOG_DETAIL, "Layer %1% - modified thickness: %2% nm", i+1, 0.1*region.lens[i]);
                        tH0 += tHstep;
                    }
                }
            }
            this->writelog(LOG_DETAIL, "QW thickness step: %1% nm", 0.1*tHstep);
        }
        n++;
    }

}

template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::
    findEnergyLevels(typename FermiNewGainSolver<GeometryType>::Levels& levels, const ActiveRegionInfo& region, double iT, bool iShowSpecLogs)
{
    if (isnan(iT) || iT < 0.) throw ComputationError(this->getId(), "Wrong temperature (%1%K)", iT);

    //this->writelog(LOG_INFO,"gain settings: strains=%1%, roughness=%2%, lifetime=%3%, matrix_elem=%4%, matrix_elem_sc_fact=%5%, cond_qw_shift=%6%, vale_qw_shift=%7%, Tref=%8%, showspeclogs=%9%",
    //               strains, roughness, lifetime, matrix_elem, matrix_elem_sc_fact, cond_qw_shift, vale_qw_shift, Tref, iShowSpecLogs);

    if (strains)
    {
        if (!this->materialSubstrate) throw ComputationError(this->getId(), "No layer with role 'substrate' has been found");
        if (iShowSpecLogs)
            for (int i=0; i<region.size(); ++i)
            {
                double e = (this->materialSubstrate->lattC(iT,'a') - region.getLayerMaterial(i)->lattC(iT,'a')) / region.getLayerMaterial(i)->lattC(iT,'a');
                this->writelog(LOG_RESULT, "Layer %1% - strain: %2%%3%", i+1, e*100., '%');
            }
    }

    int tStrType = buildStructure(levels, iT, region, iShowSpecLogs);
    if (tStrType == 0) this->writelog(LOG_DETAIL, "I-type QW for Ec-Evhh and Ec-Evlh.");
    else if (tStrType == 1) this->writelog(LOG_DETAIL, "I-type QW for Ec-Evhh.");
    else if (tStrType == 2) this->writelog(LOG_DETAIL, "I-type QW for Ec-Evlh.");
    else this->writelog(LOG_DETAIL, "No I-type QW both for Ec-Evhh and Ec-Evlh.");

    // double tCladEg = region.getLayerMaterial(0)->CB(iT,0.) - region.getLayerMaterial(0)->VB(iT,0.); // cladding Eg (eV) TODO
    double tCladEg = region.getLayerMaterial(0)->Eg(iT,0.); // cladding Eg (eV) TODO

    std::vector<QW::Struktura *> tHoles; tHoles.clear();
    if (!levels.mEvhh) tHoles.push_back(levels.mpStrEvhh.get());
    if (!levels.mEvlh) tHoles.push_back(levels.mpStrEvlh.get());
    
    if ((!levels.mEc)&&((!levels.mEvhh)||(!levels.mEvlh)))
    {
        std::vector<double> tDso; tDso.clear();
        for (int i=0; i<region.size(); ++i) // LUKASZ 26.06
            tDso.push_back(region.getLayerMaterial(i)->Dso(iT));
        bool tShowM = false;
        if (iShowSpecLogs) tShowM = true;
        levels.aktyw = plask::make_shared<QW::ObszarAktywny>(levels.mpStrEc.get(), tHoles, tCladEg, tDso, roughness, iT, matrix_elem_sc_fact, tShowM); // roughness = 0.05 for example // TODO
        levels.aktyw->zrob_macierze_przejsc();
    }
    
    levels.invalid = false;
}

template <typename GeometryType>
int FermiNewGainSolver<GeometryType>::buildStructure(Levels& levels, double T, const ActiveRegionInfo& region, bool iShowSpecLogs)
{
    this->writelog(LOG_DETAIL, "Determining levels");

    levels.mEc = buildEc(levels, T, region, iShowSpecLogs);
    levels.mEvhh = buildEvhh(levels, T, region, iShowSpecLogs);
    levels.mEvlh = buildEvlh(levels, T, region, iShowSpecLogs);

    if (!levels.mEc)
    {
        if (levels.invalid) this->writelog(LOG_DETAIL, "Computing energy levels for electrons");
        levels.mpStrEc.reset(new QW::Struktura(levels.mpEc, QW::Struktura::el));
    }
    if (!levels.mEvhh)
    {
        if (levels.invalid) this->writelog(LOG_DETAIL, "Computing energy levels for heavy holes");
        levels.mpStrEvhh.reset(new QW::Struktura(levels.mpEvhh, QW::Struktura::hh));
    }
    if (!levels.mEvlh)
    {
        if (levels.invalid) this->writelog(LOG_DETAIL, "Computing energy levels for light holes");
        levels.mpStrEvlh.reset(new QW::Struktura(levels.mpEvlh, QW::Struktura::lh));
    }

    if ((!levels.mEc)&&(!levels.mEvhh)&&(!levels.mEvlh)) return 0;  // E-HH and E-LH
    else if ((!levels.mEc)&&(!levels.mEvhh)) return 1; // only E-HH
    else if ((!levels.mEc)&&(!levels.mEvlh)) return 2; // only E-LH
    else return -1;
}

template <typename GeometryType>
int FermiNewGainSolver<GeometryType>::buildEc(Levels& levels, double T, const ActiveRegionInfo& region, bool iShowSpecLogs)
{
    levels.mpEc.clear();

    int tN = region.size(); // number of all layers in the active region (QW, barr, external)

    double eClad1 = 0.; // TODO
    double eClad2 = 0.; // TODO

    bool tfStructOK = true;

    double tDEc = region.getLayerMaterial(0)->CB(T,eClad1); // Ec0 for cladding

    double tX = 0.;
    double tEc = (region.getLayerMaterial(0)->CB(T,eClad1)-tDEc);
    if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - CB: %2% eV", 1, region.getLayerMaterial(0)->CB(T,eClad1));
    levels.mpEc.emplace_back(new QW::WarstwaSkraj(QW::WarstwaSkraj::lewa, region.getLayerMaterial(0)->Me(T,eClad1).c00, region.getLayerMaterial(0)->Me(T,eClad1).c11, tX, tEc)); // left cladding
    for (int i=1; i<tN-1; ++i)
    {
        double e = 0.;
        if (strains) e = (this->materialSubstrate->lattC(T,'a') - region.getLayerMaterial(i)->lattC(T,'a')) / region.getLayerMaterial(i)->lattC(T,'a');
        double tH = region.lens[i]; // tH (A)
        double tCBaddShift(0.);
        if (region.isQW(i)) tCBaddShift = cond_qw_shift;
        tEc = (region.getLayerMaterial(i)->CB(T,e)+tCBaddShift-tDEc);
        if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - CB: %2% eV", i+1, region.getLayerMaterial(i)->CB(T,e)+tCBaddShift);
        levels.mpEc.emplace_back(new QW::Warstwa(region.getLayerMaterial(i)->Me(T,e).c00, region.getLayerMaterial(i)->Me(T,e).c11, tX, tEc, (tX+tH), tEc)); // wells and barriers
        tX += tH;
        if (region.getLayerMaterial(i)->CB(T,e) >= tDEc)
            tfStructOK = false;
    }
    tEc = (region.getLayerMaterial(tN-1)->CB(T,eClad2)-tDEc);
    if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - CB: %2% eV", tN, region.getLayerMaterial(tN-1)->CB(T,eClad2));
    levels.mpEc.emplace_back(new QW::WarstwaSkraj(QW::WarstwaSkraj::prawa, region.getLayerMaterial(tN-1)->Me(T,eClad2).c00, region.getLayerMaterial(tN-1)->Me(T,eClad2).c11, tX, tEc)); // right cladding

    if (tfStructOK) return 0; // band structure OK
    else return -1;
}

template <typename GeometryType>
int FermiNewGainSolver<GeometryType>::buildEvhh(Levels& levels, double T, const ActiveRegionInfo& region, bool iShowSpecLogs)
{
    levels.mpEvhh.clear();

    int tN = region.size(); // number of all layers int the active region (QW, barr, external)

    double eClad1 = 0.; // TODO
    double eClad2 = 0.; // TODO

    bool tfStructOK = true;

    double tDEvhh = region.getLayerMaterial(0)->VB(T,eClad1,'G','H'); // Ev0 for cladding

        double tX = 0.;
        double tEvhh = -(region.getLayerMaterial(0)->VB(T,eClad1,'G','H')-tDEvhh);
        if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - VB(hh): %2% eV", 1, region.getLayerMaterial(0)->VB(T,eClad1,'G','H'));
        levels.mpEvhh.emplace_back(new QW::WarstwaSkraj(QW::WarstwaSkraj::lewa, region.getLayerMaterial(0)->Mhh(T,eClad1).c00, region.getLayerMaterial(0)->Mhh(T,eClad1).c11, tX, tEvhh)); // left cladding
        for (int i=1; i<tN-1; ++i)
        {
            double e = 0.;
            if (strains) e = (this->materialSubstrate->lattC(T,'a') - region.getLayerMaterial(i)->lattC(T,'a')) / region.getLayerMaterial(i)->lattC(T,'a');
            double tH = region.lens[i]; // tH (A)
            double tVBaddShift(0.);
            if (region.isQW(i)) tVBaddShift = vale_qw_shift;
            tEvhh = -(region.getLayerMaterial(i)->VB(T,e,'G','H')+tVBaddShift-tDEvhh);
            if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - VB(hh): %2% eV", i+1, region.getLayerMaterial(i)->VB(T,e,'G','H')+tVBaddShift);
            levels.mpEvhh.emplace_back(new QW::Warstwa(region.getLayerMaterial(i)->Mhh(T,e).c00, region.getLayerMaterial(i)->Mhh(T,e).c11, tX, tEvhh, (tX+tH), tEvhh)); // wells and barriers
            tX += tH;
            if (region.getLayerMaterial(i)->VB(T,e,'G','H') <= tDEvhh)
                tfStructOK = false;
        }
        tEvhh = -(region.getLayerMaterial(tN-1)->VB(T,eClad2,'G','H')-tDEvhh);
        if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - VB(hh): %2% eV", tN, region.getLayerMaterial(tN-1)->VB(T,eClad2,'G','H'));
        levels.mpEvhh.emplace_back(new QW::WarstwaSkraj(QW::WarstwaSkraj::prawa, region.getLayerMaterial(tN-1)->Mhh(T,eClad2).c00, region.getLayerMaterial(tN-1)->Mhh(T,eClad2).c11, tX, tEvhh)); // add delete somewhere! TODO

        if (tfStructOK) return 0; // band structure OK
        else return -1;
}

template <typename GeometryType>
int FermiNewGainSolver<GeometryType>::buildEvlh(Levels& levels, double T, const ActiveRegionInfo& region, bool iShowSpecLogs)
{
    levels.mpEvlh.clear();

    int tN = region.size(); // number of all layers int the active region (QW, barr, external)

    double eClad1 = 0.; // TODO
    double eClad2 = 0.; // TODO

    bool tfStructOK = true;

    double tDEvlh = region.getLayerMaterial(0)->VB(T,eClad1,'G','L'); // Ev0 for cladding

        double tX = 0.;
        double tEvlh = -(region.getLayerMaterial(0)->VB(T,eClad1,'G','L')-tDEvlh);
        if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - VB(lh): %2% eV", 1, region.getLayerMaterial(0)->VB(T,eClad1,'G','L'));
        levels.mpEvlh.emplace_back(new QW::WarstwaSkraj(QW::WarstwaSkraj::lewa, region.getLayerMaterial(0)->Mlh(T,eClad1).c00, region.getLayerMaterial(0)->Mlh(T,eClad1).c11, tX, tEvlh)); // left cladding
        for (int i=1; i<tN-1; ++i)
        {
            double e = 0.;
            if (strains) e = (this->materialSubstrate->lattC(T,'a') - region.getLayerMaterial(i)->lattC(T,'a')) / region.getLayerMaterial(i)->lattC(T,'a');
            double tH = region.lens[i]; // tH (A)
            double tVBaddShift(0.);
            if (region.isQW(i)) tVBaddShift = vale_qw_shift;
            tEvlh = -(region.getLayerMaterial(i)->VB(T,e,'G','L')+tVBaddShift-tDEvlh);
            if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - VB(lh): %2% eV", i+1, region.getLayerMaterial(i)->VB(T,e,'G','L')+tVBaddShift);
            levels.mpEvlh.emplace_back(new QW::Warstwa(region.getLayerMaterial(i)->Mlh(T,e).c00, region.getLayerMaterial(i)->Mlh(T,e).c11, tX, tEvlh, (tX+tH), tEvlh)); // wells and barriers
            tX += tH;
            if (region.getLayerMaterial(i)->VB(T,e,'G','L') <= tDEvlh)
                tfStructOK = false;
        }
        tEvlh = -(region.getLayerMaterial(tN-1)->VB(T,eClad2,'G','L')-tDEvlh);
        if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - VB(lh): %2% eV", tN, region.getLayerMaterial(tN-1)->VB(T,eClad2,'G','L'));
        levels.mpEvlh.emplace_back(new QW::WarstwaSkraj(QW::WarstwaSkraj::prawa, region.getLayerMaterial(tN-1)->Mlh(T,eClad2).c00, region.getLayerMaterial(tN-1)->Mlh(T,eClad2).c11, tX, tEvlh)); // add delete somewhere! TODO

        if (tfStructOK) return 0; // band structure OK
        else return -1;
}


template <typename GeometryType>
QW::Gain FermiNewGainSolver<GeometryType>::getGainModule(double wavelength, double T, double n,
                                                         const ActiveRegionInfo& region, const Levels& levels, bool iShowSpecLogs)
{
    if (isnan(n) || n < 0.) throw ComputationError(this->getId(), "Wrong carrier concentration (%1%/cm3)", n);
    if (isnan(T) || T < 0.) throw ComputationError(this->getId(), "Wrong temperature (%1%K)", T);

    if ((!levels.mEc)&&((!levels.mEvhh)||(!levels.mEvlh)))
    {
        double tQWTotH = region.qwtotallen*0.1; // total thickness of QWs (nm)

        // calculating nR
        double tQWnR = 0.; // sty
        int tNoOfQWs = 0;
        int tN = region.size(); // number of all layers int the active region (QW, barr, external)
        for (int i=1; i<tN-1; ++i)
        {
            if (region.isQW(i))
            {
                tQWnR += region.getLayerMaterial(i)->nr(wavelength,T);
                tNoOfQWs++;
            }
        }
        tQWnR /= (tNoOfQWs*1.);
        //double tQWnR = region.materialQW->nr(wavelength,T); // QW nR // earlier

        double tEgClad = region.getLayerMaterial(0)->CB(T,0.) - region.getLayerMaterial(0)->VB(T,0.); // cladding Eg (eV) TODO
        //this->writelog(LOG_DEBUG, "mEgCladT z FermiNew: %1% eV", tEgClad);

        // this->writelog(LOG_DEBUG, "Creating gain module");
        QW::Gain gainModule;
        gainModule.setGain(levels.aktyw, n*(tQWTotH*1e-7), T, tQWnR, tEgClad);

        // this->writelog(LOG_DEBUG, "Recalculating carrier concentrations..");
        double tStep = 1e-1*n; // TODO
        double nQW=n;
        double iN1=0.,tn1=0.;

        for (int i=0; i<5; i++)
        {
            while(1)
            {
                iN1=n*10.;

                gainModule.setNsurf( iN1*(tQWTotH*1*1e-7) );
                double tFlc = gainModule.policz_qFlc();
                //double tFlv = gainModule1.policz_qFlv();

                std::vector<double> tN = levels.mpStrEc->koncentracje_w_warstwach(tFlc, T);
                auto tMaxElem = std::max_element(tN.begin(), tN.end());
                tn1 = *tMaxElem;
                tn1 = QW::Struktura::koncentracja_na_cm_3(tn1);
                if(tn1>=nQW)
                    n-=tStep;
                else
                {
                    n += tStep;
                    break;
                }
            }
            tStep *= 0.1;
            n -= tStep;
        }

        if (iShowSpecLogs)
        {
            if (!levels.mEc) levels.mpStrEc->showEnergyLevels("electrons", round(region.qwtotallen/region.qwlen));
            if (!levels.mEvhh) levels.mpStrEvhh->showEnergyLevels("heavy holes", round(region.qwtotallen/region.qwlen));
            if (!levels.mEvlh) levels.mpStrEvlh->showEnergyLevels("light holes", round(region.qwtotallen/region.qwlen));
        }
        // this->writelog(LOG_DETAIL, "Calculating quasi-Fermi levels and carrier concentrations..");
        double tFe = gainModule.policz_qFlc();
        double tFp = gainModule.policz_qFlv();
        if (iShowSpecLogs)
        {
            this->writelog(LOG_DETAIL, "Quasi-Fermi level for electrons: %1% eV from cladding conduction band edge", tFe);
            this->writelog(LOG_DETAIL, "Quasi-Fermi level for holes: %1% eV from cladding valence band edge", -tFp);
            std::vector<double> tN = levels.mpStrEc->koncentracje_w_warstwach(tFe, T);
            for(int i = 0; i <= (int) tN.size() - 1; i++)
                this->writelog(LOG_DETAIL, "Carriers concentration in layer %d: %.2e cm(-3)", i+1, QW::Struktura::koncentracja_na_cm_3(tN[i]));
        }

        return gainModule;
    }
    else if (levels.mEc)
        throw BadInput(this->getId(), "Conduction QW depth negative for electrons, check VB values of active-region materials");
    else //if ((mEvhh)&&(mEvlh))
        throw BadInput(this->getId(), "Valence QW depth negative for both heavy holes and light holes, check VB values of active-region materials");
}


static const shared_ptr<OrderedAxis> zero_axis(new OrderedAxis({0.}));

/// Base for lazy data implementation
template <typename GeometryT>
struct FermiNewGainSolver<GeometryT>::DataBase: public LazyDataImpl<double>
{
    struct AveragedData {
        shared_ptr<const RectangularMesh<2>> mesh;
        LazyData<double> data;
        double factor;
        const FermiNewGainSolver<GeometryT>* solver;
        const char* name;
        AveragedData(const FermiNewGainSolver<GeometryT>* solver, const char* name,
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

    FermiNewGainSolver<GeometryT>* solver;                 ///< Solver
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

    DataBase(FermiNewGainSolver<GeometryT>* solver, const shared_ptr<const MeshD<2>>& dst_mesh):
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
        for (size_t reg = 0; reg != solver->regions.size(); ++reg)
        {
            if (regpoints[reg]->size() == 0) {
                data[reg] = LazyData<double>(dest_mesh->size(), 0.);
                continue;
            }
            DataVector<double> values(regpoints[reg]->size());
            AveragedData temps(solver, "temperature", regpoints[reg], solver->regions[reg]);
            AveragedData concs(temps); concs.name = "carriers concentration";
            temps.data = solver->inTemperature(temps.mesh, interp);
            concs.data = solver->inCarriersConcentration(temps.mesh, interp);
            if (solver->build_struct_once)
                solver->findEnergyLevels(solver->region_levels[reg], solver->regions[reg], solver->Tref);
            std::exception_ptr error;
            #pragma omp parallel for
            for (size_t i = 0; i < regpoints[reg]->size(); ++i) {
                if (error) continue;
                try {
                    if (solver->build_struct_once) {
                        values[i] = getValue(wavelength, temps[i], max(concs[i], 1e-9),
                                             solver->regions[reg], solver->region_levels[reg]);
                    } else {
                        Levels levels;
                        solver->findEnergyLevels(levels, solver->regions[reg], temps[i]);
                        values[i] = getValue(wavelength, temps[i], max(concs[i], 1e-9),
                                             solver->regions[reg], levels);
                    }
                } catch(...) {
                    #pragma omp critical
                    error = std::current_exception();
                }
            }
            if (error) std::rethrow_exception(error);
            data[reg] = interpolate(make_shared<RectangularMesh<2>>(regpoints[reg], zero_axis),
                                    values, dest_mesh, interp);
        }
    }

    virtual double getValue(double wavelength, double temp, double conc,
                            const ActiveRegionInfo& region, const Levels& levels) = 0;

    size_t size() const override { return dest_mesh->size(); }

    double at(size_t i) const override {
        for (size_t reg = 0; reg != solver->regions.size(); ++reg)
//             if (solver->regions[reg].contains(dest_mesh->at(i)))
            if (solver->regions[reg].inQW(dest_mesh->at(i)))
                return data[reg][i];
        return 0.;
    }
};

template <typename GeometryT>
struct FermiNewGainSolver<GeometryT>::GainData: public FermiNewGainSolver<GeometryT>::DataBase
{
    template <typename... Args>
    GainData(Args... args): DataBase(args...) {}

    double getValue(double wavelength, double temp, double conc,
                    const ActiveRegionInfo& region, const Levels& levels) override
    {
        QW::Gain gainModule = this->solver->getGainModule(wavelength, temp, conc, region, levels);

        if ((!levels.mEc)&&((!levels.mEvhh)||(!levels.mEvlh)))
        {
            double L = region.qwtotallen / region.totallen; // no unit

            if (!this->solver->lifetime) return gainModule.wzmocnienie_calk_bez_splotu(nm_to_eV(wavelength)) / L;
            else return gainModule.wzmocnienie_calk_ze_splotem(nm_to_eV(wavelength),phys::hb_eV*1e12/this->solver->lifetime) / L;
        }
        if (levels.mEc)
            throw BadInput(this->solver->getId(), "Conduction QW depth negative for e, check VB values of active-region materials");
        else // if ((mEvhh) && (mEvlh))
            throw BadInput(this->solver->getId(), "Valence QW depth negative both for hh and lh, check VB values of active-region materials");
        return NAN;
    }
};

template <typename GeometryT>
struct FermiNewGainSolver<GeometryT>::DgDnData: public FermiNewGainSolver<GeometryT>::DataBase
{
    template <typename... Args>
    DgDnData(Args... args): DataBase(args...) {}

    double getValue(double wavelength, double temp, double conc,
                    const ActiveRegionInfo& region, const Levels& levels) override
    {
        double h = 0.5 * this->solver->differenceQuotient;
        double conc1 = (1.-h) * conc, conc2 = (1.+h) * conc;
        
        QW::Gain gainModule1 = this->solver->getGainModule(wavelength, temp, conc1, region, levels);
        QW::Gain gainModule2 = this->solver->getGainModule(wavelength, temp, conc2, region, levels);

        if ((!levels.mEc)&&((!levels.mEvhh)||(!levels.mEvlh)))
        {
            double L = region.qwtotallen / region.totallen; // no unit
            double gain1, gain2;
            if (!this->solver->lifetime) {
                gain1 = gainModule1.wzmocnienie_calk_bez_splotu(nm_to_eV(wavelength)) / L;
                gain2 = gainModule2.wzmocnienie_calk_bez_splotu(nm_to_eV(wavelength)) / L;
            } else {
                gain1 = gainModule1.wzmocnienie_calk_ze_splotem(nm_to_eV(wavelength),phys::hb_eV*1e12/this->solver->lifetime) / L;
                gain2 = gainModule2.wzmocnienie_calk_ze_splotem(nm_to_eV(wavelength),phys::hb_eV*1e12/this->solver->lifetime) / L;
            }
            return (gain2 - gain1) / (2.*h*conc);
        }
        if (levels.mEc)
            throw BadInput(this->solver->getId(), "Conduction QW depth negative for e, check VB values of active-region materials");
        else // if ((mEvhh) && (mEvlh))
            throw BadInput(this->solver->getId(), "Valence QW depth negative both for hh and lh, check VB values of active-region materials");
        return NAN;
    }
};

template <typename GeometryT>
struct FermiNewGainSolver<GeometryT>::LuminescenceData: public FermiNewGainSolver<GeometryT>::DataBase
{
    template <typename... Args>
    LuminescenceData(Args... args): DataBase(args...) {}

    double getValue(double wavelength, double temp, double conc,
                    const ActiveRegionInfo& region, const Levels& levels) override
    {
        QW::Gain gainModule = this->solver->getGainModule(wavelength, temp, conc, region, levels);

        if ((!levels.mEc)&&((!levels.mEvhh)||(!levels.mEvlh)))
        {
            double L = region.qwtotallen / region.totallen; // no unit
            return gainModule.luminescencja_calk(nm_to_eV(wavelength)) / L;
        }
        if (levels.mEc)
            throw BadInput(this->solver->getId(), "Conduction QW depth negative for e, check VB values of active-region materials");
        else // if ((mEvhh) && (mEvlh))
            throw BadInput(this->solver->getId(), "Valence QW depth negative both for hh and lh, check VB values of active-region materials");
        return NAN;
    }
};

template <typename GeometryType>
const LazyData<double> FermiNewGainSolver<GeometryType>::getGain(const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    this->writelog(LOG_DETAIL, "Calculating gain");
    this->initCalculation(); // This must be called before any calculation!

    GainData* data = new GainData(this, dst_mesh);
    data->compute(wavelength, getInterpolationMethod<INTERPOLATION_SPLINE>(interp));

    return LazyData<double>(data);
}

template <typename GeometryType>
const LazyData<double> FermiNewGainSolver<GeometryType>::getDgDn(const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    this->writelog(LOG_DETAIL, "Calculating gain over carriers derivative");
    this->initCalculation(); // This must be called before any calculation!

    DgDnData* data = new DgDnData(this, dst_mesh);
    data->compute(wavelength, getInterpolationMethod<INTERPOLATION_SPLINE>(interp));

    return LazyData<double>(data);
}

template <typename GeometryType>
const LazyData<double> FermiNewGainSolver<GeometryType>::getLuminescence(const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    this->writelog(LOG_DETAIL, "Calculating luminescence");
    this->initCalculation(); // This must be called before any calculation!

    LuminescenceData* data = new LuminescenceData(this, dst_mesh);
    data->compute(wavelength, getInterpolationMethod<INTERPOLATION_SPLINE>(interp));

    return LazyData<double>(data);
}

template <typename GeometryType>
GainSpectrum<GeometryType> FermiNewGainSolver<GeometryType>::getGainSpectrum(const Vec<2>& point)
{
    this->initCalculation();
    return GainSpectrum<GeometryType>(this, point);
}

template <typename GeometryType>
LuminescenceSpectrum<GeometryType> FermiNewGainSolver<GeometryType>::getLuminescenceSpectrum(const Vec<2>& point)
{
    this->initCalculation();
    return LuminescenceSpectrum<GeometryType>(this, point);
}

template <> std::string FermiNewGainSolver<Geometry2DCartesian>::getClassName() const { return "gain.FermiNew2D"; }
template <> std::string FermiNewGainSolver<Geometry2DCylindrical>::getClassName() const { return "gain.FermiNewCyl"; }

template struct PLASK_SOLVER_API FermiNewGainSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API FermiNewGainSolver<Geometry2DCylindrical>;

}}} // namespace

