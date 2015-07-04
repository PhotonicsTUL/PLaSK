#include "ferminew.h"

namespace plask { namespace solvers { namespace FermiNew {

template <typename GeometryType>
FermiNewGainSolver<GeometryType>::FermiNewGainSolver(const std::string& name): SolverWithMesh<GeometryType,OrderedMesh1D>(name),
    outGain(this, &FermiNewGainSolver<GeometryType>::getGain),
    outLuminescence(this, &FermiNewGainSolver<GeometryType>::getLuminescence)/*, // LUKASZ
    outGainOverCarriersConcentration(this, &FermiNewGainSolver<GeometryType>::getdGdn)*/ // getDelegated will be called whether provider value is requested
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
            build_struct_once = reader.getAttribute<bool>("build-struct-once", build_struct_once);
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

    calcLev = true;

    outGain.fireChanged();
    outLuminescence.fireChanged();
}


template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
{
    //TODO (if needed)
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
            this->writelog(LOG_DETAIL, "Layer %1% - thickness: %2% nm", i+1, 0.1*region.lens[i]);
        }
        this->writelog(LOG_DETAIL, "Number of QWs in the above active region: %1%", tNoOfQWs);
        this->writelog(LOG_DETAIL, "QW initial thickness: %1% nm", 0.1*region.qwlen);

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
                        this->writelog(LOG_DETAIL, "Layer %1% - modified thickness: %2% nm", i+1, 0.1*region.lens[i]);
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
void FermiNewGainSolver<GeometryType>::findEnergyLevels(const ActiveRegionInfo& region, double iT, bool iShowSpecLogs)
{
    if (isnan(iT) || iT < 0.) throw ComputationError(this->getId(), "Wrong temperature (%1%K)", iT);

    //this->writelog(LOG_INFO,"gain settings: strains=%1%, roughness=%2%, lifetime=%3%, matrix_elem=%4%, matrix_elem_sc_fact=%5%, cond_qw_shift=%6%, vale_qw_shift=%7%, Tref=%8%, showspeclogs=%9%",
    //               strains, roughness, lifetime, matrix_elem, matrix_elem_sc_fact, cond_qw_shift, vale_qw_shift, Tref, iShowSpecLogs);

    if (strains)
    {
        if (!this->materialSubstrate) throw ComputationError(this->getId(), "No layer with role 'substrate' has been found");

        for (int i=0; i<region.size(); ++i)
        {
            double e = (this->materialSubstrate->lattC(iT,'a') - region.getLayerMaterial(i)->lattC(iT,'a')) / region.getLayerMaterial(i)->lattC(iT,'a');
            if (iShowSpecLogs) this->writelog(LOG_RESULT, "Layer %1% - strain: %2%%3%", i+1, e*100., '%');
        }
    }

    int tStrType = buildStructure(iT, region, iShowSpecLogs);
    if (tStrType == 0) this->writelog(LOG_INFO, "I-type QW for Ec-Evhh and Ec-Evlh.");
    else if (tStrType == 1) this->writelog(LOG_INFO, "I-type QW for Ec-Evhh.");
    else if (tStrType == 2) this->writelog(LOG_INFO, "I-type QW for Ec-Evlh.");
    else this->writelog(LOG_INFO, "No I-type QW both for Ec-Evhh and Ec-Evlh.");

    double tCladEg = region.getLayerMaterial(0)->CB(iT,0.) - region.getLayerMaterial(0)->VB(iT,0.); // cladding Eg (eV) TODO

    std::vector<QW::Struktura *> tHoles; tHoles.clear();
    if (!mEvhh)
        tHoles.push_back(&(*mpStrEvhh));
    if (!mEvlh)
        tHoles.push_back(&(*mpStrEvlh));
    if ((!mEc)&&((!mEvhh)||(!mEvlh)))
    {
        std::vector<double> tDso; tDso.clear();
        for (int i=0; i<region.size(); ++i) // LUKASZ 26.06
            tDso.push_back(region.getLayerMaterial(i)->Dso(iT));
        bool tShowM = false;
        if (iShowSpecLogs) tShowM = true;
        aktyw = plask::shared_ptr<QW::ObszarAktywny>(new QW::ObszarAktywny(&(*mpStrEc), tHoles, tCladEg, tDso, roughness, iT, matrix_elem_sc_fact, tShowM)); // roughness = 0.05 for example // TODO
        aktyw->zrob_macierze_przejsc();
    }
}

template <typename GeometryType>
QW::Gain FermiNewGainSolver<GeometryType>::getGainModule(double wavelength, double T, double n, const ActiveRegionInfo& region, bool iShowSpecLogs)
{
    if (isnan(n) || n < 0.) throw ComputationError(this->getId(), "Wrong carrier concentration (%1%/cm3)", n);
    if (isnan(T) || T < 0.) throw ComputationError(this->getId(), "Wrong temperature (%1%K)", T);

    /*if (doTrefCalc)
    {
        findEnergyLevelsForTref(region, iShowSpecLogs);
        doTrefCalc = (!build_struct_once); // TODO tu ma (powinno) byc false by bylo szybko
    }*/

    if (build_struct_once)
    {
        if (calcLev)
        {
            findEnergyLevels(region, Tref, iShowSpecLogs);
            calcLev = false;
        }
    }
    else
        findEnergyLevels(region, T, iShowSpecLogs);

    if ((!mEc)&&((!mEvhh)||(!mEvlh)))
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

        this->writelog(LOG_DEBUG, "Creating gain module");
        QW::Gain gainModule;
        gainModule.setGain(aktyw, n*(tQWTotH*1e-7), T, tQWnR, tEgClad);

        this->writelog(LOG_DEBUG, "Recalculating carrier concentrations..");
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

                std::vector<double> tN = mpStrEc->koncentracje_w_warstwach(tFlc, T);
                auto tMaxElem = std::max_element(tN.begin(), tN.end());
                tn1 = *tMaxElem;
                tn1 = QW::Struktura::koncentracja_na_cm_3(tn1);
                this->writelog(LOG_DEBUG, "max. conc.: %1%", tn1); // TEST
                if(tn1>=nQW) n-=tStep;
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
            if (!mEc) mpStrEc->showEnergyLevels("electrons", round(region.qwtotallen/region.qwlen));
            if (!mEvhh) mpStrEvhh->showEnergyLevels("heavy holes", round(region.qwtotallen/region.qwlen));
            if (!mEvlh) mpStrEvlh->showEnergyLevels("light holes", round(region.qwtotallen/region.qwlen));
        }
        this->writelog(LOG_DETAIL, "Calculating quasi-Fermi levels and carrier concentrations..");
        double tFe = gainModule.policz_qFlc();
        double tFp = gainModule.policz_qFlv();
        if (iShowSpecLogs)
        {
            this->writelog(LOG_RESULT, "Quasi-Fermi level for electrons: %1% eV from cladding conduction band edge", tFe);
            this->writelog(LOG_RESULT, "Quasi-Fermi level for holes: %1% eV from cladding valence band edge", -tFp);
            std::vector<double> tN = mpStrEc->koncentracje_w_warstwach(tFe, T);
            for(int i = 0; i <= (int) tN.size() - 1; i++)
                this->writelog(LOG_RESULT, "carrier concentration in layer %1%: %2% cm^-3", i+1, QW::Struktura::koncentracja_na_cm_3(tN[i]));
        }

        return gainModule;
    }
    else if (mEc)
        throw BadInput(this->getId(), "Conduction QW depth negative for e, check VB values of active-region materials");
    else //if ((mEvhh)&&(mEvlh))
        throw BadInput(this->getId(), "Valence QW depth negative both for hh and lh, check VB values of active-region materials");

    /*
    if (isnan(n) || n < 0.) throw ComputationError(this->getId(), "Wrong carrier concentration (%1%/cm3)", n);
    if (isnan(T) || T < 0.) throw ComputationError(this->getId(), "Wrong temperature (%1%K)", T);

    if (if_strain)
    {
        if (!this->materialSubstrate) throw ComputationError(this->getId(), "No layer with role 'substrate' has been found");

        for (int i=0; i<region.size(); ++i)
        {
            double e = (this->materialSubstrate->lattC(T,'a') - region.getLayerMaterial(i)->lattC(T,'a')) / region.getLayerMaterial(i)->lattC(T,'a');
            //double tH = region.getLayerBox(i).height(); // (um)
            //this->writelog(LOG_RESULT, "Layer %1% - strain: %2%%3%, thickness: %4%nm", i+1, e*100., '%', tH*1000.);
            if (iShowSpecLogs) writelog(LOG_RESULT, "Layer %1% - strain: %2%%3%", i+1, e*100., '%');
        }
    }
    // ODTAD PRZENOSIMY DO OSOBNEJ FUNKCJI
    int tStrType = buildStructure(T, region, iShowSpecLogs);
    if (tStrType == 0) writelog(LOG_INFO, "I-type QW for Ec-Evhh and Ec-Evlh.");
    else if (tStrType == 1) writelog(LOG_INFO, "I-type QW for Ec-Evhh.");
    else if (tStrType == 2) writelog(LOG_INFO, "I-type QW for Ec-Evlh.");
    else writelog(LOG_INFO, "No I-type QW both for Ec-Evhh and Ec-Evlh.");

    double tCladEg = region.getLayerMaterial(0)->CB(T,0.) - region.getLayerMaterial(0)->VB(T,0.); // cladding Eg (eV) TODO
    double tQWTotH = region.qwtotallen*0.1; // total thickness of QWs (nm)
    double tQWnR = region.materialQW->nr(wavelength,T); // QW nR

    std::vector<QW::struktura *> tHoles; tHoles.clear();
    if (!mEvhh)
        tHoles.push_back(&(*mpStrEvhh));
    if (!mEvlh)
        tHoles.push_back(&(*mpStrEvlh));
    if ((!mEc)&&((!mEvhh)||(!mEvlh)))
    {
        std::vector<double> tDso; tDso.clear();
        for (int i=0; i<region.size(); ++i) // LUKASZ 26.06
            tDso.push_back(region.getLayerMaterial(i)->Dso(T));
        bool tShowM = false;
        if (iShowSpecLogs) tShowM = true;
        plask::shared_ptr<QW::obszar_aktywny> aktyw(new QW::obszar_aktywny(&(*mpStrEc), tHoles, tCladEg, tDso, roughness, matrixelemscfact, tShowM)); // roughness = 0.05 for example // TODO
        aktyw->zrob_macierze_przejsc();
        // TU KONCZYMY PRZENOSZENIE DO OSOBNEJ FUNKCJI BO DALEJ WCHODZI TEMPERATURA
        n = recalcConc(aktyw, n, tQWTotH, T, tQWnR); // LUKASZ
        QW::Gain gainModule(aktyw, n*(tQWTotH*1e-7), T, tQWnR);

        if (iShowSpecLogs)
        {
            if (!mEc) mpStrEc->showEnergyLevels("electrons", round(region.qwtotallen/region.qwlen));
            if (!mEvhh) mpStrEvhh->showEnergyLevels("heavy holes", round(region.qwtotallen/region.qwlen));
            if (!mEvlh) mpStrEvlh->showEnergyLevels("light holes", round(region.qwtotallen/region.qwlen));
        }
        writelog(LOG_INFO, "Calculating quasi Fermi levels and carrier concentrations..");
        double tFe = gainModule.policz_qFlc();
        double tFp = gainModule.policz_qFlv();
        if (iShowSpecLogs)
        {
            writelog(LOG_RESULT, "Quasi-Fermi level for electrons: %1% eV from cladding band edge", tFe);
            writelog(LOG_RESULT, "Quasi-Fermi level for holes: %1% eV from cladding band edge", -tFp);
            std::vector<double> tN = mpStrEc->koncentracje_w_warstwach(tFe, T);
            for(int i = 0; i <= (int) tN.size() - 1; i++)
                writelog(LOG_RESULT, "carrier concentration in layer %1%: %2% cm^-3", i+1, QW::struktura::koncentracja_na_cm_3(tN[i]));
        }

        return gainModule;
    }
    else if (mEc)
        throw BadInput(this->getId(), "Conduction QW depth negative for e, check VB values of active-region materials");
    else //if ((mEvhh)&&(mEvlh))
        throw BadInput(this->getId(), "Valence QW depth negative both for hh and lh, check VB values of active-region materials");
    */
}

template <typename GeometryType>
int FermiNewGainSolver<GeometryType>::buildStructure(double T, const ActiveRegionInfo& region, bool iShowSpecLogs)
{
    this->writelog(LOG_INFO, "Building structure");

    mEc = buildEc(T, region, iShowSpecLogs);
    mEvhh = buildEvhh(T, region, iShowSpecLogs);
    mEvlh = buildEvlh(T, region, iShowSpecLogs);

    if (!mEc)
    {
        //this->writelog(LOG_INFO, "Computing energy levels for electrons");
        mpStrEc = new QW::Struktura(mpEc, QW::Struktura::el);
    }
    if (!mEvhh)
    {
        //this->writelog(LOG_INFO, "Computing energy levels for heavy holes");
        mpStrEvhh = new QW::Struktura(mpEvhh, QW::Struktura::hh);
    }
    if (!mEvlh)
    {
        //this->writelog(LOG_INFO, "Computing energy levels for light holes");
        mpStrEvlh = new QW::Struktura(mpEvlh, QW::Struktura::lh);
    }

    if ((!mEc)&&(!mEvhh)&&(!mEvlh)) return 0;  // E-HH and E-LH
    else if ((!mEc)&&(!mEvhh)) return 1; // only E-HH
    else if ((!mEc)&&(!mEvlh)) return 2; // only E-LH
    else return -1;
}

template <typename GeometryType>
int FermiNewGainSolver<GeometryType>::buildEc(double T, const ActiveRegionInfo& region, bool iShowSpecLogs)
{
    mpEc.clear();

    int tN = region.size(); // number of all layers in the active region (QW, barr, external)

    double eClad1 = 0.; // TODO
    double eClad2 = 0.; // TODO

    bool tfStructOK = true;

    double tDEc = region.getLayerMaterial(0)->CB(T,eClad1); // Ec0 for cladding

    double tX = 0.;
    double tEc = (region.getLayerMaterial(0)->CB(T,eClad1)-tDEc);
    if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - CB: %2% eV", 1, region.getLayerMaterial(0)->CB(T,eClad1));
    mpLay = new QW::WarstwaSkraj(QW::WarstwaSkraj::lewa, region.getLayerMaterial(0)->Me(T,eClad1).c00, region.getLayerMaterial(0)->Me(T,eClad1).c11, tX, tEc); // left cladding
    mpEc.push_back(mpLay);
    for (int i=1; i<tN-1; ++i)
    {
        double e = 0.;
        if (strains) e = (this->materialSubstrate->lattC(T,'a') - region.getLayerMaterial(i)->lattC(T,'a')) / region.getLayerMaterial(i)->lattC(T,'a');
        double tH = region.lens[i]; // tH (A)
        double tCBaddShift(0.);
        if (region.isQW(i)) tCBaddShift = cond_qw_shift;
        tEc = (region.getLayerMaterial(i)->CB(T,e)+tCBaddShift-tDEc);
        if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - CB: %2% eV", i+1, region.getLayerMaterial(i)->CB(T,e)+tCBaddShift);
        mpLay = new QW::Warstwa(region.getLayerMaterial(i)->Me(T,e).c00, region.getLayerMaterial(i)->Me(T,e).c11, tX, tEc, (tX+tH), tEc); // wells and barriers
        mpEc.push_back(mpLay); tX += tH;
        if (region.getLayerMaterial(i)->CB(T,e) >= tDEc)
            tfStructOK = false;
    }
    tEc = (region.getLayerMaterial(tN-1)->CB(T,eClad2)-tDEc);
    if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - CB: %2% eV", tN, region.getLayerMaterial(tN-1)->CB(T,eClad2));
    mpLay = new QW::WarstwaSkraj(QW::WarstwaSkraj::prawa, region.getLayerMaterial(tN-1)->Me(T,eClad2).c00, region.getLayerMaterial(tN-1)->Me(T,eClad2).c11, tX, tEc); // right cladding
    mpEc.push_back(mpLay); // add delete somewhere! TODO

    if (tfStructOK) return 0; // band structure OK
    else return -1;
}

template <typename GeometryType>
int FermiNewGainSolver<GeometryType>::buildEvhh(double T, const ActiveRegionInfo& region, bool iShowSpecLogs)
{
    mpEvhh.clear();

    int tN = region.size(); // number of all layers int the active region (QW, barr, external)

    double eClad1 = 0.; // TODO
    double eClad2 = 0.; // TODO

    bool tfStructOK = true;

    double tDEvhh = region.getLayerMaterial(0)->VB(T,eClad1,'G','H'); // Ev0 for cladding

        double tX = 0.;
        double tEvhh = -(region.getLayerMaterial(0)->VB(T,eClad1,'G','H')-tDEvhh);
        if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - VB(hh): %2% eV", 1, region.getLayerMaterial(0)->VB(T,eClad1,'G','H'));
        mpLay = new QW::WarstwaSkraj(QW::WarstwaSkraj::lewa, region.getLayerMaterial(0)->Mhh(T,eClad1).c00, region.getLayerMaterial(0)->Mhh(T,eClad1).c11, tX, tEvhh); // left cladding
        mpEvhh.push_back(mpLay);
        for (int i=1; i<tN-1; ++i)
        {
            double e = 0.;
            if (strains) e = (this->materialSubstrate->lattC(T,'a') - region.getLayerMaterial(i)->lattC(T,'a')) / region.getLayerMaterial(i)->lattC(T,'a');
            double tH = region.lens[i]; // tH (A)
            double tVBaddShift(0.);
            if (region.isQW(i)) tVBaddShift = vale_qw_shift;
            tEvhh = -(region.getLayerMaterial(i)->VB(T,e,'G','H')+tVBaddShift-tDEvhh);
            if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - VB(hh): %2% eV", i+1, region.getLayerMaterial(i)->VB(T,e,'G','H')+tVBaddShift);
            mpLay = new QW::Warstwa(region.getLayerMaterial(i)->Mhh(T,e).c00, region.getLayerMaterial(i)->Mhh(T,e).c11, tX, tEvhh, (tX+tH), tEvhh); // wells and barriers
            mpEvhh.push_back(mpLay); tX += tH;
            if (region.getLayerMaterial(i)->VB(T,e,'G','H') <= tDEvhh)
                tfStructOK = false;
        }
        tEvhh = -(region.getLayerMaterial(tN-1)->VB(T,eClad2,'G','H')-tDEvhh);
        if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - VB(hh): %2% eV", tN, region.getLayerMaterial(tN-1)->VB(T,eClad2,'G','H'));
        mpLay = new QW::WarstwaSkraj(QW::WarstwaSkraj::prawa, region.getLayerMaterial(tN-1)->Mhh(T,eClad2).c00, region.getLayerMaterial(tN-1)->Mhh(T,eClad2).c11, tX, tEvhh);
        mpEvhh.push_back(mpLay); // add delete somewhere! TODO

        if (tfStructOK) return 0; // band structure OK
        else return -1;
}

template <typename GeometryType>
int FermiNewGainSolver<GeometryType>::buildEvlh(double T, const ActiveRegionInfo& region, bool iShowSpecLogs)
{
    mpEvlh.clear();

    int tN = region.size(); // number of all layers int the active region (QW, barr, external)

    double eClad1 = 0.; // TODO
    double eClad2 = 0.; // TODO

    bool tfStructOK = true;

    double tDEvlh = region.getLayerMaterial(0)->VB(T,eClad1,'G','L'); // Ev0 for cladding

        double tX = 0.;
        double tEvlh = -(region.getLayerMaterial(0)->VB(T,eClad1,'G','L')-tDEvlh);
        if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - VB(lh): %2% eV", 1, region.getLayerMaterial(0)->VB(T,eClad1,'G','L'));
        mpLay = new QW::WarstwaSkraj(QW::WarstwaSkraj::lewa, region.getLayerMaterial(0)->Mlh(T,eClad1).c00, region.getLayerMaterial(0)->Mlh(T,eClad1).c11, tX, tEvlh); // left cladding
        mpEvlh.push_back(mpLay);
        for (int i=1; i<tN-1; ++i)
        {
            double e = 0.;
            if (strains) e = (this->materialSubstrate->lattC(T,'a') - region.getLayerMaterial(i)->lattC(T,'a')) / region.getLayerMaterial(i)->lattC(T,'a');
            double tH = region.lens[i]; // tH (A)
            double tVBaddShift(0.);
            if (region.isQW(i)) tVBaddShift = vale_qw_shift;
            tEvlh = -(region.getLayerMaterial(i)->VB(T,e,'G','L')+tVBaddShift-tDEvlh);
            if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - VB(lh): %2% eV", i+1, region.getLayerMaterial(i)->VB(T,e,'G','L')+tVBaddShift);
            mpLay = new QW::Warstwa(region.getLayerMaterial(i)->Mlh(T,e).c00, region.getLayerMaterial(i)->Mlh(T,e).c11, tX, tEvlh, (tX+tH), tEvlh); // wells and barriers
            mpEvlh.push_back(mpLay); tX += tH;
            if (region.getLayerMaterial(i)->VB(T,e,'G','L') <= tDEvlh)
                tfStructOK = false;
        }
        tEvlh = -(region.getLayerMaterial(tN-1)->VB(T,eClad2,'G','L')-tDEvlh);
        if (iShowSpecLogs) this->writelog(LOG_DETAIL, "Layer %1% - VB(lh): %2% eV", tN, region.getLayerMaterial(tN-1)->VB(T,eClad2,'G','L'));
        mpLay = new QW::WarstwaSkraj(QW::WarstwaSkraj::prawa, region.getLayerMaterial(tN-1)->Mlh(T,eClad2).c00, region.getLayerMaterial(tN-1)->Mlh(T,eClad2).c11, tX, tEvlh);
        mpEvlh.push_back(mpLay); // add delete somewhere! TODO

        if (tfStructOK) return 0; // band structure OK
        else return -1;
}

template <typename GeometryType>
const LazyData<double> FermiNewGainSolver<GeometryType>::getGain(const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    if (interp == INTERPOLATION_DEFAULT) interp = INTERPOLATION_SPLINE;

    this->initCalculation(); // This must be called before any calculation!

    auto mesh2 = make_shared<RectangularMesh<2>>();
    if (this->mesh) {
        auto verts = make_shared<OrderedAxis>();
        for (auto p: *dst_mesh) verts->addPoint(p.vert());
        mesh2->setAxis0(this->mesh); mesh2->setAxis1(verts);
    }
    const shared_ptr<const MeshD<2>> src_mesh((this->mesh)? mesh2 : dst_mesh);

    DataVector<const double> nOnMesh = inCarriersConcentration(src_mesh, interp); // carriers concentration on the mesh
    DataVector<const double> TOnMesh = inTemperature(src_mesh, interp); // temperature on the mesh
    DataVector<double> gainOnMesh(src_mesh->size(), 0.);

    InterpolationFlags flags(this->geometry);
    std::vector<std::pair<size_t,size_t>> points;
    for (size_t i = 0; i != src_mesh->size(); i++)
        for (size_t r = 0; r != regions.size(); ++r)
            if (regions[r].contains(flags.wrap(src_mesh->at(i))) && nOnMesh[i] > 0.)
                points.push_back(std::make_pair(i,r));

    //#pragma omp parallel for // do not use parallel computations now LUKASZ 2014.10.16
    this->writelog(LOG_INFO, "Running gain calculations");
    for (int j = 0; j < points.size(); j++)
    {
        size_t i = points[j].first;

        double T = TOnMesh[i];
        double n = nOnMesh[i];

        const ActiveRegionInfo& region = regions[points[j].second];

        QW::Gain gainModule = getGainModule(wavelength, T, n, region);

        if ((!mEc)&&((!mEvhh)||(!mEvlh)))
        {
            /* std::vector<double> tN = mpStrEc->koncentracje_w_warstwach(tFe, T);
            for(int i = 0; i <= (int) tN.size() - 1; i++)
                this->writelog(LOG_RESULT, "koncentracja_na_cm_3 w warstwie %1% wynosi %2%", i, QW::struktura::koncentracja_na_cm_3(tN[i])); */
            double L = region.qwtotallen / region.totallen; // no unit

            if (!lifetime) gainOnMesh[i] = gainModule.wzmocnienie_calk_bez_splotu(nm_to_eV(wavelength)) / L; //20.10.2014 adding lifetime
            else gainOnMesh[i] = gainModule.wzmocnienie_calk_ze_splotem(nm_to_eV(wavelength),phys::hb_eV*1e12/lifetime) / L; //20.10.2014 adding lifetime

            this->writelog(LOG_DATA, "pos(%1%um,%2%um): gain=%3%cm-1", (dst_mesh->at(i)).c0, (dst_mesh->at(i)).c1, gainOnMesh[i]);
        }
        else if (mEc)
            throw BadInput(this->getId(), "Conduction QW depth negative for e, check VB values of active-region materials");
        else // if ((mEvhh) && (mEvlh))
            throw BadInput(this->getId(), "Valence QW depth negative both for hh and lh, check VB values of active-region materials");
    }

    if (this->mesh)
    {
        return interpolate(mesh2, gainOnMesh, dst_mesh, interp, this->geometry);
    }
    else
    {
        return gainOnMesh;
    }
} // LUKASZ

template <typename GeometryType>
const LazyData<double> FermiNewGainSolver<GeometryType>::getLuminescence(const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    if (interp == INTERPOLATION_DEFAULT) interp = INTERPOLATION_SPLINE;

    this->writelog(LOG_INFO, "Calculating luminescence");
    this->initCalculation(); // This must be called before any calculation!

    auto mesh2 = make_shared<RectangularMesh<2>>();    //RectilinearMesh2D
    if (this->mesh) {
        auto verts = make_shared<OrderedAxis>();
        for (auto p: *dst_mesh) verts->addPoint(p.vert());
        mesh2->setAxis0(this->mesh); mesh2->setAxis1(verts);
    }
    const shared_ptr<const MeshD<2>> src_mesh((this->mesh)? mesh2 : dst_mesh);

    DataVector<const double> nOnMesh = inCarriersConcentration(src_mesh, interp); // carriers concentration on the mesh
    DataVector<const double> TOnMesh = inTemperature(src_mesh, interp); // temperature on the mesh
    DataVector<double> luminescenceOnMesh(src_mesh->size(), 0.);

    std::vector<std::pair<size_t,size_t>> points;
    for (size_t i = 0; i != src_mesh->size(); i++)
        for (size_t r = 0; r != regions.size(); ++r)
            if (regions[r].contains(src_mesh->at(i)) && nOnMesh[i] > 0.)
                points.push_back(std::make_pair(i,r));

    //#pragma omp parallel for // do not use parallel computations now LUKASZ 2014.10.16
    for (int j = 0; j < points.size(); j++)
    {
        size_t i = points[j].first;

        double T = TOnMesh[i];
        double n = nOnMesh[i];

        const ActiveRegionInfo& region = regions[points[j].second];

        QW::Gain gainModule = getGainModule(wavelength, T, n, region);

        if ((!mEc)&&((!mEvhh)||(!mEvlh)))
        {
            double L = region.qwtotallen / region.totallen; // no unit

            luminescenceOnMesh[i] = gainModule.luminescencja_calk(nm_to_eV(wavelength)) / L;
            this->writelog(LOG_RESULT, "calculated luminescence: %1% ?", luminescenceOnMesh[i]);
        }
        else if (mEc)
            throw BadInput(this->getId(), "Conduction QW depth negative for e, check VB values of active-region materials");
        else // if ((mEvhh) && (mEvlh))
            throw BadInput(this->getId(), "Valence QW depth negative both for hh and lh, check VB values of active-region materials");
    }

    if (this->mesh)
    {
        return interpolate(mesh2, luminescenceOnMesh,dst_mesh, interp, this->geometry);
    }
    else
    {
        return luminescenceOnMesh;
    }
} // LUKASZ

/*
template <typename GeometryType>
const DataVector<double> FermiNewGainSolver<GeometryType>::getdGdn(const MeshD<2>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    if (interp == INTERPOLATION_DEFAULT) interp = INTERPOLATION_SPLINE;

    this->writelog(LOG_INFO, "Calculating gain over carriers concentration first derivative");
    this->initCalculation(); // This must be called before any calculation!

    RectilinearMesh2D mesh2;
    if (this->mesh) {
        OrderedAxis verts;
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
}*/ // LUKASZ


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
