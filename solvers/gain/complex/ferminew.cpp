#include "ferminew.h"

namespace plask { namespace solvers { namespace ferminew {

template <typename GeometryType>
FerminewGainSolver<GeometryType>::FerminewGainSolver(const std::string& name): SolverWithMesh<GeometryType,RectilinearMesh1D>(name),
    outGain(this, &FerminewGainSolver<GeometryType>::getGain)/*, // LUKASZ
    outGainOverCarriersConcentration(this, &FerminewGainSolver<GeometryType>::getdGdn)*/ // getDelegated will be called whether provider value is requested
{
    inTemperature = 300.; // temperature receiver has some sensible value
    roughness = 0.05; // [-]
    matrixelem = 0.; // [m0*eV]
    differenceQuotient = 0.01;  // [%]
    if_strain = false;
    inTemperature.changedConnectMethod(this, &FerminewGainSolver<GeometryType>::onInputChange);
    inCarriersConcentration.changedConnectMethod(this, &FerminewGainSolver<GeometryType>::onInputChange);
}

template <typename GeometryType>
FerminewGainSolver<GeometryType>::~FerminewGainSolver() {
    inTemperature.changedDisconnectMethod(this, &FerminewGainSolver<GeometryType>::onInputChange);
    inCarriersConcentration.changedDisconnectMethod(this, &FerminewGainSolver<GeometryType>::onInputChange);
}


template <typename GeometryType>
void FerminewGainSolver<GeometryType>::loadConfiguration(XMLReader& reader, Manager& manager)
{
    // Load a configuration parameter from XML.
    while (reader.requireTagOrEnd())
    {
        std::string param = reader.getNodeName();
        if (param == "config") {
            roughness = reader.getAttribute<double>("roughness", roughness);
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
void FerminewGainSolver<GeometryType>::onInitialize() // In this function check if geometry and mesh are set
{
    if (!this->geometry) throw NoGeometryException(this->getId());

    detectActiveRegions();

    //TODO

    outGain.fireChanged();
}


template <typename GeometryType>
void FerminewGainSolver<GeometryType>::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
{
    //TODO (if needed)
}

/*template <typename GeometryType>
void FerminewGainSolver<GeometryType>::compute()
{
    this->initCalculation(); // This must be called before any calculation!
}*/

template <typename GeometryType>
void FerminewGainSolver<GeometryType>::detectActiveRegions()
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
                        // LUKASZ: jestem w obszarze z rola "active" po raz pierwszy
                        // LUKASZ: trzeba dodac warstwe, ktora jest tuz pod obszarem czynnym
                        // LUKASZ: i trzeba to zrobic tylko raz

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
                // LUKASZ: obszar czynny juz byl
                // LUKASZ: jestem poza obszarem z rola "active"
                // LUKASZ: trzeba dodac warstwe, ktora jest tuz nad obszarem czynnym
                // LUKASZ: i trzeba to zrobic tylko raz


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
QW::gain FerminewGainSolver<GeometryType>::getGainModule(double wavelength, double T, double n, const ActiveRegionInfo& region)
{
    if (isnan(n) || n < 0.) throw ComputationError(this->getId(), "Wrong carrier concentration (%1%/cm3)", n);
    if (isnan(T) || T < 0.) throw ComputationError(this->getId(), "Wrong temperature (%1%K)", T);

    if (if_strain)
    {
        if (!this->materialSubstrate) throw ComputationError(this->getId(), "No layer with role 'substrate' has been found");

        for (int i=0; i<region.size(); ++i)
        {
            double e = (this->materialSubstrate->lattC(T,'a') - region.getLayerMaterial(i)->lattC(T,'a')) / region.getLayerMaterial(i)->lattC(T,'a');
            double tH = region.getLayerBox(i).height(); // (um) czy na pewno h powinno byc w um i czy w ten sposob uzyskuje dobra wartosc? // TODO
            writelog(LOG_RESULT, "Layer %1% - strain: %2%, thickness: %3%", i, e, tH);
        }
    }

    buildStructure(T, region);

    double tCladEg = region.getLayerMaterial(0)->CB(T,0.) - region.getLayerMaterial(0)->VB(T,0.); // cladding Eg (eV) TODO
    double tQWDso = region.materialQW->Dso(T); // QW Dso (eV)
    double tQWTotH = region.qwtotallen*0.1; // total thickness of QWs (nm)
    double tQWnR = region.materialQW->nr(wavelength,T); // QW nR

    std::vector<QW::struktura *> tHoles;
    if (!mEvhh)
        tHoles.push_back(&(*mpStrEvhh));
    if (!mEvlh)
        tHoles.push_back(&(*mpStrEvlh));
    if ((!mEc)&&((!mEvhh)||(!mEvlh)))
    {
        plask::shared_ptr<QW::obszar_aktywny> aktyw(new QW::obszar_aktywny(&(*mpStrEc), tHoles, tCladEg, tQWDso, roughness)); // roughness = 0.05 for example // TODO
        aktyw->zrob_macierze_przejsc();
        //writelog(LOG_INFO, "Do funkcji Adama idzie %1%, %2%, %3% i %4%", n, tQWTotH, T, tQWnR);
        //n = przelicz_nQW_na_npow(aktyw, n, tQWTotH, T, tQWnR); // ADAM
        QW::gain gainModule(aktyw, n*(tQWTotH*1e-7), T, tQWnR); // TODO

        writelog(LOG_INFO, "Calculating quasi Fermi levels and carrier concentrations..");
        double tFe = gainModule.policz_qFlc();
        double tFp = gainModule.policz_qFlv();
        writelog(LOG_RESULT, "Fe %1%", tFe);
        writelog(LOG_RESULT, "Fp %1%", tFp);
        std::vector<double> tN = mpStrEc->koncentracje_w_warstwach(tFe, T);
        for(int i = 0; i <= (int) tN.size() - 1; i++)
            writelog(LOG_RESULT, "koncentracja_na_cm_3 w warstwie %1% wynosi %2%", i, QW::struktura::koncentracja_na_cm_3(tN[i]));

        /*double tGehh = gainModule.wzmocnienie_od_pary_pasm(nm_to_eV(wavelength), 0, 0);
        double tGelh = gainModule.wzmocnienie_od_pary_pasm(nm_to_eV(wavelength), 0, 1);
        //gainOnMesh[i] = tGehh+tGelh;
        //writelog(LOG_RESULT, "gainOnMesh %1%", gainOnMesh[i]);
        writelog(LOG_RESULT, "gainOnMesh %1%", tGehh+tGelh);*/

        return gainModule;
    }
    else if (mEc)
        throw BadInput(this->getId(), "Conduction QW depth negative for e, check VB values of active-region materials");
    else //if ((mEvhh)&&(mEvlh))
        throw BadInput(this->getId(), "Valence QW depth negative both for hh and lh, check VB values of active-region materials");
}

template <typename GeometryType>
int FerminewGainSolver<GeometryType>::buildStructure(double T, const ActiveRegionInfo& region)
{
    writelog(LOG_INFO, "Building structure");

    mEc = buildEc(T, region);
    mEvhh = buildEvhh(T, region);
    mEvlh = buildEvlh(T, region);

    if (!mEc) mpStrEc = new QW::struktura(mpEc, QW::struktura::el);
    if (!mEvhh) mpStrEvhh = new QW::struktura(mpEvhh, QW::struktura::hh);
    if (!mEvlh) mpStrEvlh = new QW::struktura(mpEvlh, QW::struktura::lh);

    if ((!mEc)&&(!mEvhh)&&(!mEvlh)) return 0;  // E-HH and E-LH
    else if ((!mEc)&&(!mEvhh)) return 1; // only E-HH
    else if ((!mEc)&&(!mEvlh)) return 2; // only E-LH
    else return -1;
}

template <typename GeometryType>
int FerminewGainSolver<GeometryType>::buildEc(double T, const ActiveRegionInfo& region)
{
    //writelog(LOG_INFO, "Building Ec");

    mpEc.clear();

    int tN = region.size(); // number of all layers int the active region (QW, barr, external)

    //writelog(LOG_DETAIL, "number of all layers int the active region: %1%", tN);

    double eClad1 = 0.; // TODO
    double eClad2 = 0.; // TODO

    bool tfStructOK = true;

    double tDEc = region.getLayerMaterial(0)->CB(T,eClad1); // Ec0 for cladding

    double tX = 0.;
    mpLay = new QW::warstwa_skraj(QW::warstwa_skraj::lewa, region.getLayerMaterial(0)->Me(T,eClad1).c00, region.getLayerMaterial(0)->Me(T,eClad1).c11, tX, (region.getLayerMaterial(0)->CB(T,eClad1)-tDEc)); // left cladding
    mpEc.push_back(mpLay);
    for (int i=1; i<tN-1; ++i)
    {
        double e = (this->materialSubstrate->lattC(T,'a') - region.getLayerMaterial(i)->lattC(T,'a')) / region.getLayerMaterial(i)->lattC(T,'a');
        double tH = region.getLayerBox(i).height()*1e4; // tH (A)
        mpLay = new QW::warstwa(region.getLayerMaterial(i)->Me(T,e).c00, region.getLayerMaterial(i)->Me(T,e).c11, tX, (region.getLayerMaterial(i)->CB(T,e)-tDEc), (tX+tH), (region.getLayerMaterial(i)->CB(T,e)-tDEc)); // wells and barriers
        //writelog(LOG_DETAIL, "Ec dla warstwy %1%: %2%", i, region.getLayerMaterial(i)->CB(T,e)-tDEc);
        mpEc.push_back(mpLay); tX += tH;
        if (region.getLayerMaterial(i)->CB(T,e) >= tDEc)
            tfStructOK = false;
    }
    mpLay = new QW::warstwa_skraj(QW::warstwa_skraj::prawa, region.getLayerMaterial(tN-1)->Me(T,eClad2).c00, region.getLayerMaterial(tN-1)->Me(T,eClad2).c11, tX, (region.getLayerMaterial(tN-1)->CB(T,eClad2)-tDEc)); // right cladding
    mpEc.push_back(mpLay); // add delete somewhere! TODO

    if (tfStructOK)
        return 0; // band structure OK
    else
        return -1;
}

template <typename GeometryType>
int FerminewGainSolver<GeometryType>::buildEvhh(double T, const ActiveRegionInfo& region)
{
    //writelog(LOG_INFO, "Building Evhh");

    mpEvhh.clear();

    int tN = region.size(); // number of all layers int the active region (QW, barr, external)

    double eClad1 = 0.; // TODO
    double eClad2 = 0.; // TODO

    bool tfStructOK = true;

    double tDEvhh = region.getLayerMaterial(0)->VB(T,eClad1,'G','H'); // Ev0 for cladding

        double tX = 0.;
        mpLay = new QW::warstwa_skraj(QW::warstwa_skraj::lewa, region.getLayerMaterial(0)->Mhh(T,eClad1).c00, region.getLayerMaterial(0)->Mhh(T,eClad1).c11, tX, (-region.getLayerMaterial(0)->VB(T,eClad1,'G','H')+tDEvhh)); // left cladding
        mpEvhh.push_back(mpLay);
        for (int i=1; i<tN-1; ++i)
        {
            double e = (this->materialSubstrate->lattC(T,'a') - region.getLayerMaterial(i)->lattC(T,'a')) / region.getLayerMaterial(i)->lattC(T,'a');
            double tH = region.getLayerBox(i).height()*1e4; // tH (A)
            mpLay = new QW::warstwa(region.getLayerMaterial(i)->Mhh(T,e).c00, region.getLayerMaterial(i)->Mhh(T,e).c11, tX, (-region.getLayerMaterial(i)->VB(T,e,'G','H')+tDEvhh), (tX+tH), (-region.getLayerMaterial(i)->VB(T,e,'G','H')+tDEvhh)); // wells and barriers
            //writelog(LOG_DETAIL, "Evhh dla warstwy %1%: %2%", i, -region.getLayerMaterial(i)->VB(T,e,'G','H')+tDEvhh);
            mpEvhh.push_back(mpLay); tX += tH;
            if (region.getLayerMaterial(i)->VB(T,e,'G','H') <= tDEvhh)
                tfStructOK = false;
        }
        mpLay = new QW::warstwa_skraj(QW::warstwa_skraj::prawa, region.getLayerMaterial(tN-1)->Mhh(T,eClad2).c00, region.getLayerMaterial(tN-1)->Mhh(T,eClad2).c11, tX, (-region.getLayerMaterial(tN-1)->VB(T,eClad2,'G','H')+tDEvhh));
        mpEvhh.push_back(mpLay); // add delete somewhere! TODO

        if (tfStructOK)
            return 0; // band structure OK
        else
            return -1;
}

template <typename GeometryType>
int FerminewGainSolver<GeometryType>::buildEvlh(double T, const ActiveRegionInfo& region)
{
    //writelog(LOG_INFO, "Building Evlh");

    mpEvlh.clear();

    int tN = region.size(); // number of all layers int the active region (QW, barr, external)

    double eClad1 = 0.; // TODO
    double eClad2 = 0.; // TODO

    bool tfStructOK = true;

    double tDEvlh = region.getLayerMaterial(0)->VB(T,eClad1,'G','L'); // Ev0 for cladding

        double tX = 0.;
        mpLay = new QW::warstwa_skraj(QW::warstwa_skraj::lewa, region.getLayerMaterial(0)->Mlh(T,eClad1).c00, region.getLayerMaterial(0)->Mlh(T,eClad1).c11, tX, (-region.getLayerMaterial(0)->VB(T,eClad1,'G','L')+tDEvlh)); // left cladding
        mpEvlh.push_back(mpLay);
        for (int i=1; i<tN-1; ++i)
        {
            double e = (this->materialSubstrate->lattC(T,'a') - region.getLayerMaterial(i)->lattC(T,'a')) / region.getLayerMaterial(i)->lattC(T,'a');
            double tH = region.getLayerBox(i).height()*1e4; // tH (A)
            mpLay = new QW::warstwa(region.getLayerMaterial(i)->Mlh(T,e).c00, region.getLayerMaterial(i)->Mlh(T,e).c11, tX, (-region.getLayerMaterial(i)->VB(T,e,'G','L')+tDEvlh), (tX+tH), (-region.getLayerMaterial(i)->VB(T,e,'G','L')+tDEvlh)); // wells and barriers
            //writelog(LOG_DETAIL, "Evlh dla warstwy %1%: %2%", i, -region.getLayerMaterial(i)->VB(T,e,'G','L')+tDEvlh);
            mpEvlh.push_back(mpLay); tX += tH;
            if (region.getLayerMaterial(i)->VB(T,e,'G','L') <= tDEvlh)
                tfStructOK = false;
        }
        mpLay = new QW::warstwa_skraj(QW::warstwa_skraj::prawa, region.getLayerMaterial(tN-1)->Mlh(T,eClad2).c00, region.getLayerMaterial(tN-1)->Mlh(T,eClad2).c11, tX, (-region.getLayerMaterial(tN-1)->VB(T,eClad2,'G','L')+tDEvlh));
        mpEvlh.push_back(mpLay); // add delete somewhere! TODO

        if (tfStructOK)
            return 0; // band structure OK
        else
            return -1;
}

template <typename GeometryType> //ADAM
double FerminewGainSolver<GeometryType>::przelicz_nQW_na_npow(plask::shared_ptr<QW::obszar_aktywny> iAktyw, double iN, double iQWTotH, double iT, double iQWnR)
{
    // dobrze jakby to dzialalo

    writelog(LOG_INFO, "Do funkcji Adama poszlo %1%, %2%, %3% i %4%", iN, (iQWTotH*1*1e-7), iT, iQWnR);

    double nQW=iN;
    double iN1=0.,tn1=0.;
    int jj;
    jj = 1;
    while((1)&&(jj>0))
    {
        iN1=iN;
        QW::gain gainModule(iAktyw, iN*(iQWTotH*1*1e-7), iT, iQWnR); // (&aktyw, 3e18*8e-7, mT, mSAR->getWellNref())
        double tFlc1 = gainModule.policz_qFlc();
        double tFlv1 = gainModule.policz_qFlv();
        writelog(LOG_INFO, "Flc1 %1%", tFlc1);
        writelog(LOG_INFO, "Flv1 %1%", tFlv1);
        //std::vector<double> tN = mpStrEc->koncentracje_w_warstwach(tFe, mT);
        //tn1=QW::struktura::koncentracja_na_cm_3(tN[2]);
        //cout<<tn1<<"\n";
        if(tn1>=nQW) iN/=2.;
        else break;
        jj--;
    } writelog(LOG_INFO, "Raz %1%", iN);
    double iN2=0.,tn2=0.;
    iN*=2.;
    jj = 1;
    while((1)&&(jj>0))
    {
        iN2=iN;
        QW::gain gainModule(iAktyw, iN*(iQWTotH*1*1e-7), iT, iQWnR); // (&aktyw, 3e18*8e-7, mT, mSAR->getWellNref())
        double tFlc2 = gainModule.policz_qFlc();
        double tFlv2 = gainModule.policz_qFlv();
        writelog(LOG_INFO, "Flc2 %1%", tFlc2);
        writelog(LOG_INFO, "Flv2 %1%", tFlv2);
        //std::vector<double> tN = mpStrEc->koncentracje_w_warstwach(tFe, mT);
        //tn2=QW::struktura::koncentracja_na_cm_3(tN[2]);
        //cout<<tn2<<"\n";
        if(tn2<=nQW) {iN1=iN2; tn1=tn2; iN*=2.;}
        else break;
        jj--;
    } writelog(LOG_INFO, "Dwa %1%", iN);
    double iN3=0.,tn3=0.;
    jj = 10;
    while((1)&&(jj>0)) //metoda siecznych
    {
        iN=iN1-(tn1-nQW)*(iN2-iN1)/(tn2-tn1);
        iN3=iN;
        QW::gain gainModule(iAktyw, iN*(iQWTotH*1*1e-7), iT, iQWnR); // (&aktyw, 3e18*8e-7, mT, mSAR->getWellNref())
        double tFlc3 = gainModule.policz_qFlc();
        double tFlv3 = gainModule.policz_qFlv();
        writelog(LOG_INFO, "Flc3 %1%", tFlc3);
        writelog(LOG_INFO, "Flv3 %1%", tFlv3);
        //std::vector<double> tN = mpStrEc->koncentracje_w_warstwach(tFe, mT);
        //tn3=QW::struktura::koncentracja_na_cm_3(tN[2]);
        //cout<<tn3<<"\n";
        if(abs(tn3-nQW)/nQW<1e-6) break;
        if(tn3-nQW>0) {iN1=iN3; tn1=tn3;}
        else{iN2=iN3; tn2=tn3;}
        jj--;
    } writelog(LOG_INFO, "Trzy %1%", iN);
    return iN;
}


template <typename GeometryType>
const LazyData<double> FerminewGainSolver<GeometryType>::getGain(const shared_ptr<const MeshD<2>>& dst_mesh, double wavelength, InterpolationMethod interp)
{
    if (interp == INTERPOLATION_DEFAULT) interp = INTERPOLATION_SPLINE;

    this->writelog(LOG_INFO, "Calculating gain");
    this->initCalculation(); // This must be called before any calculation!

    auto mesh2 = make_shared<RectangularMesh<2>>();    //RectilinearMesh2D
    if (this->mesh) {
        auto verts = make_shared<RectilinearAxis>();
        for (auto p: *dst_mesh) verts->addPoint(p.vert());
        mesh2->setAxis0(this->mesh); mesh2->setAxis1(verts);
    }
    const shared_ptr<const MeshD<2>> src_mesh((this->mesh)? mesh2 : dst_mesh);

    auto geo_mesh = make_shared<const WrappedMesh<2>>(src_mesh, this->geometry);

    DataVector<const double> nOnMesh = inCarriersConcentration(geo_mesh, interp); // carriers concentration on the mesh
    DataVector<const double> TOnMesh = inTemperature(geo_mesh, interp); // temperature on the mesh
    DataVector<double> gainOnMesh(geo_mesh->size(), 0.);

    std::vector<std::pair<size_t,size_t>> points;
    for (size_t i = 0; i != geo_mesh->size(); i++)
        for (size_t r = 0; r != regions.size(); ++r)
            if (regions[r].contains(geo_mesh->at(i)) && nOnMesh[i] > 0.)
                points.push_back(std::make_pair(i,r));

    #pragma omp parallel for
    for (int j = 0; j < points.size(); j++)
    {
        size_t i = points[j].first;

        double T = TOnMesh[i];
        double n = nOnMesh[i];

        const ActiveRegionInfo& region = regions[points[j].second];

        QW::gain gainModule = getGainModule(wavelength, T, n, region);

        /*if (isnan(n) || n < 0) throw ComputationError(this->getId(), "Wrong carrier concentration (%1%/cm3)", n);
        if (isnan(T) || T < 0) throw ComputationError(this->getId(), "Wrong temperature (%1%K)", T);

        if (if_strain)
        {
            if (!this->materialSubstrate) throw ComputationError(this->getId(), "No layer with role 'substrate' has been found");

            for (int i=0; i<region.size(); ++i)
            {
                double e = (this->materialSubstrate->lattC(T,'a') - region.getLayerMaterial(i)->lattC(T,'a')) / region.getLayerMaterial(i)->lattC(T,'a');
                double tH = region.getLayerBox(i).height(); // (um) czy na pewno h powinno byc w um i czy w ten sposob uzyskuje dobra wartosc? // TODO
                writelog(LOG_RESULT, "Layer %1% - strain: %2%, thickness: %3%", i, e, tH);
            }
        }

        buildStructure(T, region);

        double tCladEg = region.getLayerMaterial(0)->CB(T,0.) - region.getLayerMaterial(0)->VB(T,0.); // cladding Eg (eV) TODO
        double tQWDso = region.materialQW->Dso(T); // QW Dso (eV)
        double tQWTotH = region.qwtotallen*0.1; // total thickness of QWs (nm)
        double tQWnR = region.materialQW->nr(wavelength,T); // QW nR

        std::vector<QW::struktura *> tHoles;
        tHoles.clear();
        if (!mEvhh)
            tHoles.push_back(&(*mpStrEvhh));
        if (!mEvlh)
            tHoles.push_back(&(*mpStrEvlh));*/
        if ((!mEc)&&((!mEvhh)||(!mEvlh)))
        {
            /*QW::obszar_aktywny aktyw(&(*mpStrEc), tHoles, tCladEg, tQWDso, roughness); // roughness = 0.05 for example // TODO
            aktyw.zrob_macierze_przejsc();
            //writelog(LOG_INFO, "Do funkcji Adama idzie %1%, %2%, %3% i %4%", n, tQWTotH, T, tQWnR);
            //n = przelicz_nQW_na_npow(aktyw, n, tQWTotH, T, tQWnR); // ADAM
            QW::gain gainModule(&aktyw, n*(tQWTotH*1e-7), T, tQWnR); // TODO
            writelog(LOG_INFO, "T %1%", T);
            writelog(LOG_INFO, "n %1%", n);
            writelog(LOG_INFO, "nPow %1%", n*(tQWTotH*1e-7));
            writelog(LOG_INFO, "tQWnR %1%", tQWnR);

            writelog(LOG_INFO, "Calculating quasi Fermi levels and carrier concentrations..");
            double tFe = gainModule.policz_qFlc();
            double tFp = gainModule.policz_qFlv();
            writelog(LOG_RESULT, "Fe %1%", tFe);
            writelog(LOG_RESULT, "Fp %1%", tFp);
            std::vector<double> tN = mpStrEc->koncentracje_w_warstwach(tFe, T);
            for(int i = 0; i <= (int) tN.size() - 1; i++)
                writelog(LOG_RESULT, "koncentracja_na_cm_3 w warstwie %1% wynosi %2%", i, QW::struktura::koncentracja_na_cm_3(tN[i]));
*/
            double L = region.qwtotallen / region.totallen; // no unit
            writelog(LOG_RESULT, "L %1%", L);
            double tGehh = gainModule.wzmocnienie_od_pary_pasm(nm_to_eV(wavelength), 0, 0) / L;
            double tGelh = gainModule.wzmocnienie_od_pary_pasm(nm_to_eV(wavelength), 0, 1) / L;
            gainOnMesh[i] = tGehh+tGelh;
            writelog(LOG_RESULT, "gainOnMesh %1%", gainOnMesh[i]);
        }
        else if (mEc)
            throw BadInput(this->getId(), "Conduction QW depth negative for e, check VB values of active-region materials");
        else //if ((mEvhh)&&(mEvlh))
            throw BadInput(this->getId(), "Valence QW depth negative both for hh and lh, check VB values of active-region materials");
    }

    if (this->mesh)
    {
        return interpolate(mesh2, gainOnMesh, make_shared<const WrappedMesh<2>>(dst_mesh, this->geometry), interp);
    }
    else
    {
        return gainOnMesh;
    }
} // LUKASZ

/*
template <typename GeometryType>
const DataVector<double> FerminewGainSolver<GeometryType>::getdGdn(const MeshD<2>& dst_mesh, double wavelength, InterpolationMethod interp)
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
}*/ // LUKASZ


template <typename GeometryType>
GainSpectrum<GeometryType> FerminewGainSolver<GeometryType>::getGainSpectrum(const Vec<2>& point)
{
    this->initCalculation();
    return GainSpectrum<GeometryType>(this, point);
}


template <> std::string FerminewGainSolver<Geometry2DCartesian>::getClassName() const { return "gain.Ferminew2D"; }
template <> std::string FerminewGainSolver<Geometry2DCylindrical>::getClassName() const { return "gain.FerminewCyl"; }

template struct FerminewGainSolver<Geometry2DCartesian>;
template struct FerminewGainSolver<Geometry2DCylindrical>;

}}} // namespace
