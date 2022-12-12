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
#include "ferminew.hpp"

namespace plask { namespace solvers { namespace FermiNew {

template <typename T> struct ptrVector {
    std::vector<T*> data;
    ~ptrVector() {
        for (T* ptr : data) delete ptr;
    }
};

template <typename GeometryType>
FermiNewGainSolver<GeometryType>::FermiNewGainSolver(const std::string& name)
    : SolverWithMesh<GeometryType, MeshAxis>(name),
      outGain(this, &FermiNewGainSolver<GeometryType>::getGain),
      outLuminescence(this, &FermiNewGainSolver<GeometryType>::getLuminescence) {
    Tref = 300.;                // [K], only for this temperature energy levels are calculated
    inTemperature = 300.;       // temperature receiver has some sensible value
    condQWshift = 0.;           // [eV]
    valeQWshift = 0.;           // [eV]
    QWwidthMod = 100.;          // [-] (if equal to 10 - differences in QW widths are to big)
    roughness = 1.00;           // [-]
    lifetime = 0.0;             // [ps]
    matrixElem = 0.;            // [m0*eV]
    differenceQuotient = 0.01;  // [%]
    strains = false;
    build_struct_once = true;
    adjust_widths = true;
    inTemperature.changedConnectMethod(this, &FermiNewGainSolver<GeometryType>::onInputChange);
    inCarriersConcentration.changedConnectMethod(this, &FermiNewGainSolver<GeometryType>::onInputChange);
}

template <typename GeometryType> FermiNewGainSolver<GeometryType>::~FermiNewGainSolver() {
    disconnectModGeometry();
    inTemperature.changedDisconnectMethod(this, &FermiNewGainSolver<GeometryType>::onInputChange);
    inCarriersConcentration.changedDisconnectMethod(this, &FermiNewGainSolver<GeometryType>::onInputChange);
}

template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::loadConfiguration(XMLReader& reader, Manager& manager) {
    // Load a configuration parameter from XML.
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "config") {
            roughness = reader.getAttribute<double>("roughness", roughness);
            lifetime = reader.getAttribute<double>("lifetime", lifetime);
            matrixElem = reader.getAttribute<double>("matrix-elem", matrixElem);
            condQWshift = reader.getAttribute<double>("cond-qw-shift", condQWshift);
            valeQWshift = reader.getAttribute<double>("vale-qw-shift", valeQWshift);
            Tref = reader.getAttribute<double>("Tref", Tref);
            strains = reader.getAttribute<bool>("strained", strains);
            adjust_widths = reader.getAttribute<bool>("adjust-layers", adjust_widths);
            build_struct_once = reader.getAttribute<bool>("fast-levels", build_struct_once);
            reader.requireTagEnd();
            // } else if (param == "levels") {
            // std::string els, hhs, lhs;
            // if (reader.hasAttribute("el") || reader.hasAttribute("hh") || reader.hasAttribute("lh")) {
            //     els = reader.requireAttribute("el");
            //     hhs = reader.requireAttribute("hh");
            //     lhs = reader.requireAttribute("lh");
            //     reader.requireTagEnd();
            // } else {
            //     while (reader.requireTagOrEnd()) {
            //         if (reader.getNodeName() == "el") els = reader.requireTextInCurrentTag();
            //         else if (reader.getNodeName() == "hh") hhs = reader.requireTextInCurrentTag();
            //         else if (reader.getNodeName() == "lh") lhs = reader.requireTextInCurrentTag();
            //         else throw XMLUnexpectedElementException(reader, "<el>, <hh>, or <lh>");
            //     }
            //     if (els == "") throw XMLUnexpectedElementException(reader, "<el>");
            //     if (hhs == "") throw XMLUnexpectedElementException(reader, "<hh>");
            //     if (lhs == "") throw XMLUnexpectedElementException(reader, "<lh>");
            // }
            // boost::char_separator<char> sep(", ");
            // boost::tokenizer<boost::char_separator<char>> elt(els, sep), hht(hhs, sep), lht(lhs, sep);
            // /*double *el = nullptr, *hh = nullptr, *lh = nullptr;
            // try {
            //     el = new double[std::distance(elt.begin(), elt.end())+1];
            //     hh = new double[std::distance(hht.begin(), hht.end())+1];
            //     lh = new double[std::distance(lht.begin(), lht.end())+1];
            //     double* e = el; for (const auto& i: elt) *(e++) = - boost::lexical_cast<double>(i); *e = 1.;
            //     double* h = hh; for (const auto& i: hht) *(h++) = - boost::lexical_cast<double>(i); *h = 1.;
            //     double* l = lh; for (const auto& i: lht) *(l++) = - boost::lexical_cast<double>(i); *l = 1.;
            // } catch(...) {
            //     delete[] el; delete[] hh; delete[] lh;
            // }*/
            // std::unique_ptr<double[]> el(new double[std::distance(elt.begin(), elt.end())+1]);
            // std::unique_ptr<double[]> hh(new double[std::distance(hht.begin(), hht.end())+1]);
            // std::unique_ptr<double[]> lh(new double[std::distance(lht.begin(), lht.end())+1]);
            // double* e = el.get(); for (const auto& i: elt) *(e++) = - boost::lexical_cast<double>(i); *e = 1.;
            // double* h = hh.get(); for (const auto& i: hht) *(h++) = - boost::lexical_cast<double>(i); *h = 1.;
            // double* l = lh.get(); for (const auto& i: lht) *(l++) = - boost::lexical_cast<double>(i); *l = 1.;
            // /*if (extern_levels) {
            //     delete[] extern_levels->el; delete[] extern_levels->hh; delete[] extern_levels->lh;
            // }
            // extern_levels.reset(ExternalLevels(el.release(), hh.release(), lh.release()));*/
        } else {
            if (param == "geometry") {
                auto name = reader.getAttribute("mod");
                if (name) {
                    auto found = manager.geometrics.find(*name);
                    if (found == manager.geometrics.end())
                        throw BadInput(this->getId(), "Geometry '{0}' not found", *name);
                    else {
                        auto geometry = dynamic_pointer_cast<GeometryType>(found->second);
                        if (!geometry) throw BadInput(this->getId(), "Geometry '{0}' of wrong type", *name);
                        this->setModGeometry(geometry);
                    }
                }
            }
            this->parseStandardConfiguration(reader, manager, "<geometry>, <mesh>, <levels>, or <config>");
        }
    }
}

template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::onInitialize()  // In this function check if geometry and mesh are set
{
    if (!this->geometry) throw NoGeometryException(this->getId());

    prepareActiveRegionsInfo();
    if (build_struct_once) {
        region_levels.resize(regions.size());
    }

    outGain.fireChanged();
    outLuminescence.fireChanged();
}

template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::onInvalidate()  // This will be called when e.g. geometry or mesh changes and
                                                       // your results become outdated
{
    region_levels.clear();
}

/*template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::compute() {
    this->initCalculation(); // This must be called before any calculation!
}*/

template <typename GeometryType>
std::list<typename FermiNewGainSolver<GeometryType>::ActiveRegionData>
FermiNewGainSolver<GeometryType>::detectActiveRegions(const shared_ptr<GeometryType>& geometry) {
    std::list<typename FermiNewGainSolver<GeometryType>::ActiveRegionData> regions;

    shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(geometry->getChild());
    shared_ptr<RectangularMesh<2>> points = mesh->getElementMesh();

    size_t ileft = 0, iright = points->axis[0]->size();
    bool in_active = false;

    bool added_bottom_cladding = false;
    bool added_top_cladding = false;

    for (size_t r = 0; r < points->axis[1]->size(); ++r) {
        bool had_active = false;  // indicates if we had active region in this layer
        shared_ptr<Material> layer_material;
        bool layer_QW = false;

        for (size_t c = 0; c < points->axis[0]->size(); ++c) {  // In the (possible) active region
            auto point = points->at(c, r);
            auto tags = geometry->getRolesAt(point);
            bool active = false;
            for (const auto& tag : tags)
                if (tag.substr(0, 6) == "active") {
                    active = true;
                    break;
                }
            bool QW = tags.find("QW") != tags.end() /* || tags.find("QD") != tags.end()*/;
            bool substrate = tags.find("substrate") != tags.end();

            if (substrate) {
                if (!materialSubstrate)
                    materialSubstrate = geometry->getMaterial(point);
                else if (*materialSubstrate != *geometry->getMaterial(point))
                    throw Exception("{0}: Non-uniform substrate layer.", this->getId());
            }

            if (QW && !active)
                throw Exception("{0}: All marked quantum wells must belong to marked active region.", this->getId());

            if (c < ileft) {
                if (active) throw Exception("{0}: Left edge of the active region not aligned.", this->getId());
            } else if (c >= iright) {
                if (active) throw Exception("{0}: Right edge of the active region not aligned.", this->getId());
            } else {
                // Here we are inside potential active region
                if (active) {
                    if (!had_active) {
                        if (!in_active) {  // active region is starting set-up new region info
                            regions.emplace_back(mesh->at(c, r));
                            added_bottom_cladding = (tags.find("cladding") != tags.end());
                            added_top_cladding = false;
                            ileft = c;
                        } else {
                            if (added_top_cladding)
                                throw Exception(
                                    "{0}: Only the first or the last layer in an active region can have 'cladding' "
                                    "role",
                                    this->getId());
                            if (tags.find("cladding") != tags.end()) added_top_cladding = true;
                        }
                        layer_material = geometry->getMaterial(point);
                        layer_QW = QW;
                    } else {
                        if (*layer_material != *geometry->getMaterial(point))
                            throw Exception("{0}: Non-uniform active region layer.", this->getId());
                        if (layer_QW != QW)
                            throw Exception("{0}: Quantum-well role of the active region layer not consistent.",
                                            this->getId());
                    }
                } else if (had_active) {
                    if (!in_active) {
                        iright = c;
                    } else
                        throw Exception("{0}: Right edge of the active region not aligned.", this->getId());
                }
                had_active |= active;
            }
        }
        in_active = had_active;

        // Now fill-in the layer info
        if (!regions.empty()) {
            ActiveRegionData& region = regions.back();
            if (!added_bottom_cladding) {
                if (r == 0)
                    throw Exception("{0}: Active region cannot start from the edge of the structure.", this->getId());
                // add layer below active region (cladding) LUKASZ
                auto bottom_material = geometry->getMaterial(points->at(ileft, r - 1));
                for (size_t cc = ileft; cc < iright; ++cc)
                    if (*geometry->getMaterial(points->at(cc, r - 1)) != *bottom_material)
                        throw Exception("{0}: Material below active region not uniform.", this->getId());
                double w = mesh->axis[0]->at(iright) - mesh->axis[0]->at(ileft);
                double h = mesh->axis[1]->at(r) - mesh->axis[1]->at(r - 1);
                region.origin += Vec<2>(0., -h);
                // this->writelog(LOG_DEBUG, "Adding bottom cladding; h = {0}",h);
                region.layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w, h), bottom_material));
                added_bottom_cladding = true;
            }

            double h = mesh->axis[1]->at(r + 1) - mesh->axis[1]->at(r);
            double w = mesh->axis[0]->at(iright) - mesh->axis[0]->at(ileft);
            if (in_active) {
                size_t n = region.layers->getChildrenCount();
                shared_ptr<Block<2>> last;
                if (n > 0)
                    last = static_pointer_cast<Block<2>>(
                        static_pointer_cast<Translation<2>>(region.layers->getChildNo(n - 1))->getChild());
                assert(!last || last->size.c0 == w);
                if (last && layer_material == last->getRepresentativeMaterial() &&
                    layer_QW == region.isQW(region.size() - 1)) {
                    // TODO check if usage of getRepresentativeMaterial is fine here (was material)
                    last->setSize(w, last->size.c1 + h);
                } else {
                    auto layer = plask::make_shared<Block<2>>(Vec<2>(w, h), layer_material);
                    if (layer_QW) layer->addRole("QW");
                    region.layers->push_back(layer);
                    // if (layer_QW) this->writelog(LOG_DEBUG, "Adding qw; h = {0}",h);
                    // else this->writelog(LOG_DEBUG, "Adding barrier; h = {0}",h);
                }
            } else {
                if (!added_top_cladding) {
                    // add layer above active region (top cladding)
                    auto top_material = geometry->getMaterial(points->at(ileft, r));
                    for (size_t cc = ileft; cc < iright; ++cc)
                        if (*geometry->getMaterial(points->at(cc, r)) != *top_material)
                            throw Exception("{0}: Material above quantum well not uniform.", this->getId());
                    region.layers->push_back(plask::make_shared<Block<2>>(Vec<2>(w, h), top_material));
                    // this->writelog(LOG_DEBUG, "Adding top cladding; h = {0}",h);

                    ileft = 0;
                    iright = points->axis[0]->size();
                    added_top_cladding = true;
                }
            }
        }
    }
    if (!regions.empty() && regions.back().isQW(regions.back().size() - 1))
        throw Exception("{0}: Quantum-well at the edge of the structure.", this->getId());

    this->writelog(LOG_INFO, "Found {0} active region{1}", regions.size(), (regions.size() == 1) ? "" : "s");
    size_t n = 0;
    for (auto& region : regions) {
        if (region.layers->getChildrenCount() <= 2) throw Exception("Not enough layers in the active region {}", n);
        region.summarize(this);
        this->writelog(LOG_INFO, "Active region {0}: {1} nm single QW, {2} nm all QW, {3} nm total", n++,
                       0.1 * region.qwlen, 0.1 * region.qwtotallen, 0.1 * region.totallen);
    }

    // energy levels for active region with two identical QWs won't be calculated so QW widths must be changed
    this->writelog(LOG_DETAIL, "Updating QW widths");
    n = 0;
    for (auto& region : regions) {
        region.lens.clear();
        int N = region.size();  // number of all layers in the active region (QW, barr, external)
        int nQW = 0;            // number of QWs counter
        for (int i = 0; i < N; ++i) {
            if (region.isQW(i)) {
                nQW++;
                region.QWs.insert(i - 1);
            }
            region.lens.push_back(region.getLayerBox(i).height() * 1e4);  // in [A]
            this->writelog(LOG_DEBUG, "Layer {0} thickness: {1} nm", i + 1, 0.1 * region.lens[i]);
        }
        this->writelog(LOG_DEBUG, "Number of QWs in the above active region: {0}", nQW);
        this->writelog(LOG_DEBUG, "QW initial thickness: {0} nm", 0.1 * region.qwlen);

        if (adjust_widths) {
            double hstep = region.qwlen / QWwidthMod;
            if (!(nQW % 2)) {
                double dh0 = -(floor(nQW / 2) + 0.5) * hstep;
                for (int i = 0; i < N; ++i) {
                    if (region.isQW(i)) {
                        region.lens[i] += dh0;
                        this->writelog(LOG_DEBUG, "Layer {0} thickness: {1} nm", i + 1, 0.1 * region.lens[i]);
                        dh0 += hstep;
                    }
                }
            } else {
                double dh0 = -(int(nQW / 2)) * hstep;
                for (int i = 0; i < N; ++i) {
                    if (region.isQW(i)) {
                        region.lens[i] += dh0;
                        this->writelog(LOG_DEBUG, "Layer {0} modified thickness: {1} nm", i + 1, 0.1 * region.lens[i]);
                        dh0 += hstep;
                    }
                }
            }
            this->writelog(LOG_DEBUG, "QW thickness step: {0} nm", 0.1 * hstep);
        }
        n++;
    }

    return regions;
}

template <typename GeometryType> void FermiNewGainSolver<GeometryType>::prepareActiveRegionsInfo() {
    auto regs = detectActiveRegions(this->geometry);
    regions.clear();
    regions.reserve(regs.size());
    regions.assign(regs.begin(), regs.end());
    if (geometry_mod) {
        regs = detectActiveRegions(this->geometry_mod);
        if (regs.size() != regions.size())
            throw Exception("Modified geometry has different number of active regions ({}) than the main one ({})",
                            regs.size(), regions.size());
        auto region = regions.begin();
        for (const auto& reg : regs) (region++)->mod.reset(std::move(reg));
    }
}

template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::findEnergyLevels(Levels& levels,
                                                        const ActiveRegionInfo& region,
                                                        double T,
                                                        bool showDetails) {
    if (isnan(T) || T < 0.) throw ComputationError(this->getId(), "Wrong temperature ({0}K)", T);

    this->writelog(LOG_DETAIL, "Determining energy levels");

    buildStructure(T, region, levels.bandsEc, levels.bandsEvhh, levels.bandsEvlh, showDetails);

    std::vector<double> dso;
    dso.reserve(region.size());
    for (int i = 0; i < region.size(); ++i) dso.push_back(region.getLayerMaterial(i)->Dso(T));

    std::vector<kubly::struktura*> holes;
    holes.reserve(2);
    if (levels.bandsEvhh) holes.push_back(levels.bandsEvhh.get());
    if (levels.bandsEvlh) holes.push_back(levels.bandsEvlh.get());

    levels.Eg = region.getLayerMaterial(0)->CB(T, 0.) - region.getLayerMaterial(0)->VB(T, 0.);  // cladding Eg (eV)

    if (region.mod) {
        buildStructure(T, *region.mod, levels.modbandsEc, levels.modbandsEvhh, levels.modbandsEvlh, showDetails);

        std::vector<kubly::struktura*> modholes;
        modholes.reserve(2);
        if (levels.modbandsEvhh) modholes.push_back(levels.modbandsEvhh.get());
        if (levels.modbandsEvlh) modholes.push_back(levels.modbandsEvlh.get());
        levels.activeRegion =
            plask::make_shared<kubly::obszar_aktywny>(levels.bandsEc.get(), holes, levels.modbandsEc.get(), modholes,
                                                      levels.Eg, dso, roughness, matrixElem, T);
    } else {
        levels.activeRegion = plask::make_shared<kubly::obszar_aktywny>(levels.bandsEc.get(), holes, levels.Eg, dso,
                                                                        roughness, matrixElem, T);
    }
    levels.activeRegion->zrob_macierze_przejsc();
}

template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::buildStructure(double T,
                                                      const ActiveRegionData& region,
                                                      std::unique_ptr<kubly::struktura>& bandsEc,
                                                      std::unique_ptr<kubly::struktura>& bandsEvhh,
                                                      std::unique_ptr<kubly::struktura>& bandsEvlh,
                                                      bool showDetails) {
    if (strains) {
        if (!this->materialSubstrate)
            throw ComputationError(this->getId(), "No layer with role 'substrate' has been found");
        if (showDetails)
            for (int i = 0; i < region.size(); ++i) {
                double e = (this->materialSubstrate->lattC(T, 'a') - region.getLayerMaterial(i)->lattC(T, 'a')) /
                           region.getLayerMaterial(i)->lattC(T, 'a');
                if ((i == 0) || (i == region.size()-1))
                    e = 0.;
                this->writelog(LOG_DEBUG, "Layer {0} - strain: {1}{2}", i + 1, e * 100., '%');
            }
    }

    kubly::struktura* Ec = buildEc(T, region, showDetails);
    kubly::struktura* Evhh = buildEvhh(T, region, showDetails);
    kubly::struktura* Evlh = buildEvlh(T, region, showDetails);

    if (!Ec)
        throw BadInput(this->getId(),
                       "Conduction QW depth negative for electrons, check CB values of active-region materials");
    if (!Evhh && !Evlh)
        throw BadInput(this->getId(),
                       "Valence QW depth negative for both heavy holes and light holes, check VB values of "
                       "active-region materials");

    bandsEc.reset(Ec);
    bandsEc->gwiazdki.reserve(region.QWs.size());
    bandsEc->gwiazdki.assign(region.QWs.begin(), region.QWs.end());

    if (Evhh) {
        bandsEvhh.reset(Evhh);
        bandsEvhh->gwiazdki.reserve(region.QWs.size());
        bandsEvhh->gwiazdki.assign(region.QWs.begin(), region.QWs.end());
    }
    if (Evlh) {
        bandsEvlh.reset(Evlh);
        bandsEvlh->gwiazdki.reserve(region.QWs.size());
        bandsEvlh->gwiazdki.assign(region.QWs.begin(), region.QWs.end());
    }
}

template <typename GeometryType>
kubly::struktura* FermiNewGainSolver<GeometryType>::buildEc(double T,
                                                            const ActiveRegionData& region,
                                                            bool showDetails) {
    ptrVector<kubly::warstwa> levels;

    int N = region.size();  // number of all layers in the active region (QW, barr, external)

    double lattSub, straine = 0.;

    if (strains) {
        lattSub = this->materialSubstrate->lattC(T, 'a');
    }

    double DEc = region.getLayerMaterial(0)->CB(T, 0.);  // Ec0 for cladding

    double x = 0.;
    double Ec = region.getLayerMaterial(0)->CB(T, 0.) - DEc;
    if (showDetails) this->writelog(LOG_DEBUG, "Layer {0} CB: {1} eV", 1, region.getLayerMaterial(0)->CB(T, 0.));
    levels.data.emplace_back(new kubly::warstwa_skraj(kubly::warstwa_skraj::lewa,
                                                      region.getLayerMaterial(0)->Me(T, 0.).c11,
                                                      region.getLayerMaterial(0)->Me(T, 0.).c00, x,
                                                      Ec));  // left cladding
    for (int i = 1; i < N - 1; ++i) {
        if (strains) straine = lattSub / region.getLayerMaterial(i)->lattC(T, 'a') - 1.;
        double h = region.lens[i];  // tH (A)
        double CBshift = 0.;
        if (region.isQW(i)) CBshift = condQWshift;
        Ec = region.getLayerMaterial(i)->CB(T, straine) + CBshift - DEc;
        if (showDetails)
            this->writelog(LOG_DEBUG, "Layer {0} CB: {1} eV", i + 1,
                           region.getLayerMaterial(i)->CB(T, straine) + CBshift);
        levels.data.emplace_back(new kubly::warstwa(region.getLayerMaterial(i)->Me(T, straine).c11,
                                                    region.getLayerMaterial(i)->Me(T, straine).c00, x, Ec, (x + h),
                                                    Ec));  // wells and barriers
        x += h;
        if (region.getLayerMaterial(i)->CB(T, straine) > DEc) return nullptr;
    }

    Ec = (region.getLayerMaterial(N - 1)->CB(T, 0.) - DEc);
    if (showDetails)
        this->writelog(LOG_DEBUG, "Layer {0} CB: {1} eV", N, region.getLayerMaterial(N - 1)->CB(T, 0.));
    levels.data.emplace_back(new kubly::warstwa_skraj(kubly::warstwa_skraj::prawa,
                                                      region.getLayerMaterial(N - 1)->Me(T, 0.).c11,
                                                      region.getLayerMaterial(N - 1)->Me(T, 0.).c00, x,
                                                      Ec));  // right cladding

    this->writelog(LOG_DETAIL, "Computing energy levels for electrons");
    return new kubly::struktura(levels.data, kubly::struktura::el);
}

template <typename GeometryType>
kubly::struktura* FermiNewGainSolver<GeometryType>::buildEvhh(double T,
                                                              const ActiveRegionData& region,
                                                              bool showDetails) {
    ptrVector<kubly::warstwa> levels;

    int N = region.size();  // number of all layers int the active region (QW, barr, external)

    double lattSub, straine = 0.;

    if (strains) {
        lattSub = this->materialSubstrate->lattC(T, 'a');
    }

    double DEvhh = region.getLayerMaterial(0)->VB(T, 0., '*', 'H');  // Ev0 for cladding

    double x = 0.;
    double Evhh = -(region.getLayerMaterial(0)->VB(T, 0., '*', 'H') - DEvhh);
    if (showDetails)
        this->writelog(LOG_DEBUG, "Layer {0} VB(hh): {1} eV", 1, region.getLayerMaterial(0)->VB(T, 0., '*', 'H'));
    levels.data.emplace_back(new kubly::warstwa_skraj(kubly::warstwa_skraj::lewa,
                                                      region.getLayerMaterial(0)->Mhh(T, 0.).c11,
                                                      region.getLayerMaterial(0)->Mhh(T, 0.).c00, x,
                                                      Evhh));  // left cladding
    for (int i = 1; i < N - 1; ++i) {
        if (strains) straine = lattSub / region.getLayerMaterial(i)->lattC(T, 'a') - 1.;
        double h = region.lens[i];  // h (A)
        double VBshift = 0.;
        if (region.isQW(i)) VBshift = valeQWshift;
        Evhh = -(region.getLayerMaterial(i)->VB(T, straine, '*', 'H') + VBshift - DEvhh);
        if (showDetails)
            this->writelog(LOG_DEBUG, "Layer {0} VB(hh): {1} eV", i + 1,
                           region.getLayerMaterial(i)->VB(T, straine, '*', 'H') + VBshift);
        levels.data.emplace_back(new kubly::warstwa(region.getLayerMaterial(i)->Mhh(T, straine).c11,
                                                    region.getLayerMaterial(i)->Mhh(T, straine).c00, x, Evhh, (x + h),
                                                    Evhh));  // wells and barriers
        x += h;
        if (region.getLayerMaterial(i)->VB(T, straine, '*', 'H') < DEvhh) return nullptr;
    }

    Evhh = -(region.getLayerMaterial(N - 1)->VB(T, 0., '*', 'H') - DEvhh);
    if (showDetails)
        this->writelog(LOG_DEBUG, "Layer {0} VB(hh): {1} eV", N,
                       region.getLayerMaterial(N - 1)->VB(T, 0., '*', 'H'));
    levels.data.emplace_back(new kubly::warstwa_skraj(kubly::warstwa_skraj::prawa,
                                                      region.getLayerMaterial(N - 1)->Mhh(T, 0.).c11,
                                                      region.getLayerMaterial(N - 1)->Mhh(T, 0.).c00, x, Evhh));

    this->writelog(LOG_DETAIL, "Computing energy levels for heavy holes");
    return new kubly::struktura(levels.data, kubly::struktura::hh);
}

template <typename GeometryType>
kubly::struktura* FermiNewGainSolver<GeometryType>::buildEvlh(double T,
                                                              const ActiveRegionData& region,
                                                              bool showDetails) {
    ptrVector<kubly::warstwa> levels;

    int N = region.size();  // number of all layers int the active region (QW, barr, external)

    double lattSub, straine = 0.;

    if (strains) {
        lattSub = this->materialSubstrate->lattC(T, 'a');
    }

    double DEvlh = region.getLayerMaterial(0)->VB(T, 0., '*', 'L');  // Ev0 for cladding

    double x = 0.;
    double Evlh = -(region.getLayerMaterial(0)->VB(T, 0., '*', 'L') - DEvlh);
    if (showDetails)
        this->writelog(LOG_DEBUG, "Layer {0} VB(lh): {1} eV", 1, region.getLayerMaterial(0)->VB(T, 0., '*', 'L'));
    levels.data.emplace_back(new kubly::warstwa_skraj(kubly::warstwa_skraj::lewa,
                                                      region.getLayerMaterial(0)->Mlh(T, 0.).c11,
                                                      region.getLayerMaterial(0)->Mlh(T, 0.).c00, x,
                                                      Evlh));  // left cladding
    for (int i = 1; i < N - 1; ++i) {
        if (strains) straine = lattSub / region.getLayerMaterial(i)->lattC(T, 'a') - 1.;
        double h = region.lens[i];  // tH (A)
        double VBshift = 0.;
        if (region.isQW(i)) VBshift = valeQWshift;
        Evlh = -(region.getLayerMaterial(i)->VB(T, straine, '*', 'L') + VBshift - DEvlh);
        if (showDetails)
            this->writelog(LOG_DEBUG, "Layer {0} VB(lh): {1} eV", i + 1,
                           region.getLayerMaterial(i)->VB(T, straine, '*', 'L') + VBshift);
        levels.data.emplace_back(new kubly::warstwa(region.getLayerMaterial(i)->Mlh(T, straine).c11,
                                                    region.getLayerMaterial(i)->Mlh(T, straine).c00, x, Evlh, (x + h),
                                                    Evlh));  // wells and barriers
        x += h;
        if (region.getLayerMaterial(i)->VB(T, straine, '*', 'L') < DEvlh) return nullptr;
        ;
    }

    Evlh = -(region.getLayerMaterial(N - 1)->VB(T, 0., '*', 'L') - DEvlh);
    if (showDetails)
        this->writelog(LOG_DEBUG, "Layer {0} VB(lh): {1} eV", N,
                       region.getLayerMaterial(N - 1)->VB(T, 0., '*', 'L'));
    levels.data.emplace_back(new kubly::warstwa_skraj(kubly::warstwa_skraj::prawa,
                                                      region.getLayerMaterial(N - 1)->Mlh(T, 0.).c11,
                                                      region.getLayerMaterial(N - 1)->Mlh(T, 0.).c00, x, Evlh));

    this->writelog(LOG_DETAIL, "Computing energy levels for light holes");
    return new kubly::struktura(levels.data, kubly::struktura::lh);
}

template <typename GeometryType>
void FermiNewGainSolver<GeometryType>::showEnergyLevels(std::string str,
                                                        const std::unique_ptr<kubly::struktura>& structure,
                                                        double nQW) {
    std::vector<kubly::stan>::const_iterator it_stan = structure->rozwiazania.begin();
    for (int iQW = 1; it_stan != structure->rozwiazania.end(); iQW++) {
        bool calc_avg = true;
        double avg_energy_level = 0.;
        for (int i = 0; i < nQW; i++) {
            avg_energy_level += it_stan->poziom;
            this->writelog(plask::LOG_DETAIL, "QW {0} - energy level for {1}: {2} eV from cladding band edge", iQW, str,
                           it_stan->poziom);
            it_stan++;
            if (it_stan == structure->rozwiazania.end()) {
                calc_avg = false;
                break;
            }
        }
        if (calc_avg)
            this->writelog(plask::LOG_DETAIL, "QW {0} - average energy level for {1}: {2} eV from cladding band edge",
                           iQW, str, avg_energy_level / nQW);
    }
}

template <typename GeometryType>
kubly::wzmocnienie FermiNewGainSolver<GeometryType>::getGainModule(double wavelength,
                                                                   double T,
                                                                   double n,
                                                                   const ActiveRegionInfo& region,
                                                                   const Levels& levels,
                                                                   bool showDetails) {
    if (isnan(T) || T < 0.) throw ComputationError(this->getId(), "Wrong temperature ({0}K)", T);
    if (isnan(n)) throw ComputationError(this->getId(), "Wrong carriers concentration ({0}/cm3)", n);
    n = max(n, 1e-6);  // To avoid hangs

    double QWh = region.qwtotallen * 0.1;  // total thickness of QWs (nm)

    // calculating nR
    double QWnr = 0.;
    int nQW = 0;
    int N = region.size();  // number of all layers int the active region (QW, barr, external)
    for (int i = 1; i < N - 1; ++i) {
        if (region.isQW(i)) {
            QWnr += region.getLayerMaterial(i)->nr(wavelength, T);
            nQW++;
        }
    }
    QWnr /= nQW;

    double Eg = region.getLayerMaterial(0)->CB(T, 0.) - region.getLayerMaterial(0)->VB(T, 0.);
    // TODO Add strain
    double deltaEg = Eg - levels.Eg;

    // this->writelog(LOG_DEBUG, "Creating gain module");
    kubly::wzmocnienie gainModule(
        levels.activeRegion.get(), n * (QWh * 1e-7), T, QWnr, deltaEg, 10. * QWh,  // QWh in Å
        region.mod ? kubly::wzmocnienie::Z_POSZERZENIEM : kubly::wzmocnienie::Z_CHROPOWATOSCIA);

    // this->writelog(LOG_DEBUG, "Recalculating carrier concentrations..");
    // double step = 1e-1 * n;  // TODO
    // double concQW = n;
    // double in1 = 0., tn1 = 0.;

    // for (int i = 0; i < 5; i++) {
    //     while (1) {
    //         in1 = n * 10.;

    //         gainModule.nosniki_c = gainModule.nosniki_v = gainModule.przel_gest_z_cm2(in1 * QWh * 1e-7);
    //         double qFc = gainModule.qFlc;
    //         // double tFlv = gainModule.qFlv;

    //         std::vector<double> tN = levels.bandsEc->koncentracje_w_warstwach(qFc, T);
    //         auto maxElem = std::max_element(tN.begin(), tN.end());
    //         tn1 = *maxElem;
    //         tn1 = kubly::struktura::koncentracja_na_cm_3(tn1);
    //         if (tn1 >= concQW)
    //             n -= step;
    //         else {
    //             n += step;
    //             break;
    //         }
    //     }
    //     step *= 0.1;
    //     n -= step;
    // }

    double qFc = gainModule.qFlc;
    double qFv = gainModule.qFlv;
    if (showDetails) {
        this->writelog(LOG_DEBUG, "Quasi-Fermi level for electrons: {0} eV from cladding conduction band edge", qFc);
        this->writelog(LOG_DEBUG, "Quasi-Fermi level for holes: {0} eV from cladding valence band edge", -qFv);
        std::vector<double> ne = levels.bandsEc->koncentracje_w_warstwach(qFc, T);
        std::vector<double> nlh = levels.bandsEvlh->koncentracje_w_warstwach(-qFv, T);
        std::vector<double> nhh = levels.bandsEvhh->koncentracje_w_warstwach(-qFv, T);
        for (int i = 0; i <= (int)ne.size() - 1; i++)
            this->writelog(LOG_DEBUG, "Carriers concentration in layer {:d} [cm(-3)]: el:{:.3e} lh:{:.3e} hh:{:.3e} ",
                           i + 1, kubly::struktura::koncentracja_na_cm_3(ne[i]),
                           kubly::struktura::koncentracja_na_cm_3(nlh[i]),
                           kubly::struktura::koncentracja_na_cm_3(nhh[i]));
    }

    return gainModule;
}

static const shared_ptr<OrderedAxis> zero_axis(new OrderedAxis({0.}));

/// Base for lazy data implementation
template <typename GeometryT, typename T> struct DataBase : public LazyDataImpl<T> {
    struct AveragedData {
        shared_ptr<const RectangularMesh<2>> mesh;
        LazyData<double> data;
        double factor;
        const FermiNewGainSolver<GeometryT>* solver;
        const char* name;
        AveragedData(const FermiNewGainSolver<GeometryT>* solver,
                     const char* name,
                     const shared_ptr<const MeshAxis>& haxis,
                     const typename FermiNewGainSolver<GeometryT>::ActiveRegionInfo& region)
            : solver(solver), name(name) {
            auto vaxis = plask::make_shared<OrderedAxis>();
            for (size_t n = 0; n != region.size(); ++n) {
                if (region.isQW(n)) {
                    auto box = region.getLayerBox(n);
                    vaxis->addPoint(0.5 * (box.lower.c1 + box.upper.c1));
                }
            }
            mesh = plask::make_shared<const RectangularMesh<2>>(const_pointer_cast<MeshAxis>(haxis), vaxis,
                                                                RectangularMesh<2>::ORDER_01);
            factor = 1. / vaxis->size();
        }
        size_t size() const { return mesh->axis[0]->size(); }
        double operator[](size_t i) const {
            double val = 0.;
            for (size_t j = 0; j != mesh->axis[1]->size(); ++j) {
                auto v = data[mesh->index(i, j)];
                if (isnan(v))
                    throw ComputationError(solver->getId(), "Wrong {0} ({1}) at {2}", name, v, mesh->at(i, j));
                v = max(v, 1e-6);  // To avoid hangs
                val += v;
            }
            return val * factor;
        }
    };

    FermiNewGainSolver<GeometryT>* solver;        ///< Solver
    std::vector<shared_ptr<MeshAxis>> regpoints;  ///< Points in each active region
    std::vector<LazyData<double>> data;           ///< Computed interpolations in each active region
    shared_ptr<const MeshD<2>> dest_mesh;         ///< Destination mesh

    void setupFromAxis(const shared_ptr<MeshAxis>& axis) {
        regpoints.reserve(solver->regions.size());
        InterpolationFlags flags(solver->geometry);
        for (size_t r = 0; r != solver->regions.size(); ++r) {
            std::set<double> pts;
            auto box = solver->regions[r].getBoundingBox();
            double y = 0.5 * (box.lower.c1 + box.upper.c1);
            for (double x : *axis) {
                auto p = flags.wrap(vec(x, y));
                if (solver->regions[r].contains(p)) pts.insert(p.c0);
            }
            auto msh = plask::make_shared<OrderedAxis>();
            msh->addOrderedPoints(pts.begin(), pts.end(), pts.size());
            regpoints.emplace_back(std::move(msh));
        }
    }

    DataBase(FermiNewGainSolver<GeometryT>* solver, const shared_ptr<const MeshD<2>>& dst_mesh)
        : solver(solver), dest_mesh(dst_mesh) {
        // Create horizontal points lists
        if (solver->mesh) {
            setupFromAxis(solver->mesh);
        } else if (auto rect_mesh = dynamic_pointer_cast<const RectangularMesh<2>>(dst_mesh)) {
            setupFromAxis(rect_mesh->axis[0]);
        } else {
            regpoints.reserve(solver->regions.size());
            InterpolationFlags flags(solver->geometry);
            for (size_t r = 0; r != solver->regions.size(); ++r) {
                std::set<double> pts;
                for (auto point : *dest_mesh) {
                    auto p = flags.wrap(point);
                    if (solver->regions[r].contains(p)) pts.insert(p.c0);
                }
                auto msh = plask::make_shared<OrderedAxis>();
                msh->addOrderedPoints(pts.begin(), pts.end(), pts.size());
                regpoints.emplace_back(std::move(msh));
            }
        }
    }

    void compute(double wavelength, InterpolationMethod interp) {
        // Compute gains on mesh for each active region
        data.resize(solver->regions.size());
        for (size_t reg = 0; reg != solver->regions.size(); ++reg) {
            if (regpoints[reg]->size() == 0) {
                data[reg] = LazyData<double>(dest_mesh->size(), 0.);
                continue;
            }
            DataVector<double> values(regpoints[reg]->size());
            AveragedData temps(solver, "temperature", regpoints[reg], solver->regions[reg]);
            AveragedData concs(temps);
            concs.name = "carriers concentration";
            temps.data = solver->inTemperature(temps.mesh, interp);
            concs.data = solver->inCarriersConcentration(temps.mesh, interp);
            if (solver->build_struct_once && !solver->region_levels[reg])
                solver->findEnergyLevels(solver->region_levels[reg], solver->regions[reg], solver->Tref);
            std::exception_ptr error;
#pragma omp parallel for
            for (int i = 0; i < regpoints[reg]->size(); ++i) {
                if (error) continue;
                try {
                    if (solver->build_struct_once) {
                        values[i] = getValue(wavelength, temps[i], max(concs[i], 1e-9), solver->regions[reg],
                                             solver->region_levels[reg]);
                    } else {
                        Levels levels;
                        solver->findEnergyLevels(levels, solver->regions[reg], temps[i]);
                        values[i] = getValue(wavelength, temps[i], max(concs[i], 1e-9), solver->regions[reg], levels);
                    }
                } catch (...) {
#pragma omp critical
                    error = std::current_exception();
                }
            }
            if (error) std::rethrow_exception(error);
            data[reg] = interpolate(plask::make_shared<RectangularMesh<2>>(regpoints[reg], zero_axis), values,
                                    dest_mesh, interp);
        }
    }

    virtual double getValue(double wavelength,
                            double temp,
                            double conc,
                            const typename FermiNewGainSolver<GeometryT>::ActiveRegionInfo& region,
                            const Levels& levels) = 0;

    size_t size() const override { return dest_mesh->size(); }

    T at(size_t i) const override {
        for (size_t reg = 0; reg != solver->regions.size(); ++reg)
            //             if (solver->regions[reg].contains(dest_mesh->at(i)))
            if (solver->regions[reg].inQW(dest_mesh->at(i))) return data[reg][i];
        return 0.;
    }
};

template <typename GeometryT> struct GainData : public DataBase<GeometryT, Tensor2<double>> {
    template <typename... Args> GainData(Args... args) : DataBase<GeometryT, Tensor2<double>>(args...) {}

    double getValue(double wavelength,
                    double temp,
                    double conc,
                    const typename FermiNewGainSolver<GeometryT>::ActiveRegionInfo& region,
                    const Levels& levels) override {
        kubly::wzmocnienie gainModule = this->solver->getGainModule(wavelength, temp, conc, region, levels);

        if (!this->solver->lifetime || region.mod)
            return gainModule.wzmocnienie_calk_bez_splotu(nm_to_eV(wavelength));
        else
            return gainModule.wzmocnienie_calk_ze_splotem(nm_to_eV(wavelength),
                                                          phys::hb_eV * 1e12 / this->solver->lifetime);
    }
};

template <typename GeometryT> struct DgDnData : public DataBase<GeometryT, Tensor2<double>> {
    template <typename... Args> DgDnData(Args... args) : DataBase<GeometryT, Tensor2<double>>(args...) {}

    double getValue(double wavelength,
                    double temp,
                    double conc,
                    const typename FermiNewGainSolver<GeometryT>::ActiveRegionInfo& region,
                    const Levels& levels) override {
        double h = 0.5 * this->solver->differenceQuotient;
        double conc1 = (1. - h) * conc, conc2 = (1. + h) * conc;

        kubly::wzmocnienie gainModule1 = this->solver->getGainModule(wavelength, temp, conc1, region, levels);
        kubly::wzmocnienie gainModule2 = this->solver->getGainModule(wavelength, temp, conc2, region, levels);

        double gain1, gain2;
        if (!this->solver->lifetime || region.mod) {
            gain1 = gainModule1.wzmocnienie_calk_bez_splotu(nm_to_eV(wavelength));
            gain2 = gainModule2.wzmocnienie_calk_bez_splotu(nm_to_eV(wavelength));
        } else {
            gain1 = gainModule1.wzmocnienie_calk_ze_splotem(nm_to_eV(wavelength),
                                                            phys::hb_eV * 1e12 / this->solver->lifetime);
            gain2 = gainModule2.wzmocnienie_calk_ze_splotem(nm_to_eV(wavelength),
                                                            phys::hb_eV * 1e12 / this->solver->lifetime);
        }
        return (gain2 - gain1) / (2. * h * conc);
    }
};

inline static double sumLuminescence(kubly::wzmocnienie& gain, double wavelength) {
    double E = nm_to_eV(wavelength);
    //double result = 0.;
    //for (int nr_c = 0; nr_c <= (int)gain.pasma->pasmo_przew.size() - 1; nr_c++)
    //    for (int nr_v = 0; nr_v <= (int)gain.pasma->pasmo_wal.size() - 1; nr_v++)
    //        result += gain.spont_od_pary_pasm(E, nr_c, nr_v, 0);  // TODO: consider other polarization (now only TE)
    //return result;
    return gain.lumin(E, 0.); // TODO: consider not only 0.<->TE,   TODO: add 1.<->TM, 2.<->TE+TM
}

template <typename GeometryT> struct LuminescenceData : public DataBase<GeometryT, double> {
    template <typename... Args> LuminescenceData(Args... args) : DataBase<GeometryT, double>(args...) {}

    double getValue(double wavelength,
                    double temp,
                    double conc,
                    const typename FermiNewGainSolver<GeometryT>::ActiveRegionInfo& region,
                    const Levels& levels) override {
        kubly::wzmocnienie gainModule = this->solver->getGainModule(wavelength, temp, conc, region, levels);

        double QWfrac = region.qwtotallen / region.totallen;
        return sumLuminescence(gainModule, wavelength) / QWfrac;
    }
};

template <typename GeometryType>
const LazyData<Tensor2<double>> FermiNewGainSolver<GeometryType>::getGain(Gain::EnumType what,
                                                                          const shared_ptr<const MeshD<2>>& dst_mesh,
                                                                          double wavelength,
                                                                          InterpolationMethod interp) {
    if (what == Gain::DGDN) {
        this->writelog(LOG_DETAIL, "Calculating gain over carriers concentration derivative");
        this->initCalculation();  // This must be called before any calculation!
        DgDnData<GeometryType>* data = new DgDnData<GeometryType>(this, dst_mesh);
        data->compute(wavelength, getInterpolationMethod<INTERPOLATION_SPLINE>(interp));
        return LazyData<Tensor2<double>>(data);
    } else {
        this->writelog(LOG_DETAIL, "Calculating gain");
        this->initCalculation();  // This must be called before any calculation!
        GainData<GeometryType>* data = new GainData<GeometryType>(this, dst_mesh);
        data->compute(wavelength, getInterpolationMethod<INTERPOLATION_SPLINE>(interp));
        return LazyData<Tensor2<double>>(data);
    }
}

template <typename GeometryType>
const LazyData<double> FermiNewGainSolver<GeometryType>::getLuminescence(const shared_ptr<const MeshD<2>>& dst_mesh,
                                                                         double wavelength,
                                                                         InterpolationMethod interp) {
    this->writelog(LOG_DETAIL, "Calculating luminescence");
    this->initCalculation();  // This must be called before any calculation!

    LuminescenceData<GeometryType>* data = new LuminescenceData<GeometryType>(this, dst_mesh);
    data->compute(wavelength, getInterpolationMethod<INTERPOLATION_SPLINE>(interp));

    return LazyData<double>(data);
}

template <typename GeometryT>
GainSpectrum<GeometryT>::GainSpectrum(FermiNewGainSolver<GeometryT>* solver, const Vec<2> point)
    : solver(solver), point(point) {
    auto mesh = plask::make_shared<const OnePointMesh<2>>(point);
    T = solver->inTemperature(mesh)[0];
    n = solver->inCarriersConcentration(mesh)[0];
    for (reg = 0; reg < solver->regions.size(); ++reg) {
        if (solver->regions[reg].contains(point)) {
            solver->inTemperature.changedConnectMethod(this, &GainSpectrum::onTChange);
            solver->inCarriersConcentration.changedConnectMethod(this, &GainSpectrum::onNChange);
            return;
        };
    }
    throw BadInput(solver->getId(), "Point {0} does not belong to any active region", point);
}

template <typename GeometryT> double GainSpectrum<GeometryT>::getGain(double wavelength) {
    if (!gMod) {
        Levels* levels;
        if (solver->build_struct_once && !solver->region_levels[reg]) {
            solver->findEnergyLevels(solver->region_levels[reg], solver->regions[reg], solver->Tref);
            levels = &solver->region_levels[reg];
        } else {
            this->levels.reset(new Levels);
            levels = this->levels.get();
            solver->findEnergyLevels(*levels, solver->regions[reg], T, true);
        }
        gMod.reset(new kubly::wzmocnienie(std::move(solver->getGainModule(wavelength, T, n, solver->regions[reg], *levels, true))));
    }

    double E = nm_to_eV(wavelength);
    double tau = solver->getLifeTime();
    if (!tau || solver->regions[reg].mod)
        return (gMod->wzmocnienie_calk_bez_splotu(E));  // 20.10.2014 adding lifetime
    else
        return (gMod->wzmocnienie_calk_ze_splotem(E, phys::hb_eV * 1e12 / tau));  // 20.10.2014 adding lifetime
}

template <typename GeometryT>
LuminescenceSpectrum<GeometryT>::LuminescenceSpectrum(FermiNewGainSolver<GeometryT>* solver, const Vec<2> point)
    : solver(solver), point(point) {
    auto mesh = plask::make_shared<const OnePointMesh<2>>(point);
    T = solver->inTemperature(mesh)[0];
    n = solver->inCarriersConcentration(mesh)[0];
    for (reg = 0; reg < solver->regions.size(); ++reg) {
        if (solver->regions[reg].contains(point)) {
            solver->inTemperature.changedConnectMethod(this, &LuminescenceSpectrum::onTChange);
            solver->inCarriersConcentration.changedConnectMethod(this, &LuminescenceSpectrum::onNChange);
            return;
        };
    }
    throw BadInput(solver->getId(), "Point {0} does not belong to any active region", point);
}

template <typename GeometryT> double LuminescenceSpectrum<GeometryT>::getLuminescence(double wavelength) {
    if (!gMod) {
        Levels* levels;
        if (solver->build_struct_once && !solver->region_levels[reg]) {
            solver->findEnergyLevels(solver->region_levels[reg], solver->regions[reg], solver->Tref);
            levels = &solver->region_levels[reg];
        } else {
            this->levels.reset(new Levels);
            levels = this->levels.get();
            solver->findEnergyLevels(*levels, solver->regions[reg], T, true);
        }
        gMod.reset(new kubly::wzmocnienie(std::move(solver->getGainModule(wavelength, T, n, solver->regions[reg], *levels, true))));
    }
    double QWfrac = solver->regions[reg].qwtotallen / solver->regions[reg].totallen;
    return sumLuminescence(*gMod, wavelength) / QWfrac;
}

template <typename GeometryType>
GainSpectrum<GeometryType> FermiNewGainSolver<GeometryType>::getGainSpectrum(const Vec<2>& point) {
    this->initCalculation();
    return GainSpectrum<GeometryType>(this, point);
}

template <typename GeometryType>
LuminescenceSpectrum<GeometryType> FermiNewGainSolver<GeometryType>::getLuminescenceSpectrum(const Vec<2>& point) {
    this->initCalculation();
    return LuminescenceSpectrum<GeometryType>(this, point);
}

template <> std::string FermiNewGainSolver<Geometry2DCartesian>::getClassName() const { return "gain.FermiNew2D"; }
template <> std::string FermiNewGainSolver<Geometry2DCylindrical>::getClassName() const { return "gain.FermiNewCyl"; }

template struct PLASK_SOLVER_API FermiNewGainSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API FermiNewGainSolver<Geometry2DCylindrical>;

template struct PLASK_SOLVER_API GainSpectrum<Geometry2DCartesian>;
template struct PLASK_SOLVER_API GainSpectrum<Geometry2DCylindrical>;

template struct PLASK_SOLVER_API LuminescenceSpectrum<Geometry2DCartesian>;
template struct PLASK_SOLVER_API LuminescenceSpectrum<Geometry2DCylindrical>;

}}}  // namespace plask::solvers::FermiNew
