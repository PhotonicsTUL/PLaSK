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
#include "solver3d.hpp"
#include "expansion3d.hpp"
#include "../diagonalizer.hpp"

namespace plask { namespace optical { namespace modal {

FourierSolver3D::FourierSolver3D(const std::string& name): ModalSolver<SolverOver<Geometry3D>>(name),
    size_long(12), size_tran(12),
    klong(0.), ktran(0.),
    symmetry_long(Expansion::E_UNSPECIFIED), symmetry_tran(Expansion::E_UNSPECIFIED),
    dct(2),
    expansion_rule(RULE_DIRECT),
    expansion(this),
    refine_long(16), refine_tran(16),
    grad_smooth(1e-3),
    outGradients(this, &FourierSolver3D::getGradients)
{
    pml_tran.factor = {1., -2.};
    pml_long.factor = {1., -2.};
}

static inline void updatePML(PML& pml, XMLReader& reader) {
    pml.factor = reader.getAttribute<dcomplex>("factor", pml.factor);
    pml.size = reader.getAttribute<double>("size", pml.size);
    pml.dist = reader.getAttribute<double>("dist", pml.dist);
    pml.order = reader.getAttribute<double>("shape", pml.order);
}

template <typename T>
static inline void readComaAttr(XMLReader& reader, const std::string& attr, T& long_field, T& tran_field, bool nocommon=false) {
    if (reader.hasAttribute(attr)) {
        std::string value = reader.requireAttribute<std::string>(attr);
        if (value.find(',') == std::string::npos) {
            if (nocommon) throw XMLBadAttrException(reader, attr, value);
            long_field = tran_field = boost::lexical_cast<T>(value);
        } else {
            auto values = splitString2(value, ',');
            long_field = boost::lexical_cast<T>(values.first);
            tran_field = boost::lexical_cast<T>(values.second);
        }
        if (reader.hasAttribute(attr+"-long")) throw XMLConflictingAttributesException(reader, attr, attr+"-long");
        if (reader.hasAttribute(attr+"-tran")) throw XMLConflictingAttributesException(reader, attr, attr+"-tran");
    } else {
        long_field = reader.getAttribute<T>(attr+"-long", long_field);
        tran_field = reader.getAttribute<T>(attr+"-tran", tran_field);
    }
}

static inline Expansion::Component readSymmetry(const FourierSolver3D* solver, const XMLReader& reader, const std::string& repr) {
    AxisNames* axes = nullptr;
    if (solver->getGeometry()) axes = &solver->getGeometry()->axisNames;
    if (repr == "none" || repr == "NONE" || repr == "None")
        return Expansion::E_UNSPECIFIED;
    else if (repr == "Etran" || repr == "Et" || (axes && repr == "E"+axes->getNameForTran()) ||
                repr == "Hlong" || repr == "Hl" || (axes && repr == "H"+axes->getNameForLong()))
        return Expansion::E_TRAN;
    else if (repr == "Elong" || repr == "El" || (axes && repr == "E"+axes->getNameForLong()) ||
                repr == "Htran" || repr == "Ht" || (axes && repr == "H"+axes->getNameForTran()))
        return Expansion::E_LONG;
    else
        throw XMLBadAttrException(reader, "symmetry", repr, "symmetric field component name (maybe you need to specify the geometry first)");
}

void FourierSolver3D::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "expansion") {
            readComaAttr(reader, "size", size_long, size_tran);
            readComaAttr(reader, "refine", refine_long, refine_tran);
            if (reader.hasAttribute("oversampling")) {
                reader.ignoreAttribute("oversampling");
                writelog(LOG_WARNING, "obsolete 'oversampling' attribute in XML line {}", reader.getLineNr());
            }
            if (reader.hasAttribute("oversampling-long")) {
                reader.ignoreAttribute("oversampling-long");
                writelog(LOG_WARNING, "obsolete 'oversampling-long' attribute in XML line {}", reader.getLineNr());
            }
            if (reader.hasAttribute("oversampling-tran")) {
                reader.ignoreAttribute("oversampling-tran");
                writelog(LOG_WARNING, "obsolete 'oversampling-tran' attribute in XML line {}", reader.getLineNr());
            }
            smooth = reader.getAttribute<double>("smooth", smooth);
            int dc = reader.getAttribute<int>("dct", dct);
            if (dc != 1 && dc != 2)
                throw XMLBadAttrException(reader, "dct", boost::lexical_cast<std::string>(dc), "\"1\" or \"2\"");
            dct = dc;
            group_layers = reader.getAttribute<bool>("group-layers", group_layers);
            lam0 = reader.getAttribute<double>("lam0", NAN);
            always_recompute_gain = reader.getAttribute<bool>("update-gain", always_recompute_gain);
            max_temp_diff = reader.getAttribute<double>("temp-diff", max_temp_diff);
            temp_dist = reader.getAttribute<double>("temp-dist", temp_dist);
            temp_layer = reader.getAttribute<double>("temp-layer", temp_layer);
            expansion_rule = reader.enumAttribute<ExpansionRule>("rule")
                                .value("direct", RULE_DIRECT)
                                .value("inverse", RULE_INVERSE)
                                .value("combined", RULE_COMBINED)
                                .value("old", RULE_OLD)
                             .get(expansion_rule);
            grad_smooth = reader.getAttribute<double>("grad-smooth", grad_smooth);
            reader.requireTagEnd();
        } else if (param == "pmls") {
            updatePML(pml_long, reader);
            updatePML(pml_tran, reader);
            while (reader.requireTagOrEnd()) {
                std::string node = reader.getNodeName();
                if (node == "long") {
                    updatePML(pml_long, reader);
                    reader.requireTagEnd();
                } else if (node == "tran") {
                    updatePML(pml_tran, reader);
                    reader.requireTagEnd();
                } else throw XMLUnexpectedElementException(reader, "<tran>, <long>, or </pmls>", node);
            }
        } else if (param == "mode") {
            emission = reader.enumAttribute<Emission>("emission")
                            .value("undefined", EMISSION_UNSPECIFIED)
                            .value("top", EMISSION_TOP)
                            .value("bottom", EMISSION_BOTTOM)
                       .get(emission);
            if (reader.hasAttribute("wavelength")) { //TODO Remove in the future
                writelog(LOG_WARNING, "XML line {:d} in <mode>: Attribute 'wavelength' is obsolete, use 'lam' instead", reader.getLineNr());
                if (reader.hasAttribute("lam")) throw XMLConflictingAttributesException(reader, "wavelength", "lam");
                k0 = 2e3*PI / reader.requireAttribute<dcomplex>("wavelength");
            }
            if (reader.hasAttribute("lam")) k0 = 2e3*PI / reader.requireAttribute<dcomplex>("lam");
            ktran = reader.getAttribute<dcomplex>("k-tran", ktran);
            klong = reader.getAttribute<dcomplex>("k-long", klong);
            std::string sym_tran, sym_long;
            readComaAttr(reader, "symmetry", sym_long, sym_tran, true);
            if (sym_long != "") setSymmetryLong(readSymmetry(this, reader, sym_long));
            if (sym_tran != "") setSymmetryTran(readSymmetry(this, reader, sym_tran));
            reader.requireTagEnd();
        } else
            parseCommonModalConfiguration(reader, manager);
    }
}


void FourierSolver3D::onInitialize()
{
    this->setupLayers();
    if (this->interface == -1)
        Solver::writelog(LOG_DETAIL, "Initializing Fourier3D solver ({0} layers in the stack)",
                                     this->stack.size());
    else
        Solver::writelog(LOG_DETAIL, "Initializing Fourier3D solver ({0} layers in the stack, interface after {1} layer{2})",
                                     this->stack.size(), this->interface, (this->interface==1)? "" : "s");
    setExpansionDefaults();
    expansion.init();
    this->recompute_integrals = true;
}


void FourierSolver3D::onInvalidate()
{
    modes.clear();
    expansion.reset();
    transfer.reset();
}


size_t FourierSolver3D::findMode(FourierSolver3D::What what, dcomplex start)
{
    expansion.setSymmetryLong(symmetry_long);
    expansion.setSymmetryTran(symmetry_tran);
    expansion.setLam0(this->lam0);
    Solver::initCalculation();
    ensureInterface();
    if (!transfer) initTransfer(expansion, false);
    std::unique_ptr<RootDigger> root;
    switch (what) {
        case FourierSolver3D::WHAT_WAVELENGTH:
            expansion.setKlong(klong);
            expansion.setKtran(ktran);
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'lam' converged to NaN");
                expansion.setK0(2e3*PI/x); return transfer->determinant();
            }, "lam");
            break;
        case FourierSolver3D::WHAT_K0:
            expansion.setKlong(klong);
            expansion.setKtran(ktran);
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'k0' converged to NaN");
                expansion.setK0(x); return transfer->determinant();
            }, "k0");
            break;
        case FourierSolver3D::WHAT_KLONG:
            if (expansion.symmetric_long())
                throw Exception("{}: Cannot search for longitudinal wavevector with longitudinal symmetry", getId());
            expansion.setK0(this->k0);
            expansion.setKtran(ktran);
            transfer->fields_determined = Transfer::DETERMINED_NOTHING;
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'klong' converged to NaN");
                expansion.klong = x; return transfer->determinant();
            }, "klong");
            break;
        case FourierSolver3D::WHAT_KTRAN:
            if (expansion.symmetric_tran())
                throw Exception("{}: Cannot search for transverse wavevector with transverse symmetry", getId());
            expansion.setK0(this->k0);
            expansion.setKlong(klong);
            transfer->fields_determined = Transfer::DETERMINED_NOTHING;
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'ktran' converged to NaN");
                expansion.ktran = x; return transfer->determinant();
            }, "ktran");
            break;
    }
    root->find(start);
    return insertMode();
}


double FourierSolver3D::getWavelength(size_t n) {
    if (n >= modes.size()) throw NoValue(ModeWavelength::NAME);
    return real(2e3*M_PI / modes[n].k0);
}


size_t FourierSolver3D::initIncidence(Transfer::IncidentDirection side, Expansion::Component polarization, dcomplex lam)
{

    if (polarization == ExpansionPW3D::E_UNSPECIFIED)
        throw BadInput(getId(), "unspecified incident polarization for reflectivity computation");
    if (expansion.symmetry_long == Expansion::Component(3-polarization))
        throw BadInput(getId(), "current longitudinal symmetry is inconsistent with the specified incident polarization");
    if (expansion.symmetry_tran == Expansion::Component(3-polarization))
        throw BadInput(getId(), "current transverse symmetry is inconsistent with the specified incident polarization");
    return ModalSolver<SolverOver<Geometry3D>>::initIncidence(side, lam);
}

cvector FourierSolver3D::incidentVector(Transfer::IncidentDirection side, Expansion::Component polarization, dcomplex lam)
{
    size_t layer = initIncidence(side, polarization, lam);

    size_t idx = (polarization == ExpansionPW3D::E_LONG)? expansion.iEx(0,0) : expansion.iEy(0,0);
    cvector physical(expansion.matrixSize(), 0.);
    physical[idx] = 1.;

    cvector incident = transfer->diagonalizer->invTE(layer) * physical;
    scaleIncidentVector(incident, layer);
    return incident;
}


cvector FourierSolver3D::incidentGaussian(Transfer::IncidentDirection side, Expansion::Component polarization,
                                          double sigma_long, double sigma_tran, double center_long, double center_tran,
                                          dcomplex lam)
{
    size_t layer = initIncidence(side, polarization, lam);

    double bl = 2.*PI / (expansion.front-expansion.back) * (expansion.symmetric_long()? 0.5 : 1.0),
           bt = 2.*PI / (expansion.right-expansion.left) * (expansion.symmetric_tran()? 0.5 : 1.0);
    dcomplex dl = I * bl * (center_long - expansion.back), dt = I * bt * (center_tran - expansion.left);
    double cl2 = - 0.5 * sigma_long*sigma_long * bl*bl, ct2 = - 0.5 * sigma_tran*sigma_tran * bt*bt;

    cvector physical(expansion.matrixSize(), 0.);
    for (int it = -int(size_tran); it <= int(size_tran); ++it) {
        dcomplex vt = exp(ct2 * double(it*it) - dt*double(it));
        for (int il = -int(size_long); il <= int(size_long); ++il) {
            size_t idx = (polarization == Expansion::E_LONG)? expansion.iEx(il, it) : expansion.iEy(il, it);
            physical[idx] = vt * exp(cl2 * double(il*il) - dl*double(il));
        }
    }

    cvector incident = transfer->diagonalizer->invTE(layer) * physical;
    scaleIncidentVector(incident, layer);
    return incident;
}

LazyData<double> FourierSolver3D::getGradients(GradientFunctions::EnumType what,
                                               const shared_ptr<const MeshD<3>>& dst_mesh,
                                               InterpolationMethod interp) {
    initCalculation();
    computeIntegrals();
    DataVector<double> destination(dst_mesh->size());
    auto levels = makeLevelsAdapter(dst_mesh);
    while (auto level = levels->yield()) {
        auto dest = expansion.getGradients(what, level, interp);
        for (size_t i = 0; i != level->size(); ++i) destination[level->index(i)] = dest[i];
    }
    return destination;
}

}}} // namespace
