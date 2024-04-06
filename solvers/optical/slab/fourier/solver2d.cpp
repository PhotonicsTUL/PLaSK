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
#include "solver2d.hpp"
#include "expansion2d.hpp"
#include "../diagonalizer.hpp"

namespace plask { namespace optical { namespace slab {

FourierSolver2D::FourierSolver2D(const std::string& name):
    SlabSolver<SolverWithMesh<Geometry2DCartesian,MeshAxis>>(name),
    beta(0.), ktran(0.),
    symmetry(Expansion::E_UNSPECIFIED),
    polarization(Expansion::E_UNSPECIFIED),
    size(12),
    dct(2),
    ftt(FOURIER_DISCRETE),
    expansion(this),
    refine(32),
    outNeff(this, &FourierSolver2D::getEffectiveIndex, &FourierSolver2D::nummodes)
{}


void FourierSolver2D::loadConfiguration(XMLReader& reader, Manager& manager)
{
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "expansion") {
            size = reader.getAttribute<size_t>("size", size);
            refine = reader.getAttribute<size_t>("refine", refine);
            smooth = reader.getAttribute<double>("smooth", smooth);
            if (reader.hasAttribute("oversampling")) {
                reader.ignoreAttribute("oversampling");
                writelog(LOG_WARNING, "obsolete 'oversampling' attribute in XML line {}", reader.getLineNr());
            }
            ftt = reader.enumAttribute<FourierType>("ft")
                .value("discrete", FOURIER_DISCRETE)
                .value("analytic", FOURIER_ANALYTIC)
                .get(ftt);
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
            reader.requireTagEnd();
        } else if (param == "pml") {
            pml.factor = reader.getAttribute<dcomplex>("factor", pml.factor);
            pml.size = reader.getAttribute<double>("size", pml.size);
            pml.dist = reader.getAttribute<double>("dist", pml.dist);
            if (reader.hasAttribute("order")) { //TODO Remove in the future
                writelog(LOG_WARNING, "XML line {:d} in <pml>: Attribute 'order' is obsolete, use 'shape' instead", reader.getLineNr());
                if (reader.hasAttribute("shape")) throw XMLConflictingAttributesException(reader, "order", "shape");
                pml.order = reader.requireAttribute<double>("order");
            }
            pml.order = reader.getAttribute<double>("shape", pml.order);
            reader.requireTagEnd();
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
            if (reader.hasAttribute("beta")) {
                if (reader.hasAttribute("k-long")) throw XMLConflictingAttributesException(reader, "beta", "k-long");
                beta = reader.requireAttribute<dcomplex>("beta");
            } else
                beta = reader.getAttribute<dcomplex>("k-long", beta);
            if (reader.hasAttribute("symmetry")) {
                std::string repr = reader.requireAttribute("symmetry");
                Expansion::Component val;
                AxisNames* axes = nullptr;
                if (geometry) axes = &geometry->axisNames;
                if (repr == "none" || repr == "NONE" || repr == "None")
                    val = Expansion::E_UNSPECIFIED;
                else if (repr == "Etran" || repr == "Et" || (axes && repr == "E"+axes->getNameForTran()) ||
                         repr == "Hlong" || repr == "Hl" || (axes && repr == "H"+axes->getNameForLong()))
                    val = Expansion::E_TRAN;
                else if (repr == "Elong" || repr == "El" || (axes && repr == "E"+axes->getNameForLong()) ||
                         repr == "Htran" || repr == "Ht" || (axes && repr == "H"+axes->getNameForTran()))
                    val = Expansion::E_LONG;
                else
                    throw XMLBadAttrException(reader, "symmetry", repr, "symmetric field component name (maybe you need to specify the geometry first)");
                setSymmetry(val);
            }
            if (reader.hasAttribute("polarization")) {
                std::string repr = reader.requireAttribute("polarization");
                Expansion::Component val;
                AxisNames* axes = nullptr;
                if (geometry) axes = &geometry->axisNames;
                if (repr == "none" || repr == "NONE" || repr == "None")
                    val = Expansion::E_UNSPECIFIED;
                else if (repr == "Etran" || repr == "Et" || (axes && repr == "E"+axes->getNameForTran()) ||
                         repr == "Hlong" || repr == "Hl" || (axes && repr == "H"+axes->getNameForLong()) || repr == "TM")
                    val = Expansion::E_TRAN;
                else if (repr == "Elong" || repr == "El" || (axes && repr == "E"+axes->getNameForLong()) ||
                         repr == "Htran" || repr == "Ht" || (axes && repr == "H"+axes->getNameForTran()) || repr == "TE")
                    val = Expansion::E_LONG;
                else
                    throw XMLBadAttrException(reader, "polarization", repr, "existing field component name (maybe you need to specify the geometry first)");
                setPolarization(val);
            }
            reader.requireTagEnd();
        } else if (param == "mirrors") {
            double R1 = reader.requireAttribute<double>("R1");
            double R2 = reader.requireAttribute<double>("R2");
            mirrors.reset(std::make_pair(R1,R2));
            reader.requireTagEnd();
        } else
            parseCommonSlabConfiguration(reader, manager);
    }
}


void FourierSolver2D::onInitialize()
{
    this->setupLayers();
    if (this->interface == -1)
        Solver::writelog(LOG_DETAIL, "Initializing Fourier2D solver ({0} layers in the stack)",
                                     this->stack.size());
    else
        Solver::writelog(LOG_DETAIL, "Initializing Fourier2D solver ({0} layers in the stack, interface after {1} layer{2})",
                                     this->stack.size(), this->interface, (this->interface==1)? "" : "s");
    setExpansionDefaults();
    expansion.init();
    this->recompute_integrals = true;
}


void FourierSolver2D::onInvalidate()
{
    modes.clear();
    expansion.reset();
    transfer.reset();
}


size_t FourierSolver2D::findMode(FourierSolver2D::What what, dcomplex start)
{
    expansion.setSymmetry(symmetry);
    expansion.setPolarization(polarization);
    expansion.setLam0(this->lam0);
    Solver::initCalculation();
    ensureInterface();
    if (!transfer) initTransfer(expansion, false);
    std::unique_ptr<RootDigger> root;
    switch (what) {
        case FourierSolver2D::WHAT_WAVELENGTH:
            expansion.setBeta(beta);
            expansion.setKtran(ktran);
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'lam' converged to NaN");
                expansion.setK0(2e3*PI/x); return transfer->determinant();
            }, "lam");
            break;
        case FourierSolver2D::WHAT_K0:
            expansion.setBeta(beta);
            expansion.setKtran(ktran);
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'k0' converged to NaN");
                expansion.setK0(x); return transfer->determinant();
            }, "k0");
            break;
        case FourierSolver2D::WHAT_NEFF:
            if (expansion.separated())
                throw Exception("{0}: Cannot search for effective index with polarization separation", getId());
            expansion.setK0(k0);
            expansion.setKtran(ktran);
            clearFields();
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'neff' converged to NaN");
                expansion.setBeta(x * expansion.k0); return transfer->determinant();
            }, "neff");
            break;
        case FourierSolver2D::WHAT_KTRAN:
            if (expansion.symmetric())
                throw Exception("{0}: Cannot search for transverse wavevector with symmetry", getId());
            expansion.setK0(k0);
            expansion.setBeta(beta);
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'ktran' converged to NaN");
                expansion.setKtran(x); return transfer->determinant();
            }, "ktran");
            break;
        case FourierSolver2D::WHAT_BETA:
            if (expansion.separated())
                throw Exception("{0}: Cannot search for longitudinal wavevector with polarization separation", getId());
            expansion.setK0(k0);
            expansion.setKtran(ktran);
            clearFields();
            root = getRootDigger([this](const dcomplex& x) {
                if (isnan(x)) throw ComputationError(this->getId(), "'beta' converged to NaN");
                expansion.setBeta(x); return transfer->determinant();
            }, "beta");
            break;
    }
    root->find(start);
    return insertMode();
}


double FourierSolver2D::getWavelength(size_t n) {
    if (n >= modes.size()) throw NoValue(ModeWavelength::NAME);
    return real(2e3*M_PI / modes[n].k0);
}


size_t FourierSolver2D::initIncidence(Transfer::IncidentDirection side, Expansion::Component polarization, dcomplex lam) {
    if (!isinf(geometry->getExtrusion()->getLength()))
        throw Exception("{}: Reflectivity computation for 2D geometries possible only if the extrusion length is infinite",
                        getId());
    if (polarization == Expansion::E_UNSPECIFIED)
        throw BadInput(getId(), "unspecified incident polarization for reflectivity computation");
    if (expansion.symmetric() && expansion.symmetry != polarization)
        throw BadInput(getId(), "current solver symmetry is inconsistent with the specified incident polarization");
    if (expansion.separated() && expansion.polarization != polarization)
        throw BadInput(getId(), "current solver polarization is inconsistent with the specified incident polarization");
    return SlabSolver<SolverWithMesh<Geometry2DCartesian,MeshAxis>>::initIncidence(side, lam);
}

cvector FourierSolver2D::incidentVector(Transfer::IncidentDirection side, Expansion::Component polarization, dcomplex lam)
{
    size_t layer = initIncidence(side, polarization, lam);

    size_t idx;
    if (expansion.separated()) idx = expansion.iEH(0);
    else idx = (polarization == Expansion::E_TRAN)? expansion.iEx(0) : expansion.iEz(0);
    cvector physical(expansion.matrixSize(), 0.);
    physical[idx] = (polarization == Expansion::E_TRAN)? 1. : -1.;

    cvector incident = transfer->diagonalizer->invTE(layer) * physical;
    scaleIncidentVector(incident, layer);
    return incident;
}


cvector FourierSolver2D::incidentGaussian(Transfer::IncidentDirection side, Expansion::Component polarization, double sigma, double center, dcomplex lam)
{
    size_t layer = initIncidence(side, polarization, lam);

    double b = 2.*PI / (expansion.right-expansion.left) * (expansion.symmetric()? 0.5 : 1.0);
    dcomplex d = I * b * (center - expansion.left);
    double c2 = - 0.5 * sigma*sigma * b*b;

    cvector physical(expansion.matrixSize(), 0.);
    for (int i = -int(size); i <= int(size); ++i) {
        size_t idx;
        if (expansion.separated()) idx = expansion.iEH(i);
        else idx = (polarization == Expansion::E_TRAN)? expansion.iEx(i) : expansion.iEz(i);
        dcomplex val = exp(c2 * double(i*i) - d*double(i));
        physical[idx] = (polarization == Expansion::E_TRAN)? val : -val;
    }

    cvector incident = transfer->diagonalizer->invTE(layer) * physical;
    scaleIncidentVector(incident, layer);
    return incident;
}


}}} // namespace
