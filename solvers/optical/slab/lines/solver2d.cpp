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
#include "../diagonalizer.hpp"
#include "expansion2d.hpp"

namespace plask { namespace optical { namespace slab {

LinesSolver2D::LinesSolver2D(const std::string& name)
    : SlabSolver<SolverWithMesh<Geometry2DCartesian, RegularAxis>>(name),
      beta(0.),
      ktran(0.),
      symmetry(Expansion::E_UNSPECIFIED),
      polarization(Expansion::E_UNSPECIFIED),
      expansion(this),
      density(NAN),
      refine(32),
      outNeff(this, &LinesSolver2D::getEffectiveIndex, &LinesSolver2D::nummodes) {
          pml.factor = {1., -2.};
      }

void LinesSolver2D::loadConfiguration(XMLReader& reader, Manager& manager) {
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "discretization") {
            density = reader.getAttribute<double>("density", density);
            refine = reader.getAttribute<size_t>("refine", refine);
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
            pml.order = reader.getAttribute<double>("shape", pml.order);
            reader.requireTagEnd();
        } else if (param == "mode") {
            emission = reader.enumAttribute<Emission>("emission")
                           .value("undefined", EMISSION_UNSPECIFIED)
                           .value("top", EMISSION_TOP)
                           .value("bottom", EMISSION_BOTTOM)
                           .get(emission);
            if (reader.hasAttribute("wavelength")) {  // TODO Remove in the future
                writelog(LOG_WARNING, "XML line {:d} in <mode>: Attribute 'wavelength' is obsolete, use 'lam' instead",
                         reader.getLineNr());
                if (reader.hasAttribute("lam")) throw XMLConflictingAttributesException(reader, "wavelength", "lam");
                k0 = 2e3 * PI / reader.requireAttribute<dcomplex>("wavelength");
            }
            if (reader.hasAttribute("lam")) k0 = 2e3 * PI / reader.requireAttribute<dcomplex>("lam");
            ktran = reader.getAttribute<dcomplex>("k-tran", ktran);
            beta = reader.getAttribute<dcomplex>("k-long", beta);
            if (reader.hasAttribute("symmetry")) {
                std::string repr = reader.requireAttribute("symmetry");
                Expansion::Component val;
                AxisNames* axes = nullptr;
                if (geometry) axes = &geometry->axisNames;
                if (repr == "none" || repr == "NONE" || repr == "None")
                    val = Expansion::E_UNSPECIFIED;
                else if (repr == "Etran" || repr == "Et" || (axes && repr == "E" + axes->getNameForTran()) || repr == "Hlong" ||
                         repr == "Hl" || (axes && repr == "H" + axes->getNameForLong()))
                    val = Expansion::E_TRAN;
                else if (repr == "Elong" || repr == "El" || (axes && repr == "E" + axes->getNameForLong()) || repr == "Htran" ||
                         repr == "Ht" || (axes && repr == "H" + axes->getNameForTran()))
                    val = Expansion::E_LONG;
                else
                    throw XMLBadAttrException(reader, "symmetry", repr,
                                              "symmetric field component name (maybe you need to specify the geometry first)");
                setSymmetry(val);
            }
            if (reader.hasAttribute("polarization")) {
                std::string repr = reader.requireAttribute("polarization");
                Expansion::Component val;
                AxisNames* axes = nullptr;
                if (geometry) axes = &geometry->axisNames;
                if (repr == "none" || repr == "NONE" || repr == "None")
                    val = Expansion::E_UNSPECIFIED;
                else if (repr == "TE" || repr == "Etran" || repr == "Et" || (axes && repr == "E" + axes->getNameForTran()) ||
                         repr == "Hlong" || repr == "Hl" || (axes && repr == "H" + axes->getNameForLong()))
                    val = Expansion::E_TRAN;
                else if (repr == "TM" || repr == "Elong" || repr == "El" || (axes && repr == "E" + axes->getNameForLong()) ||
                         repr == "Htran" || repr == "Ht" || (axes && repr == "H" + axes->getNameForTran()))
                    val = Expansion::E_LONG;
                else
                    throw XMLBadAttrException(reader, "polarization", repr,
                                              "existing field component name (maybe you need to specify the geometry first)");
                setPolarization(val);
            }
            reader.requireTagEnd();
        } else if (param == "mirrors") {
            double R1 = reader.requireAttribute<double>("R1");
            double R2 = reader.requireAttribute<double>("R2");
            mirrors.reset(std::make_pair(R1, R2));
            reader.requireTagEnd();
        } else
            parseCommonSlabConfiguration(reader, manager);
    }
}

void LinesSolver2D::onInitialize() {
    this->setupLayers();
    if (this->interface == -1)
        Solver::writelog(LOG_DETAIL, "Initializing Lines2D solver ({0} layers in the stack)", this->stack.size());
    else
        Solver::writelog(LOG_DETAIL, "Initializing Lines2D solver ({0} layers in the stack, interface after {1} layer{2})",
                         this->stack.size(), this->interface, (this->interface == 1) ? "" : "s");
    setExpansionDefaults();
    expansion.init();
    this->recompute_integrals = true;
}

void LinesSolver2D::onInvalidate() {
    modes.clear();
    expansion.reset();
    transfer.reset();
}

size_t LinesSolver2D::findMode(LinesSolver2D::What what, dcomplex start) {
    expansion.setSymmetry(symmetry);
    expansion.setPolarization(polarization);
    expansion.setLam0(this->lam0);
    Solver::initCalculation();
    ensureInterface();
    if (!transfer) initTransfer(expansion, false);
    std::unique_ptr<RootDigger> root;
    switch (what) {
        case LinesSolver2D::WHAT_WAVELENGTH:
            expansion.setBeta(beta);
            expansion.setKtran(ktran);
            root = getRootDigger(
                [this](const dcomplex& x) {
                    if (isnan(x)) throw ComputationError(this->getId(), "'lam' converged to NaN");
                    expansion.setK0(2e3 * PI / x);
                    return transfer->determinant();
                },
                "lam");
            break;
        case LinesSolver2D::WHAT_K0:
            expansion.setBeta(beta);
            expansion.setKtran(ktran);
            root = getRootDigger(
                [this](const dcomplex& x) {
                    if (isnan(x)) throw ComputationError(this->getId(), "'k0' converged to NaN");
                    expansion.setK0(x);
                    return transfer->determinant();
                },
                "k0");
            break;
        case LinesSolver2D::WHAT_NEFF:
            if (expansion.separated())
                throw Exception("{0}: Cannot search for effective index with polarization separation", getId());
            expansion.setK0(k0);
            expansion.setKtran(ktran);
            clearFields();
            root = getRootDigger(
                [this](const dcomplex& x) {
                    if (isnan(x)) throw ComputationError(this->getId(), "'neff' converged to NaN");
                    expansion.setBeta(x * expansion.k0);
                    return transfer->determinant();
                },
                "neff");
            break;
        case LinesSolver2D::WHAT_KTRAN:
            if (expansion.symmetric()) throw Exception("{0}: Cannot search for transverse wavevector with symmetry", getId());
            expansion.setK0(k0);
            expansion.setBeta(beta);
            root = getRootDigger(
                [this](const dcomplex& x) {
                    if (isnan(x)) throw ComputationError(this->getId(), "'ktran' converged to NaN");
                    expansion.setKtran(x);
                    return transfer->determinant();
                },
                "ktran");
            break;
    }
    root->find(start);
    return insertMode();
}

double LinesSolver2D::getWavelength(size_t n) {
    if (n >= modes.size()) throw NoValue(ModeWavelength::NAME);
    return real(2e3 * M_PI / modes[n].k0);
}

size_t LinesSolver2D::initIncidence(Transfer::IncidentDirection side, Expansion::Component polarization, dcomplex lam) {
    bool changed = Solver::initCalculation() || setExpansionDefaults(isnan(lam));
    if (!isnan(lam)) {
        dcomplex k0 = 2e3 * M_PI / lam;
        if (!is_zero(k0 - expansion.getK0())) {
            expansion.setK0(k0);
            changed = true;
        }
    }

    if (polarization == Expansion::E_UNSPECIFIED)
        throw BadInput(getId(), "Unspecified incident polarization for reflectivity computation");
    if (expansion.symmetric() && expansion.symmetry != polarization)
        throw BadInput(getId(), "Current solver symmetry is inconsistent with the specified incident polarization");
    if (expansion.separated() && expansion.polarization != polarization)
        throw BadInput(getId(), "Current solver polarization is inconsistent with the specified incident polarization");

    size_t layer = stack[(side == Transfer::INCIDENCE_BOTTOM) ? 0 : stack.size() - 1];
    if (!transfer) {
        initTransfer(expansion, true);
        changed = true;
    }
    if (changed) {
        transfer->initDiagonalization();
        transfer->diagonalizer->diagonalizeLayer(layer);
    } else if (!transfer->diagonalizer->isDiagonalized(layer))
        transfer->diagonalizer->diagonalizeLayer(layer);
    return layer;
}

cvector LinesSolver2D::incidentVector(Transfer::IncidentDirection side, Expansion::Component polarization, dcomplex lam) {
    size_t layer = initIncidence(side, polarization, lam);

    cvector incident(expansion.matrixSize());

    if (expansion.separated()) {
        std::fill(incident.begin(), incident.end(), 1.);
    } else {
        if (polarization == Expansion::E_TRAN) {
            for (size_t i = 0, end = expansion.mesh->size(); i < end; ++i) {
                incident[expansion.iEx(i)] = 1.;
                incident[expansion.iEz(i)] = 0.;
            }
        } else {
            for (size_t i = 0, end = expansion.mesh->size(); i < end; ++i) {
                incident[expansion.iEx(i)] = 0.;
                incident[expansion.iEz(i)] = 1.;
            }
        }
    }

    return transfer->diagonalizer->invTE(layer) * incident;
}

cvector LinesSolver2D::incidentGaussian(Transfer::IncidentDirection side,
                                        Expansion::Component polarization,
                                        double sigma,
                                        double center,
                                        dcomplex lam) {
    throw NotImplemented("LinesSolver2D::incidentGaussian"); // TODO: write this method
    // size_t layer = initIncidence(side, polarization, lam);

    // double b = 2.*PI / (expansion.right-expansion.left) * (expansion.symmetric()? 0.5 : 1.0);
    // dcomplex d = I * b * (center - expansion.left);
    // double c2 = - 0.5 * sigma*sigma * b*b;

    // int size = int(mesh->size());

    // cvector incident(expansion.matrixSize(), 0.);
    // for (int i = -size; i <= size; ++i) {
    //     size_t idx;
    //     if (expansion.separated()) idx = expansion.iEH(i);
    //     else idx = (polarization == Expansion::E_TRAN)? expansion.iEx(i) : expansion.iEz(i);
    //     dcomplex val = exp(c2 * double(i*i) - d*double(i));
    //     incident[idx] = (polarization == Expansion::E_TRAN)? val : -val;
    // }

    // return transfer->diagonalizer->invTE(layer) * incident;
}

}}}  // namespace plask::optical::slab
