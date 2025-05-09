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
#include <memory>

#include "solvercyl.hpp"

namespace plask { namespace optical { namespace modal {

BesselSolverCyl::BesselSolverCyl(const std::string& name)
    : ModalSolver<SolverWithMesh<Geometry2DCylindrical, MeshAxis>>(name),
      domain(DOMAIN_INFINITE),
      m(1),
      size(12),
      rule(RULE_DIRECT),
      kscale(1.),
      kmax(5.),
      kmethod(WAVEVECTORS_NONUNIFORM),
      integral_error(1e-6),
      max_integration_points(1000),
      outLoss(this, &BesselSolverCyl::getModalLoss, &BesselSolverCyl::nummodes) {
    pml.dist = 20.;
    pml.size = 0.;
    this->writelog(LOG_WARNING, "This is an EXPERIMENTAL solver! Calculation results may not be reliable!");
}

void BesselSolverCyl::loadConfiguration(XMLReader& reader, Manager& manager) {
    while (reader.requireTagOrEnd()) {
        std::string param = reader.getNodeName();
        if (param == "expansion") {
            domain = reader.enumAttribute<BesselDomain>("domain")
                         .value("finite", DOMAIN_FINITE)
                         .value("infinite", DOMAIN_INFINITE)
                         .get(domain);
            size = reader.getAttribute<size_t>("size", size);
            group_layers = reader.getAttribute<bool>("group-layers", group_layers);
            lam0 = reader.getAttribute<double>("lam0", NAN);
            always_recompute_gain = reader.getAttribute<bool>("update-gain", always_recompute_gain);
            max_temp_diff = reader.getAttribute<double>("temp-diff", max_temp_diff);
            temp_dist = reader.getAttribute<double>("temp-dist", temp_dist);
            temp_layer = reader.getAttribute<double>("temp-layer", temp_layer);
            integral_error = reader.getAttribute<double>("integrals-error", integral_error);
            max_integration_points = reader.getAttribute<size_t>("integrals-points", max_integration_points);
            kscale = reader.getAttribute<double>("k-scale", kscale);
            kmax = reader.getAttribute<double>("k-max", kmax);
            kmethod = reader.enumAttribute<InfiniteWavevectors>("k-method")
                          .value("uniform", WAVEVECTORS_UNIFORM)
                          .value("nonuniform", WAVEVECTORS_NONUNIFORM)
                          .value("non-uniform", WAVEVECTORS_NONUNIFORM)
                          .value("laguerre", WAVEVECTORS_LAGUERRE)
                          .value("manual", WAVEVECTORS_MANUAL)
                          .get(kmethod);
            if (reader.hasAttribute("k-list")) {
                klist.clear();
                for (auto val : boost::tokenizer<boost::char_separator<char>>(reader.requireAttribute("k-list"),
                                                                              boost::char_separator<char>(" ,;\t\n"))) {
                    try {
                        double val = boost::lexical_cast<double>(val);
                        klist.push_back(val);
                    } catch (boost::bad_lexical_cast&) {
                        throw XMLException(reader, format("value '{0}' cannot be converted to float", val));
                    }
                }
            }
            if (reader.hasAttribute("k-weights")) {
                std::string value = reader.requireAttribute("k-weights");
                if (value.empty() || value == "auto")
                    kweights.reset();
                else {
                    kweights.reset(std::vector<double>());
                    kweights->clear();
                    for (auto val : boost::tokenizer<boost::char_separator<char>>(value, boost::char_separator<char>(" ,;\t\n"))) {
                        try {
                            double val = boost::lexical_cast<double>(val);
                            kweights->push_back(val);
                        } catch (boost::bad_lexical_cast&) {
                            throw XMLException(reader, format("value '{0}' cannot be converted to float", val));
                        }
                    }
                }
            }
            if (reader.hasAttribute("rule")) {
                rule = reader.enumAttribute<Rule>("rule")
                           .value("direct", RULE_DIRECT)
                           .value("combined1", RULE_COMBINED_1)
                           .value("combined2", RULE_COMBINED_2)
                           .value("old", RULE_OLD)
                           .require();
            }
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
            reader.requireTagEnd();
        } else if (param == "pml") {
            pml.factor = reader.getAttribute<dcomplex>("factor", pml.factor);
            pml.size = reader.getAttribute<double>("size", pml.size);
            pml.dist = reader.getAttribute<double>("dist", pml.dist);
            if (reader.hasAttribute("order")) {  // TODO Remove in the future
                writelog(LOG_WARNING, "XML line {:d} in <pml>: Attribute 'order' is obsolete, use 'shape' instead",
                         reader.getLineNr());
                pml.order = reader.requireAttribute<double>("order");
            }
            pml.order = reader.getAttribute<double>("shape", pml.order);
            reader.requireTagEnd();
        } else
            parseCommonModalConfiguration(reader, manager);
    }
}

void BesselSolverCyl::onInitialize() {
    if (size == 0)
        throw BadInput(getId(), "bessel solver size cannot be 0");
    this->setupLayers();
    std::string dom;
    switch (domain) {
        case DOMAIN_FINITE: dom = "finite"; break;
        case DOMAIN_INFINITE: dom = "infinite"; break;
        default: assert(0);
    }

    if (this->interface == -1)
        Solver::writelog(LOG_DETAIL, "Initializing BesselCyl solver in {} domain ({} layers in the stack)", dom,
                         this->stack.size());
    else
        Solver::writelog(LOG_DETAIL,
                         "Initializing BesselCyl solver in {} domain ({} layers in the stack, interface after {} layer{})", dom,
                         this->stack.size(), this->interface, (this->interface == 1) ? "" : "s");
    switch (domain) {
        case DOMAIN_FINITE: expansion.reset(new ExpansionBesselFini(this)); break;
        case DOMAIN_INFINITE: expansion.reset(new ExpansionBesselInfini(this)); break;
        default: assert(0);
    }
    setExpansionDefaults();
    expansion->init1();
    this->recompute_integrals = true;
}

void BesselSolverCyl::onInvalidate() {
    modes.clear();
    expansion->reset();
    transfer.reset();
}

size_t BesselSolverCyl::findMode(dcomplex start, int m) {
    Solver::initCalculation();
    ensureInterface();
    expansion->setLam0(this->lam0);
    expansion->setM(m);
    initTransfer(*expansion, false);
    std::unique_ptr<RootDigger> root = getRootDigger(
        [this](const dcomplex& x) {
            if (isnan(x)) throw ComputationError(this->getId(), "'lam' converged to NaN");
            expansion->setK0(2e3 * PI / x);
            return transfer->determinant();
        },
        "lam");
    root->find(start);
    return insertMode();
}


double BesselSolverCyl::getWavelength(size_t n) {
    if (n >= modes.size()) throw NoValue(ModeWavelength::NAME);
    return real(2e3 * M_PI / modes[n].k0);
}

#ifndef NDEBUG
cmatrix BesselSolverCyl::epsV_k(size_t layer) {
    Solver::initCalculation();
    computeIntegrals();
    return expansion->epsV_k(layer);
}
cmatrix BesselSolverCyl::epsTss(size_t layer) {
    Solver::initCalculation();
    computeIntegrals();
    return expansion->epsTss(layer);
}
cmatrix BesselSolverCyl::epsTpp(size_t layer) {
    Solver::initCalculation();
    computeIntegrals();
    return expansion->epsTpp(layer);
}
cmatrix BesselSolverCyl::epsTsp(size_t layer) {
    Solver::initCalculation();
    computeIntegrals();
    return expansion->epsTsp(layer);
}
cmatrix BesselSolverCyl::epsTps(size_t layer) {
    Solver::initCalculation();
    computeIntegrals();
    return expansion->epsTps(layer);
}

cmatrix BesselSolverCyl::muV_k() {
    Solver::initCalculation();
    if (auto finite_expansion = dynamic_cast<ExpansionBesselFini*>(expansion.get())) {
        computeIntegrals();
        return finite_expansion->muV_k();
    } else {
        return cmatrix();
    }
}
cmatrix BesselSolverCyl::muTss() {
    Solver::initCalculation();
    if (auto finite_expansion = dynamic_cast<ExpansionBesselFini*>(expansion.get())) {
        computeIntegrals();
        return finite_expansion->muTss();
    } else {
        return cmatrix();
    }
}
cmatrix BesselSolverCyl::muTsp() {
    Solver::initCalculation();
    if (auto finite_expansion = dynamic_cast<ExpansionBesselFini*>(expansion.get())) {
        computeIntegrals();
        return finite_expansion->muTsp();
    } else {
        return cmatrix();
    }
}
cmatrix BesselSolverCyl::muTps() {
    Solver::initCalculation();
    if (auto finite_expansion = dynamic_cast<ExpansionBesselFini*>(expansion.get())) {
        computeIntegrals();
        return finite_expansion->muTps();
    } else {
        return cmatrix();
    }
}
cmatrix BesselSolverCyl::muTpp() {
    Solver::initCalculation();
    if (auto finite_expansion = dynamic_cast<ExpansionBesselFini*>(expansion.get())) {
        computeIntegrals();
        return finite_expansion->muTpp();
    } else {
        return cmatrix();
    }
}
#endif

}}}  // namespace plask::optical::modal
