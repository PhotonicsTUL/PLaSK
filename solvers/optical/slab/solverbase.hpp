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
#ifndef PLASK__SOLVER_SLAB_SOLVERBASE_H
#define PLASK__SOLVER_SLAB_SOLVERBASE_H

#include <plask/plask.hpp>
#include "rootdigger.hpp"
#include "transfer.hpp"

#undef interface

namespace plask { namespace optical { namespace slab {

/// Information about lateral PMLs
struct PML {
    dcomplex factor;  ///< PML factor
    double size;      ///< Size of the PMLs
    double dist;      ///< Distance of the PMLs from defined computational domain
    double order;     ///< Order of the PMLs
    PML() : factor(1., 0.), size(1.), dist(0.5), order(1) {}
    PML(dcomplex factor, double size, double dist, double order) : factor(factor), size(size), dist(dist), order(order) {}
};

/**
 * Common base with layer details independent on the geomety
 */
struct PLASK_SOLVER_API SlabBase {
    /// Directions of the possible emission
    enum Emission {
        EMISSION_UNSPECIFIED = 0,  ///< Side emission (fields not normalized)
        EMISSION_TOP,              ///< Top emission
        EMISSION_BOTTOM,           ///< Bottom emission
        EMISSION_FRONT,            ///< Front emission
        EMISSION_BACK              ///< Back emission
    };

    /// Direction of the light emission for fields normalization
    Emission emission;

    /// Method of computing determinant in zero search
    Transfer::Determinant determinant_type;

  protected:
    /// Create and return rootdigger of a desired type
    std::unique_ptr<RootDigger> getRootDigger(const RootDigger::function_type& func, const char* name);

    /// Selected transfer method
    Transfer::Method transfer_method;

    /**
     * Read root digger configuration
     * \param reader XML reader
     */
    void readRootDiggerConfig(XMLReader& reader) {
        root.tolx = reader.getAttribute<double>("tolx", root.tolx);
        root.tolf_min = reader.getAttribute<double>("tolf-min", root.tolf_min);
        root.tolf_max = reader.getAttribute<double>("tolf-max", root.tolf_max);
        root.maxstep = reader.getAttribute<double>("maxstep", root.maxstep);
        root.maxiter = reader.getAttribute<int>("maxiter", root.maxiter);
        root.alpha = reader.getAttribute<double>("alpha", root.alpha);
        root.lambda_min = reader.getAttribute<double>("lambd", root.lambda_min);
        root.initial_dist = reader.getAttribute<dcomplex>("initial-range", root.initial_dist);
        root.method = reader.enumAttribute<RootDigger::Method>("method")
                          .value("brent", RootDigger::ROOT_BRENT)
                          .value("broyden", RootDigger::ROOT_BROYDEN)
                          .value("muller", RootDigger::ROOT_MULLER)
                          .get(root.method);
        reader.requireTagEnd();
    }

  public:
    /// Transfer method object (AdmittanceTransfer or ReflectionTransfer)
    std::unique_ptr<Transfer> transfer;

    /// Initialize transfer class
    void initTransfer(Expansion& expansion, bool reflection);

    /// Layer boundaries
    shared_ptr<OrderedAxis> vbounds;

    /// Centers of layers
    shared_ptr<OrderedAxis> verts;

    /// Number of distinct layers
    size_t lcount;

    /// Information if the layer has gain
    std::vector<bool> lgained;

    /// Information if the layer has data from inEpsilon
    std::vector<bool> lcomputed;

    /// Organization of layers in the stack
    std::vector<std::size_t> stack;

    /// Index of the matching interface
    std::ptrdiff_t interface;

    /// Approximate position of the matching interface
    double interface_position;

    /// Reference wavelength used for getting material parameters (nm)
    double lam0;

    /// Normalized frequency (1/µm)
    dcomplex k0;

    /// Parameters for vertical PMLs (if used)
    PML vpml;

    /// Parameters for main rootdigger
    RootDigger::Params root;

    /// Force re-computation of material coefficients/integrals
    bool recompute_integrals;

    /// Force re-computation of material coefficients/integrals in layers with gain
    bool recompute_gain_integrals;

    /// Always compute material coefficients/integrals for gained layers for current wavelength
    bool always_recompute_gain;

  protected:
    /// Can layers be automatically grouped
    bool group_layers;

    /// Maximum temperature difference for grouped layers (NAN means ignore temperature)
    double max_temp_diff;

    /// Approxximate lateral distance between points for temperature investigation
    double temp_dist;

    /// Minimum layer thickness for the purpose of temperature-based layers division
    double temp_layer;

    /// Scale the incident field vector
    void scaleIncidentVector(cvector& incident, size_t layer, double size_factor);

  public:
    SlabBase()
        : emission(EMISSION_UNSPECIFIED),
          determinant_type(Transfer::DETERMINANT_EIGENVALUE),
          transfer_method(Transfer::METHOD_AUTO),
          interface(-1),
          interface_position(NAN),
          lam0(NAN),
          k0(NAN),
          vpml(dcomplex(1., -2.), 2.0, 10., 0),
          recompute_integrals(true),
          always_recompute_gain(true),
          group_layers(true),
          max_temp_diff(NAN),
          temp_dist(0.5),
          temp_layer(0.05) {}

    virtual ~SlabBase() {}

    /// Get lam0
    double getLam0() const { return lam0; }
    /// Set lam0
    void setLam0(double lam) {
        lam0 = lam;
        if (!isnan(lam) && (isnan(real(k0)) || isnan(imag(k0)))) k0 = 2e3 * PI / lam;
    }
    /// Clear lam0
    void clearLam0() { lam0 = NAN; }

    /// Get current k0
    dcomplex getK0() const { return k0; }

    /// Set current k0
    void setK0(dcomplex k) {
        k0 = k;
        if (k0 == 0.) k0 = 1e-12;
    }

    /// Get current wavelength
    dcomplex getLam() const { return 2e3 * PI / k0; }

    /// Set current wavelength
    void setLam(dcomplex lambda) { k0 = 2e3 * PI / lambda; }

    /// Reset determined fields
    void clearFields() {
        if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
    }

    /**
     * Get layer number for vertical coordinate. Alter this coordinate to the layer local one.
     * The bottom infinite layer has always negative coordinate.
     * \param[in,out] h vertical coordinate
     * \return layer number (in the stack)
     */
    size_t getLayerFor(double& h) const {
        size_t n = vbounds->findUpIndex(h + 1e-15);
        if (n == 0)
            h -= vbounds->at(0);
        else
            h -= vbounds->at(n - 1);
        return n;
    }

    /// Get solver id
    virtual std::string getId() const = 0;

    /// Init calculations
    virtual bool initCalculation() = 0;

    /// Get stack
    /// \return layers stack
    const std::vector<std::size_t>& getStack() const { return stack; }

    /// Recompute integrals used in RE and RH matrices
    virtual void computeIntegrals() = 0;

    /// Clear computed modes
    virtual void clearModes() = 0;

    /** Set expansion parameters from default values
     * \param with_k0 Change k0
     * \returns \c true if anything was changed
     */
    virtual bool setExpansionDefaults(bool with_k0 = true) = 0;

    /// Throw exception if the interface position is unsuitable for eigenmode computations
    void ensureInterface() {
        if (interface == -1) throw BadInput(this->getId(), "no interface position set");
        if (interface == 0 || interface >= std::ptrdiff_t(stack.size()))
            throw BadInput(this->getId(), "wrong interface position {0} (min: 1, max: {1})", interface, stack.size() - 1);
    }

    /// Get solver expansion
    virtual Expansion& getExpansion() = 0;

    /**
     * Get amplitudes of incident diffraction orders
     * \param incident incident field vector
     * \param side incidence side
     */
    dvector getIncidentFluxes(const cvector& incident, Transfer::IncidentDirection side);

    /**
     * Get amplitudes of reflected diffraction orders
     * \param incident incident field vector
     * \param side incidence side
     */
    dvector getReflectedFluxes(const cvector& incident, Transfer::IncidentDirection side);

    /**
     * Get amplitudes of transmitted diffraction orders
     * \param incident incident field vector
     * \param side incidence side
     */
    dvector getTransmittedFluxes(const cvector& incident, Transfer::IncidentDirection side);

    /**
     * Get coefficients of reflected diffraction orders
     * \param incident incident field vector
     * \param side incidence side
     */
    cvector getReflectedCoefficients(const cvector& incident, Transfer::IncidentDirection side);

    /**
     * Get coefficients of transmitted diffraction orders
     * \param incident incident field vector
     * \param side incidence side
     */
    cvector getTransmittedCoefficients(const cvector& incident, Transfer::IncidentDirection side);

    /**
     * Get reflection coefficient
     * \param incident incident field vector
     * \param side incidence side
     */
    double getReflection(const cvector& incident, Transfer::IncidentDirection side) {
        double R = 0.;
        for (double r : getReflectedFluxes(incident, side)) R += r;
        return R;
    }

    /**
     * Get reflection coefficient
     * \param incident incident field vector
     * \param side incidence side
     */
    double getTransmission(const cvector& incident, Transfer::IncidentDirection side) {
        double T = 0.;
        for (double t : getTransmittedFluxes(incident, side)) T += t;
        return T;
    }

#ifndef NDEBUG
    void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH);
#endif
};

}}}  // namespace plask::optical::slab

#endif  // PLASK__SOLVER_SLAB_SOLVERBASE_H
