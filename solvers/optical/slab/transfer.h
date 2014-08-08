#ifndef PLASK__SLAB_TRANSFER_H
#define PLASK__SLAB_TRANSFER_H

#include <plask/solver.h>

#include "diagonalizer.h"

namespace plask { namespace solvers { namespace slab {

struct SlabBase;

/**
 * Base class for Admittance and Reflection transfers.
 */
struct PLASK_SOLVER_API Transfer {

    /// Direction specification for reflection calculations
    enum IncidentDirection {
        INCIDENCE_TOP,      ///< Incident light propagating from top (downwards)
        INCIDENCE_BOTTOM    ///< Incident light propagating from bottom (upwards)
    };

    /// Indicates what has been determined
    enum Determined {
        DETERMINED_NOTHING = 0, ///< Nothing has been determined
        DETERMINED_RESONANT,    ///< Resonant field has been determined
        DETERMINED_REFLECTED    ///< Reflected field has been determined
    };

    /// Solver containing this transfer
    SlabBase* solver;

    /// Diagonalizer used to compute matrix of eigenvalues and eigenvectors
    std::unique_ptr<Diagonalizer> diagonalizer;

    Determined fields_determined;               ///< Are the diagonalized fields determined for all layers?
    /**
     * Create transfer object and initialize memory
     * \param solver solver counting this transfer
     * \param expansion expansion for diagonalizer
     */
    Transfer(SlabBase* solver, Expansion& expansion):
        solver(solver),
        diagonalizer(new SimpleDiagonalizer(&expansion)),   //TODO add other diagonalizer types
        fields_determined(DETERMINED_NOTHING)
    {}

    /// Compute discontinuity matrix determinant for the current parameters
    virtual dcomplex determinant() = 0;

    /**
     * Get vector of reflection coefficients
     * \param incident incident field profile
     * \param side incident light side
     */
    virtual cvector getReflectionVector(const cvector& incident, IncidentDirection direction) = 0;

    /**
     * Get vector of transmission coefficients
     * \param incident incident field profile
     * \param side incident light side
     */
    virtual cvector getTransmissionVector(const cvector& incident, IncidentDirection side) = 0;

    /**
     * Get current expansion coefficients at the matching interface
     * \return vector of current expansion coefficients at the interface
     */
    virtual cvector getInterfaceVector() = 0;

    /**
     * Compute electric field coefficients for given \a z
     * \param z position within the layer
     * \param n layer number
     * \return electric field coefficients
     */
    virtual cvector getFieldVectorE(double z, int n) = 0;

    /**
     * Compute magnetic field coefficients for given \a z
     * \param z position within the layer
     * \param n layer number
     * \return magnetic field coefficients
     */
    virtual cvector getFieldVectorH(double z, int n) = 0;

    /**
     * Get electric field at the given mesh for resonant mode.
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    virtual DataVector<Vec<3,dcomplex>> getFieldE(const shared_ptr<const Mesh>& dst_mesh,
                                                  InterpolationMethod method) = 0;

    /**
     * Get magnetic field at the given mesh for resonant mode.
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    virtual DataVector<Vec<3,dcomplex>> getFieldH(const shared_ptr<const Mesh>& dst_mesh,
                                                  InterpolationMethod method) = 0;

    /**
     * Get light magnitude for resonant mode.
     * \param power mode power
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    virtual DataVector<double> getFieldMagnitude(double power,
                                                 const shared_ptr<const Mesh>& dst_mesh,
                                                 InterpolationMethod method) = 0;

    /**
     * Get electric field at the given mesh for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    virtual DataVector<Vec<3,dcomplex>> getReflectedFieldE(const cvector& incident,
                                                           IncidentDirection side,
                                                           const shared_ptr<const Mesh>& dst_mesh,
                                                           InterpolationMethod method) = 0;

    /**
     * Get magnetic field at the given mesh for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    virtual DataVector<Vec<3,dcomplex>> getReflectedFieldH(const cvector& incident,
                                                           IncidentDirection side,
                                                           const shared_ptr<const Mesh>& dst_mesh,
                                                           InterpolationMethod method) = 0;

    /**
     * Get light magnitude for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    virtual DataVector<double> getReflectedFieldMagnitude(const cvector& incident,
                                                          IncidentDirection side,
                                                          const shared_ptr<const Mesh>& dst_mesh,
                                                          InterpolationMethod method) = 0;
};


}}} // namespace plask::solvers::slab

#endif // PLASK__SLAB_TRANSFER_H
