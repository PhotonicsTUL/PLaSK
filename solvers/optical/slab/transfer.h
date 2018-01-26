#ifndef PLASK__SLAB_TRANSFER_H
#define PLASK__SLAB_TRANSFER_H

#include <plask/solver.h>

#include "diagonalizer.h"

namespace plask { namespace optical { namespace slab {

using phys::Z0;

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

    /// Available transfer types
    enum Method {
        METHOD_AUTO,            ///< Automatically selected transfer method
        METHOD_REFLECTION,      ///< Reflection transfer
        METHOD_ADMITTANCE       ///< Admittance transfer
    };

  protected:

    cmatrix interface_field_matrix;             ///< Determined field at the interface
    dcomplex* interface_field;                  ///< Pointer to the interface field data

    cmatrix M;                                  ///< The final matrix which must fulfill M * E = 0

    cmatrix temp;                               ///< Temporary matrix

    dcomplex* evals;                            ///< Found eigenvalues of matrix M
    double* rwrk;                               ///< Temporary space
    int lwrk;                                   ///< Temporary space
    dcomplex* wrk;                              ///< Temporary space

    /// Solver containing this transfer
    SlabBase* solver;

  public:

    /// Init diagonalization
    void initDiagonalization();

    /// Diagonalizer used to compute matrix of eigenvalues and eigenvectors
    std::unique_ptr<Diagonalizer> diagonalizer;

    /// Are the diagonalized fields determined for all layers?
    Determined fields_determined;

    /**
     * Create transfer object and initialize memory
     * \param solver solver counting this transfer
     * \param expansion expansion for diagonalizer
     */
    Transfer(SlabBase* solver, Expansion& expansion);

    virtual ~Transfer();

    /// Compute discontinuity matrix determinant for the current parameters
    dcomplex determinant();

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

  protected:

    /// Get the discontinuity matrix for the whole structure
    virtual void getFinalMatrix() = 0;

    /**
     * Determine coefficients in each layer necessary for fields calculations.
     */
    virtual void determineFields() = 0;

    /**
     * Determine coefficients in each layer necessary for fields calculations.
     * This method is called for reflected fields.
     * \param incident incident field vector
     * \param side incidence side
     */
    virtual void determineReflectedFields(const cvector& incident, IncidentDirection side) = 0;

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
     * Get current expansion coefficients at the matching interface
     * \return vector of current expansion coefficients at the interface
     */
    const_cvector getInterfaceVector();

    /**
     * Compute electric field at the given mesh.
     * \param power mode power
     * \param dst_mesh target mesh
     * \param method interpolation method
     * \param reflected is this method called from reflected calculations?
     */
    LazyData<Vec<3,dcomplex>> computeFieldE(double power, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method, bool reflected);

    /**
     * Compute magnetic field at the given mesh.
     * \param power mode power
     * \param dst_mesh target mesh
     * \param method interpolation method
     * \param reflected is this method called from reflected calculations?
     */
    LazyData<Vec<3,dcomplex>> computeFieldH(double power, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method, bool reflected);

    /**
     * Compute light magnitude.
     * \param power mode power
     * \param dst_mesh destination mesh
     * \param method interpolation method
     * \param reflected is the field emitting?
     */
    LazyData<double> computeFieldMagnitude(double power, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method, bool reflected) {
        auto E = computeFieldE(1., dst_mesh, method, reflected);
        power *= 0.5 / Z0; // because <M> = ½ E conj(E) / Z0
        return LazyData<double>(E.size(), [power,E](size_t i) { return power * abs2(E[i]); });
    }

  public:

    /**
     * Get electric field at the given mesh for resonant mode.
     * \param power mode power
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    LazyData<Vec<3,dcomplex>> getFieldE(double power, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineFields();
        return computeFieldE(power, dst_mesh, method, false);
    }

    /**
     * Get magnetic field at the given mesh for resonant mode.
     * \param power mode power
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    LazyData<Vec<3,dcomplex>> getFieldH(double power, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineFields();
        return computeFieldH(power, dst_mesh, method, false);
    }

    /**
     * Get light magnitude for resonant mode.
     * \param power mode power
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<double> getFieldMagnitude(double power, const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineFields();
        return computeFieldMagnitude(power, dst_mesh, method, false);
    }

    /**
     * Get electric field at the given mesh for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    LazyData<Vec<3,dcomplex>> getReflectedFieldE(const cvector& incident, IncidentDirection side,
                                                 const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineReflectedFields(incident, side);
        return computeFieldE(1e3 * Z0, dst_mesh, method, true);
    }

    /**
     * Get magnetic field at the given mesh for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    LazyData<Vec<3,dcomplex>> getReflectedFieldH(const cvector& incident, IncidentDirection side,
                                                 const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineReflectedFields(incident, side);
        return computeFieldH(1e3 * Z0, dst_mesh, method, true);
    }

    /**
     * Get light magnitude for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<double> getReflectedFieldMagnitude(const cvector& incident, IncidentDirection side,
                                                const shared_ptr<const Mesh>& dst_mesh, InterpolationMethod method) {
        determineReflectedFields(incident, side);
        return computeFieldMagnitude(1e3 * Z0, dst_mesh, method, true);
    }

    /**
     * Compute electric field coefficients for given \a z
     * \param z position within the layer
     * \return electric field coefficients
     */
    cvector getFieldVectorE(double z);

    /**
     * Compute magnetic field coefficients for given \a z
     * \param z position within the layer
     * \return magnetic field coefficients
     */
    cvector getFieldVectorH(double z);

    /**
     * Compute electric field coefficients for given \a z
     * \param incident incident field vector
     * \param side incidence side
     * \param z position within the layer
     * \return electric field coefficients
     */
    cvector getReflectedFieldVectorE(const cvector& incident, IncidentDirection side, double z);

    /**
     * Compute magnetic field coefficients for given \a z
     * \param incident incident field vector
     * \param side incidence side
     * \param z position within the layer
     * \return magnetic field coefficients
     */
    cvector getReflectedFieldVectorH(const cvector& incident, IncidentDirection side, double z);
};


}}} // namespace plask::optical::slab

#endif // PLASK__SLAB_TRANSFER_H
