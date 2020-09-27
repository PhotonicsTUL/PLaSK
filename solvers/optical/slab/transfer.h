#ifndef PLASK__SLAB_TRANSFER_H
#define PLASK__SLAB_TRANSFER_H

#include <plask/solver.h>

#include "matrices.h"

namespace plask { namespace optical { namespace slab {

/// Field identification
enum WhichField {
    FIELD_E,  ///< Electric field
    FIELD_H   ///< Magnetic field
};

/// Light propagtion direction
enum PropagationDirection {
    PROPAGATION_TOTAL = 0,     ///< Both directions in total
    PROPAGATION_UPWARDS = 1,   ///< Upwards or towards the interface
    PROPAGATION_DOWNWARDS = 2  ///< Downwards or outside of the interface
};

using phys::Z0;

struct SlabBase;
struct Expansion;
class Diagonalizer;

/**
 * Base class for Admittance and Reflection transfers.
 */
struct PLASK_SOLVER_API Transfer {
    /// Direction specification for reflection calculations
    enum IncidentDirection {
        INCIDENCE_TOP,    ///< Incident light propagating from top (downwards)
        INCIDENCE_BOTTOM  ///< Incident light propagating from bottom (upwards)
    };

    /// Indicates what has been determined
    enum Determined {
        DETERMINED_NOTHING = 0,  ///< Nothing has been determined
        DETERMINED_RESONANT,     ///< Resonant field has been determined
        DETERMINED_REFLECTED     ///< Reflected field has been determined
    };

    /// Available transfer types
    enum Method {
        METHOD_AUTO,                   ///< Automatically selected transfer method
        METHOD_REFLECTION_ADMITTANCE,  ///< Reflection transfer (with admittance matching)
        METHOD_REFLECTION_IMPEDANCE,   ///< Reflection transfer (with impedance matching)
        METHOD_ADMITTANCE,             ///< Admittance transfer
        METHOD_IMPEDANCE,              ///< Impedance transfer
    };

  protected:
    cmatrix interface_field_matrix;  ///< Determined field at the interface
    dcomplex* interface_field;       ///< Pointer to the interface field data

    cmatrix M;  ///< The final matrix which must fulfill M * E = 0

    cmatrix temp;  ///< Temporary matrix

    dcomplex* evals;   ///< Found eigenvalues of matrix M
    double* rwrk;      ///< Temporary space
    std::size_t lwrk;  ///< Temporary space
    dcomplex* wrk;     ///< Temporary space

    cvector incident_vector;  ///< Incident vector, for which the fields are determined

    SlabBase* solver;  ///< Solver containing this transfer

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
    virtual cvector getReflectionVector(const cvector& incident, IncidentDirection side) = 0;

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
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     * \return electric field coefficients
     */
    virtual cvector getFieldVectorE(double z, std::size_t n, PropagationDirection part = PROPAGATION_TOTAL) = 0;

    /**
     * Compute magnetic field coefficients for given \a z
     * \param z position within the layer
     * \param n layer number
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     * \return magnetic field coefficients
     */
    virtual cvector getFieldVectorH(double z, std::size_t n, PropagationDirection part = PROPAGATION_TOTAL) = 0;

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
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     * \param reflected is this method called from reflected calculations?
     */
    LazyData<Vec<3, dcomplex>> computeFieldE(double power,
                                             const shared_ptr<const Mesh>& dst_mesh,
                                             InterpolationMethod method,
                                             bool reflected,
                                             PropagationDirection part = PROPAGATION_TOTAL);

    /**
     * Compute magnetic field at the given mesh.
     * \param power mode power
     * \param dst_mesh target mesh
     * \param method interpolation method
     * \param reflected is this method called from reflected calculations?
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     */
    LazyData<Vec<3, dcomplex>> computeFieldH(double power,
                                             const shared_ptr<const Mesh>& dst_mesh,
                                             InterpolationMethod method,
                                             bool reflected,
                                             PropagationDirection part = PROPAGATION_TOTAL);

    /**
     * Compute light magnitude.
     * \param power mode power
     * \param dst_mesh destination mesh
     * \param method interpolation method
     * \param reflected is the field emitting?
     */
    LazyData<double> computeFieldMagnitude(double power,
                                           const shared_ptr<const Mesh>& dst_mesh,
                                           InterpolationMethod method,
                                           bool reflected) {
        auto E = computeFieldE(1., dst_mesh, method, reflected);
        power *= 0.5 / Z0;  // because <M> = ½ E conj(E) / Z0
        return LazyData<double>(E.size(), [power, E](size_t i) { return power * abs2(E[i]); });
    }

    /**
     * Compute ½ E·conj(E) or ½ H·conj(H) integral between \a z1 and \a z2
     * \param field field to integrate
     * \param n layer number
     * \param z1 lower integration bound in local layer coordinates
     * \param z2 upper integration bound in local layer coordinates
     * \return computed integral
     */
    virtual double integrateField(WhichField field,
                                  size_t n,
                                  double z1,
                                  double z2) = 0;

  public:
    /**
     * Get electric field at the given mesh for resonant mode.
     * \param power mode power
     * \param dst_mesh target mesh
     * \param method interpolation method
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     */
    LazyData<Vec<3, dcomplex>> getFieldE(double power,
                                         const shared_ptr<const Mesh>& dst_mesh,
                                         InterpolationMethod method,
                                         PropagationDirection part = PROPAGATION_TOTAL) {
        determineFields();
        return computeFieldE(power, dst_mesh, method, false, part);
    }

    /**
     * Get magnetic field at the given mesh for resonant mode.
     * \param power mode power
     * \param dst_mesh target mesh
     * \param method interpolation method
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     */
    LazyData<Vec<3, dcomplex>> getFieldH(double power,
                                         const shared_ptr<const Mesh>& dst_mesh,
                                         InterpolationMethod method,
                                         PropagationDirection part = PROPAGATION_TOTAL) {
        determineFields();
        return computeFieldH(power, dst_mesh, method, false, part);
    }

    /**
     * Get light magnitude for resonant mode.
     * \param power mode power
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<double> getFieldMagnitude(double power,
                                       const shared_ptr<const Mesh>& dst_mesh,
                                       InterpolationMethod method) {
        determineFields();
        return computeFieldMagnitude(power, dst_mesh, method, false);
    }

    /**
     * Get electric field at the given mesh for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh target mesh
     * \param method interpolation method
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     */
    LazyData<Vec<3, dcomplex>> getScatteredFieldE(const cvector& incident,
                                                  IncidentDirection side,
                                                  const shared_ptr<const Mesh>& dst_mesh,
                                                  InterpolationMethod method,
                                                  PropagationDirection part = PROPAGATION_TOTAL) {
        determineReflectedFields(incident, side);
        return computeFieldE(1e3 * Z0, dst_mesh, method, true, part);
    }

    /**
     * Get magnetic field at the given mesh for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh target mesh
     * \param method interpolation method
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     */
    LazyData<Vec<3, dcomplex>> getScatteredFieldH(const cvector& incident,
                                                  IncidentDirection side,
                                                  const shared_ptr<const Mesh>& dst_mesh,
                                                  InterpolationMethod method,
                                                  PropagationDirection part = PROPAGATION_TOTAL) {
        determineReflectedFields(incident, side);
        return computeFieldH(1e3 * Z0, dst_mesh, method, true, part);
    }

    /**
     * Get light magnitude for reflected light.
     * \param incident incident field vector
     * \param side incidence side
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<double> getScatteredFieldMagnitude(const cvector& incident,
                                                IncidentDirection side,
                                                const shared_ptr<const Mesh>& dst_mesh,
                                                InterpolationMethod method) {
        determineReflectedFields(incident, side);
        return computeFieldMagnitude(1e3 * Z0, dst_mesh, method, true);
    }

    /**
     * Compute electric field coefficients for given \a z
     * \param z position within the layer
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     * \return electric field coefficients
     */
    cvector getFieldVectorE(double z, PropagationDirection part = PROPAGATION_TOTAL);

    /**
     * Compute magnetic field coefficients for given \a z
     * \param z position within the layer
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     * \return magnetic field coefficients
     */
    cvector getFieldVectorH(double z, PropagationDirection part = PROPAGATION_TOTAL);

    /**
     * Compute electric field coefficients for given \a z
     * \param incident incident field vector
     * \param side incidence side
     * \param z position within the layer
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     * \return electric field coefficients
     */
    cvector getScatteredFieldVectorE(const cvector& incident,
                                     IncidentDirection side,
                                     double z,
                                     PropagationDirection part = PROPAGATION_TOTAL);

    /**
     * Compute magnetic field coefficients for given \a z
     * \param incident incident field vector
     * \param side incidence side
     * \param z position within the layer
     * \param part part of the field (forward-, backward-propagating, or total) that is wanted
     * \return magnetic field coefficients
     */
    cvector getScatteredFieldVectorH(const cvector& incident,
                                     IncidentDirection side,
                                     double z,
                                     PropagationDirection part = PROPAGATION_TOTAL);

    /**
     * Get ½ E·conj(E) or ½ H·conj(H) integral between \a z1 and \a z2
     * \param field field to integrate
     * \param z1 lower integration bound
     * \param z2 upper integration bound
     * \param power mode emitted power
     * \return computed integral
     */
    double getFieldIntegral(WhichField field, double z1, double z2, double power);

    /**
     * Get ½ E·conj(E) or ½ H·conj(H) integral between \a z1 and \a z2 for reflected light
     * \param field field to integrate
     * \param incident incident field vector
     * \param side incidence side
     * \param z1 lower integration bound
     * \param z2 upper integration bound
     * \return computed integral
     */
    double getScatteredFieldIntegral(WhichField field,
                                     const cvector& incident,
                                     IncidentDirection side,
                                     double z1,
                                     double z2);
};

}}}  // namespace plask::optical::slab

#endif  // PLASK__SLAB_TRANSFER_H
