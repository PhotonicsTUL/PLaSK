#ifndef PLASK__SOLVER_SLAB_REFLECTION_CYL_H
#define PLASK__SOLVER_SLAB_REFLECTION_CYL_H

#include <plask/plask.hpp>

#include "slab_base.h"

namespace plask { namespace solvers { namespace slab {

/**
 * Reflection transformation solver in Cartesian 2D geometry.
 */
struct FourierReflectionCyl: public ModalSolver<Geometry2DCylindrical> {

    std::string getClassName() const { return "slab.FourierReflectionCyl"; }

  protected:

    /// Maximum order of orthogonal base
    size_t order;

    /// Mesh multiplier for finer computation of the refractive indices
    size_t refine;

    void onInitialize();

  public:

    /// Receiver of the wavelength
    ReceiverFor<Wavelength> inWavelength;

    /// Provider for computed effective index
    ProviderFor<EffectiveIndex>::WithValue outNeff;

    FourierReflectionCyl(const std::string& name="");

    void loadConfiguration(XMLReader& reader, Manager& manager);


    /**
     * Find the mode around the specified effective index.
     * This method remembers the determined mode, for retrieval of the field profiles.
     * \param neff initial effective index to search the mode around
     * \return determined effective index
     */
    double computeMode(dcomplex neff);

  protected:

    /**
     * Compute normalized electric field intensity 1/2 E conj(E) / P
     */
    const DataVector<const double> getIntensity(const MeshD<2>& dst_mesh, InterpolationMethod method);

};


}}} // namespace

#endif

