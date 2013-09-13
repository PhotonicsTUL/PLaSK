#ifndef PLASK__SOLVER_SLAB_REFLECTION_2D_H
#define PLASK__SOLVER_SLAB_REFLECTION_2D_H

#include <plask/plask.hpp>

#include "reflection_base.h"
#include "expansion_pw2d.h"

namespace plask { namespace solvers { namespace slab {

/**
 * Reflection transformation solver in Cartesian 2D geometry.
 */
struct FourierReflection2D: public ReflectionSolver<Geometry2DCartesian> {

    std::string getClassName() const { return "optical.FourierReflection2D"; }

    /// Information about lateral PMLs
    struct PML {
        double extinction;  ///< Extinction of the PMLs
        double size;        ///< Size of the PMLs
        double shift;       ///< Distance of the PMLs from defined computational domain
        int order;          ///< Order of the PMLs
        PML(): extinction(2.), size(1.), shift(0.5), order(2) {}
    };

  protected:

    /// Maximum order of the orthogonal base
    size_t size;

    /// Class responsoble for computing expansion coefficients
    ExpansionPW2D expansion;

    void onInitialize();

    void onInvalidate();

  public:

    /// Mesh multiplier for finer computation of the refractive indices
    size_t refine;

    /// Lateral PMLs
    PML pml;

    /// Provider for computed effective index
    ProviderFor<EffectiveIndex>::WithValue outNeff;

    FourierReflection2D(const std::string& name="");

    void loadConfiguration(XMLReader& reader, Manager& manager);

    /**
     * Find the mode around the specified effective index.
     * This method remembers the determined mode, for retrieval of the field profiles.
     * \param neff initial effective index to search the mode around
     * \return determined effective index
     */
    size_t findMode(dcomplex neff);

    /// Get order of the orthogonal base
    size_t getSize() const { return size; }
    /// Set order of the orthogonal base
    void setSize(size_t n) {
        size = n;
        invalidate();
    }

    /// \return current wavelength
    dcomplex getWavelength() const { return 2e3*M_PI / k0; }

    /**
     * Set new polarization
     * \param polar new polarization
     */
    void setWavelength(dcomplex wavelength) {
        k0 = 2e3*M_PI / wavelength;
        //if (!modes.empty()) writelog(LOG_DETAIL, "Clearing the computed modes");
        //modes.clear();
    }

    /**
     * Get refractive index after expansion
     */
    DataVector<const Tensor3<dcomplex>> getRefractiveIndexProfile(const RectilinearMesh2D& dst_mesh,
                                            InterpolationMethod interp=INTERPOLATION_DEFAULT);

  protected:

    /**
     * Compute normalized electric field intensity 1/2 E conj(E) / P
     */
    const DataVector<const double> getIntensity(size_t num, const MeshD<2>& dst_mesh, InterpolationMethod method);

};


}}} // namespace

#endif

