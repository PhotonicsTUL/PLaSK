#ifndef PLASK__SOLVER_REFLECTION_SOLVER_H
#define PLASK__SOLVER_REFLECTION_SOLVER_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace modal {

/**
 * Reflection transformation solver in Cartesian 2D geometry.
 */
struct FourierReflection2D: public SolverOver<Geometry2DCartesian> {

  protected:

    /// Layer boundaries
    RectilinearMesh1D layers;

    /// Position of the matching interface
    size_t interface;

    /// Maximum order of orthogonal base
    size_t order;

    /// Mesh multiplier for finer computation of the refractive indices
    size_t refine;

    /// Set layer boundaries
    void setLayerMesh();

    virtual void onGeometryChange(const Geometry::Event& evt) {
        this->invalidate();
        if (!layers.empty()) setLayerMesh(); // update layers
    }

  public:

    /// Receiver of the wavelength
    ReceiverFor<Wavelength> inWavelength;

    /// Receiver for the temperature
    ReceiverFor<Temperature, Geometry2DCartesian> inTemperature;

    /// Receiver for the gain
    ReceiverFor<Gain, Geometry2DCartesian> inGain;

    /// Provider for computed effective index
    ProviderFor<EffectiveIndex>::WithValue outNeff;

    /// Provider of optical field
    ProviderFor<OpticalIntensity, Geometry2DCartesian>::Delegate outIntensity;

    FourierReflection2D(const std::string& name="");

    void loadConfiguration(XMLReader& reader, Manager& manager);

    /**
     * Get the position of the matching interface.
     * \return index of the vertical mesh, where interface is set
     */
    inline size_t getInterface() { return interface; }

    /**
     * Set the position of the matching interface.
     * \param index index of the vertical mesh, where interface is set
     */
    inline void setInterface(size_t index) {
        if (layers.empty()) setLayerMesh();
        if (index >= layers.size())
            throw BadInput(getId(), "wrong interface position");
        this->writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)",  layers[index], index);
        interface = index;
    }

    /**
     * Set the position of the matching interface at the top of the provided geometry object
     * \param path path to the object in the geometry
     */
    void setInterfaceOn(const PathHints& path) {
        if (layers.empty()) setLayerMesh();
        auto boxes = geometry->getLeafsBoundingBoxes(path);
        if (boxes.size() != 1) throw NotUniqueObjectException();
        interface = std::lower_bound(layers.begin(), layers.end(), boxes[0].upper.vert()) - layers.begin();
        if (interface >= layers.size()) interface = layers.size() - 1;
        this->writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)",  layers[interface], interface);
    }

  protected:

    /**
     * Compute normalized electric field intensity 1/2 E conj(E) / P
     */
    const DataVector<const double> getIntensity(const MeshD<2>& dst_mesh, InterpolationMethod method);

};


}}} // namespace

#endif

