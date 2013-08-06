#ifndef PLASK__SOLVER_REFLECTION_SOLVER_H
#define PLASK__SOLVER_REFLECTION_SOLVER_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace modal {

/**
 * Reflection transformation solver in Cartesian 2D geometry.
 */
struct FourierReflection2D: public SolverOver<Geometry2DCartesian> {

    std::string getClassName() const { return "modal.FourierReflection2D"; }

    /// Information about lateral PMLs
    struct PML {
        double extinction;  ///< Extinction of the PMLs
        double size;        ///< Size of the PMLs
        double shift;       ///< Distance of the PMLs from defined computational domain
        double order;       ///< Order of the PMLs
    };

  protected:

    /// Layer boundaries
    RectilinearMesh1D vbounds;

    /// Vertical positions of elements in each layer set
    std::vector<RectilinearMesh1D> lverts;

    /// Organization of layers in the stack
    std::vector<std::size_t> stack;

    /// Position of the matching interface
    size_t interface;

    /// Maximum order of orthogonal base
    size_t order;

    /// Mesh multiplier for finer computation of the refractive indices
    size_t refine;

    /// Lateral PMLs
    PML pml;

    void onGeometryChange(const Geometry::Event& evt) {
        this->invalidate();
        if (!vbounds.empty()) prepareLayers(); // update layers
    }

    /// Compute layer boundaries
    void prepareLayers();

    /// Detect layer sets and set them up
    void setupLayers();

  public:

    /// Distance outside outer borders where material is sampled
    double outdist;

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
        if (vbounds.empty()) prepareLayers();
        if (index >= vbounds.size())
            throw BadInput(getId(), "wrong interface position");
        this->writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)",  vbounds[index], index);
        interface = index;
    }

    /**
     * Set the position of the matching interface.
     * \param pos vertical position close to the point where interface will be set
     */
    inline void setInterfaceAt(double pos) {
        if (vbounds.empty()) prepareLayers();
        interface = std::lower_bound(vbounds.begin(), vbounds.end(), pos) - vbounds.begin();
        if (interface >= vbounds.size()) interface = vbounds.size() - 1;
        this->writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)",  vbounds[interface], interface);
    }

    /**
     * Set the position of the matching interface at the top of the provided geometry object
     * \param path path to the object in the geometry
     */
    void setInterfaceOn(const PathHints& path) {
        if (vbounds.empty()) prepareLayers();
        auto boxes = geometry->getLeafsBoundingBoxes(path);
        if (boxes.size() != 1) throw NotUniqueObjectException();
        interface = std::lower_bound(vbounds.begin(), vbounds.end(), boxes[0].upper.vert()) - vbounds.begin();
        if (interface >= vbounds.size()) interface = vbounds.size() - 1;
        this->writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)",  vbounds[interface], interface);
    }

    /// Get stack
    /// \return layers stack
    const std::vector<std::size_t>& getStack() const { return stack; }

    /// Get list of vertical positions of layers in each set
    /// \return layer sets
    const std::vector<RectilinearMesh1D>& getLayersPoints() const { return lverts; }

  protected:

    /**
     * Compute normalized electric field intensity 1/2 E conj(E) / P
     */
    const DataVector<const double> getIntensity(const MeshD<2>& dst_mesh, InterpolationMethod method);

};


}}} // namespace

#endif

