#ifndef PLASK__OPTICAL_MODAL_H
#define PLASK__OPTICAL_MODAL_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace modal {

/**
 * Base class for all modal solvers
 */
template <typename GeometryT>
struct ModalSolver: public SolverOver<GeometryT> {

  protected:

    /// Layer boundaries
    RectilinearMesh1D vbounds;

    /// Vertical positions of elements in each layer set
    std::vector<RectilinearMesh1D> lverts;

    /// Information if the layer has gain
    std::vector<bool> lgained;

    /// Organization of layers in the stack
    std::vector<std::size_t> stack;

    /// Position of the matching interface
    size_t interface;

    void onInitialize() {
        setupLayers();
    }

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

    /// Receiver for the temperature
    ReceiverFor<Temperature, GeometryT> inTemperature;

    /// Receiver for the gain
    ReceiverFor<Gain, GeometryT> inGain;

    /// Provider of optical field
    typename ProviderFor<OpticalIntensity, GeometryT>::Delegate outIntensity;

    ModalSolver(const std::string& name="");

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
            throw BadInput(this->getId(), "wrong interface position");
        this->writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)",  vbounds[index], index);
        interface = index;
    }

    /**
     * Set the position of the matching interface.
     * \param pos vertical position close to the point where interface will be set
     */
    inline void setInterfaceAt(double pos) {
        if (vbounds.empty()) prepareLayers();
        interface = std::lower_bound(vbounds.begin(), vbounds.end(), pos-1e-12) - vbounds.begin(); // -1e-12 to compensate for truncation errors
        if (interface >= vbounds.size()) interface = vbounds.size() - 1;
        pos = vbounds[interface]; if (abs(pos) < 1e12) pos = 0.;
        this->writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)",  pos, interface);
    }

    /**
     * Set the position of the matching interface at the top of the provided geometry object
     * \param object where the interface should  be set on
     * \param path path specyfing object in the geometry
     */
    void setInterfaceOn(const shared_ptr<GeometryObject>& object, const PathHints* path=nullptr) {
        if (vbounds.empty()) prepareLayers();
        auto boxes = this->geometry->getObjectBoundingBoxes(object, path);
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
    virtual const DataVector<const double> getIntensity(const MeshD<2>& dst_mesh, InterpolationMethod method) = 0;

};


}}} // namespace

#endif

