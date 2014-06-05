#ifndef PLASK__SOLVER_SLAB_SLABBASE_H
#define PLASK__SOLVER_SLAB_SLABBASE_H

#include <plask/plask.hpp>
#include "rootdigger.h"

#undef interface

namespace plask { namespace solvers { namespace slab {

/// Information about lateral PMLs
struct PML {
    dcomplex factor;  ///< PML factor
    double size;      ///< Size of the PMLs
    double shift;     ///< Distance of the PMLs from defined computational domain
    double order;     ///< Order of the PMLs
    PML(): factor(1.,0.), size(1.), shift(0.5), order(2) {}
};

/**
 * Base class for all slab solvers
 */
template <typename GeometryT>
struct SlabSolver: public SolverOver<GeometryT> {

  protected:

    /// Determinant logger
    Data2DLog<dcomplex,dcomplex> detlog;

    /// Layer boundaries
    RectilinearAxis vbounds;

    /// Vertical positions of elements in each layer set
    std::vector<RectilinearAxis> lverts;

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

    /// Create and return rootdigger of a desired type
    std::unique_ptr<RootDigger> getRootDigger(const RootDigger::function_type& func);

  public:

    /// Distance outside outer borders where material is sampled
    double outdist;

    /// Smoothing coefficient
    double smooth;

    /// Parameters for main rootdigger
    RootDigger::Params root;

    /// Receiver for the temperature
    ReceiverFor<Temperature, GeometryT> inTemperature;

    /// Receiver for the gain
    ReceiverFor<Gain, GeometryT> inGain;

    /// Provider of the optical field intensity
    typename ProviderFor<LightMagnitude, GeometryT>::Delegate outLightMagnitude;

    /// Provider of the optical electric field
    typename ProviderFor<LightE, GeometryT>::Delegate outElectricField;

    /// Provider of the optical magnetic field
    typename ProviderFor<LightH, GeometryT>::Delegate outMagneticField;

    SlabSolver(const std::string& name="");

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
        if (index == 0 || index > vbounds.size())
            throw BadInput(this->getId(), "Cannot set interface to %1% (min: 1, max: %2%)", index, vbounds.size());
        double pos = vbounds[interface-1]; if (abs(pos) < 1e12) pos = 0.;
        this->writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)", pos, index);
        interface = index;
    }

    /**
     * Set the position of the matching interface.
     * \param pos vertical position close to the point where interface will be set
     */
    inline void setInterfaceAt(double pos) {
        if (vbounds.empty()) prepareLayers();
        interface = std::lower_bound(vbounds.begin(), vbounds.end(), pos-1e-12) - vbounds.begin() + 1; // -1e-12 to compensate for truncation errors
        if (interface > vbounds.size()) interface = vbounds.size();
        pos = vbounds[interface-1]; if (abs(pos) < 1e12) pos = 0.;
        this->writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)", pos, interface);
    }

    /**
     * Set the position of the matching interface at the bottom of the provided geometry object
     * \param object where the interface should  be set on
     * \param path path specifying object in the geometry
     */
    void setInterfaceOn(const shared_ptr<GeometryObject>& object, const PathHints* path=nullptr) {
        if (vbounds.empty()) prepareLayers();
        auto boxes = this->geometry->getObjectBoundingBoxes(object, path);
        if (boxes.size() != 1) throw NotUniqueObjectException();
        interface = std::lower_bound(vbounds.begin(), vbounds.end(), boxes[0].lower.vert()-1e-12) - vbounds.begin() + 1;
        if (interface > vbounds.size()) interface = vbounds.size();
        double pos = vbounds[interface-1]; if (abs(pos) < 1e12) pos = 0.;
        this->writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)", pos, interface);
    }

    /**
     * Set the position of the matching interface at the bottom of the provided geometry object
     * \param object where the interface should  be set on
     * \param path path specifying object in the geometry
     */
    void setInterfaceOn(const shared_ptr<GeometryObject>& object, const PathHints& path) {
        setInterfaceOn(object, &path);
    }

    /// Throw exception if the interface position is unsuitable for eigenmode computations
    void ensureInterface() {
        if (interface == 0 || interface >= stack.size())
            throw BadInput(this->getId(), "Wrong interface position %1% (min: 1, max: %2%)", interface, stack.size()-1);
    }

    /**
     * Get layer number for vertical coordinate. Alter this coordintate to the layer local one.
     * The bottom infinite layer has always negative coordinate.
     * \param[in,out] h vertical coordinate
     * \return layer number (in the stack)
     */
    size_t getLayerFor(double& h) const {
        size_t n = std::upper_bound(vbounds.begin(), vbounds.end(), h) - vbounds.begin();
        if (n == 0) h -= vbounds[0];
        else h -= vbounds[n-1];
        return n;
    }

    /// Get stack
    /// \return layers stack
    const std::vector<std::size_t>& getStack() const { return stack; }

    /// Get list of vertical positions of layers in each set
    /// \return layer sets
    const std::vector<RectilinearAxis>& getLayersPoints() const { return lverts; }

    /// Get list of vertical positions of layers in one set
    /// \param n set number
    /// \return layer sets
    const RectilinearAxis& getLayerPoints(size_t n) const { return lverts[n]; }

  protected:

    /**
     * Return number of determined modes
     */
    virtual size_t nummodes() const = 0;

    /**
     * Compute electric field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    virtual const DataVector<const Vec<3,dcomplex>> getE(size_t num, const MeshD<GeometryT::DIM>& dst_mesh, InterpolationMethod method) = 0;

    /**
     * Compute magnetic field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    virtual const DataVector<const Vec<3,dcomplex>> getH(size_t num, const MeshD<GeometryT::DIM>& dst_mesh, InterpolationMethod method) = 0;

    /**
     * Compute normalized electric field intensity 1/2 E conj(E) / P
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    virtual const DataVector<const double> getIntensity(size_t num, const MeshD<GeometryT::DIM>& dst_mesh, InterpolationMethod method) = 0;

};


}}} // namespace

#endif // PLASK__SOLVER_SLAB_SLABBASE_H

