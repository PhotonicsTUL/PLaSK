#ifndef PLASK__SOLVER_SLAB_SLABBASE_H
#define PLASK__SOLVER_SLAB_SLABBASE_H

#include <plask/plask.hpp>
#include "rootdigger.h"
#include "transfer.h"

#undef interface

namespace plask { namespace solvers { namespace slab {

/// Information about lateral PMLs
struct PML {
    dcomplex factor;  ///< PML factor
    double size;      ///< Size of the PMLs
    double dist;     ///< Distance of the PMLs from defined computational domain
    double order;     ///< Order of the PMLs
    PML(): factor(1.,0.), size(1.), dist(0.5), order(2) {}
    PML(dcomplex factor, double size, double dist, double order): factor(factor), size(size), dist(dist), order(order) {}
};

/**
 * Common base with layer details independent on the geomety
 */
struct PLASK_SOLVER_API SlabBase {

  protected:

    /// Determinant logger
    Data2DLog<dcomplex,dcomplex> detlog;

    /// Transfer method object (AdmittanceTransfer or ReflectionTransfer)
    std::unique_ptr<Transfer> transfer;

    /// Create and return rootdigger of a desired type
    std::unique_ptr<RootDigger> getRootDigger(const RootDigger::function_type& func);

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

    void initTransfer(Expansion& expansion, bool emitting);

  public:

    /// Layer boundaries
    OrderedAxis vbounds;

    /// Vertical positions of elements in each layer set
    std::vector<shared_ptr<OrderedAxis>> lverts;

    /// Information if the layer has gain
    std::vector<bool> lgained;

    /// Organization of layers in the stack
    std::vector<std::size_t> stack;

    /// Position of the matching interface
    size_t interface;

    /// Reference wavelength used for getting material parameters [nm]
    boost::optional<double> lam0;

    /// Normalized frequency [1/µm]
    dcomplex k0;
    
    /// Parameters for vertical PMLs (if used)
    PML vpml;

    /// Parameters for main rootdigger
    RootDigger::Params root;

    /// Force re-computation of material coefficients/integrals
    bool recompute_integrals;

    /// Always compute material coefficients/integtals for gained layers
    bool always_recompute_gain;

  protected:

    /// Can layers be automatically grouped
    bool group_layers;

  public:

    SlabBase():
        detlog("", "modal", "unspecified", "det"),
        transfer_method(Transfer::METHOD_AUTO),
        interface(size_t(-1)),
        k0(NAN),
        vpml(dcomplex(1.,-2.), 2.0, 10., 0),
        recompute_integrals(true), always_recompute_gain(false), group_layers(true) {}

    /// Get lam0
    double getLam0() const {
        if (lam0) return *lam0;
        else return NAN;
    }
    /// Set lam0
    void setLam0(double lam) {
        if (!lam0 || lam != *lam0) {
            lam0 = lam;
            this->recompute_integrals = true;
            if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        }
    }
    /// Clear lam0
    void clearLam0() {
        if (lam0) {
            lam0.reset();
            this->recompute_integrals = true;
            if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
        }
    }
    /// Set lam0
    void setLam0(boost::optional<double> lam) {
        if (lam) setLam0(*lam);
        else clearLam0();
    }

    /// Get current k0
    dcomplex getK0() const { return k0; }

    /// Set current k0
    void setK0(dcomplex k) {
        if (k != k0) {
            if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
            k0 = k;
            if (k0 == 0.) k0 = 1e-12;
            if (!lam0) this->recompute_integrals = true;
        }
    }

    /// Get current wavelength
    dcomplex getWavelength() const { return 2e3*M_PI / k0; }

    /// Set current wavelength
    void setWavelength(dcomplex lambda) {
        dcomplex k = 2e3*M_PI / lambda;
        if (k != k0) {
            if (transfer) transfer->fields_determined = Transfer::DETERMINED_NOTHING;
            k0 = k;
            if (!lam0) this->recompute_integrals = true;
        }
    }

    /**
     * Get layer number for vertical coordinate. Alter this coordinate to the layer local one.
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

    /// Get solver id
    virtual std::string getId() const = 0;

    /// Get stack
    /// \return layers stack
    const std::vector<std::size_t>& getStack() const { return stack; }

    /// Get list of vertical positions of layers in each set
    /// \return layer sets
    const std::vector<shared_ptr<OrderedAxis>>& getLayersPoints() const { return lverts; }

    /// Get list of vertical positions of layers in one set
    /// \param n set number
    /// \return layer sets
    shared_ptr<OrderedAxis> getLayerPoints(size_t n) const { return lverts[n]; }

    /// Recompute integrals used in RE and RH matrices
    virtual void computeIntegrals() = 0;

    /// Clear computed modes
    virtual void clear_modes() = 0;

};

/**
 * Base class for all slab solvers
 */
template <typename BaseT>
class PLASK_SOLVER_API SlabSolver: public BaseT, public SlabBase {

    /// Compute layer boundaries
    void setup_vbounds();

    /// Reset structure if input is changed
    void onInputChanged(ReceiverBase&, ReceiverBase::ChangeReason) {
        this->clear_modes();
        this->recompute_integrals = true;
    }

  protected:

    void onInitialize() {
        setupLayers();
    }

    void onGeometryChange(const Geometry::Event& evt) {
        BaseT::onGeometryChange(evt);
        if (!vbounds.empty()) setup_vbounds(); // update layers
    }

    /// Detect layer sets and set them up
    void setupLayers();

  public:

    /// Distance outside outer borders where the material is sampled
    double outdist;

    /// Smoothing coefficient
    double smooth;

    /// Receiver for the temperature
    ReceiverFor<Temperature, typename BaseT::SpaceType> inTemperature;

    /// Receiver for the gain
    ReceiverFor<Gain, typename BaseT::SpaceType> inGain;

    /// Provider of the refractive index
    typename ProviderFor<RefractiveIndex, typename BaseT::SpaceType>::Delegate outRefractiveIndex;

    /// Provider of the optical field intensity
    typename ProviderFor<LightMagnitude, typename BaseT::SpaceType>::Delegate outLightMagnitude;

    /// Provider of the optical electric field
    typename ProviderFor<LightE, typename BaseT::SpaceType>::Delegate outElectricField;

    /// Provider of the optical magnetic field
    typename ProviderFor<LightH, typename BaseT::SpaceType>::Delegate outMagneticField;

    SlabSolver(const std::string& name="");

    ~SlabSolver();

    /// Get currently selected transfer method
    Transfer::Method getTransferMethod() const { return transfer_method; }

    /// Set new transfer method
    void setTransferMethod(Transfer::Method method) {
        if (method != transfer_method) this->invalidate();
        transfer_method = method;
    }

    /// Getter for group_layers
    bool getGroupLayers() const { return group_layers; }
    
    /// Setter for group_layers
    void setGroupLayers(bool value) {
        bool changed = group_layers != value;
        group_layers = value;
        if (changed) this->invalidate();
    }

    std::string getId() const override { return Solver::getId(); }

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
        if (vbounds.empty()) setup_vbounds();
        if (index == 0 || index > vbounds.size())
            throw BadInput(this->getId(), "Cannot set interface to %1% (min: 1, max: %2%)", index, vbounds.size());
        double pos = vbounds[interface-1]; if (abs(pos) < 1e-12) pos = 0.;
        Solver::writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)", pos, index);
        interface = index;
    }

    /**
     * Set the position of the matching interface.
     * \param pos vertical position close to the point where interface will be set
     */
    inline void setInterfaceAt(double pos) {
        if (vbounds.empty()) setup_vbounds();
        interface = std::lower_bound(vbounds.begin(), vbounds.end(), pos-1e-12) - vbounds.begin() + 1; // -1e-12 to compensate for truncation errors
        if (interface > vbounds.size()) interface = vbounds.size();
        pos = vbounds[interface-1]; if (abs(pos) < 1e-12) pos = 0.;
        Solver::writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)", pos, interface);
    }

    /**
     * Set the position of the matching interface at the bottom of the provided geometry object
     * \param object where the interface should  be set on
     * \param path path specifying object in the geometry
     */
    void setInterfaceOn(const shared_ptr<GeometryObject>& object, const PathHints* path=nullptr) {
        if (vbounds.empty()) setup_vbounds();
        auto boxes = this->geometry->getObjectBoundingBoxes(object, path);
        if (boxes.size() != 1) throw NotUniqueObjectException();
        interface = std::lower_bound(vbounds.begin(), vbounds.end(), boxes[0].lower.vert()-1e-12) - vbounds.begin() + 1;
        if (interface > vbounds.size()) interface = vbounds.size();
        double pos = vbounds[interface-1]; if (abs(pos) < 1e-12) pos = 0.;
        Solver::writelog(LOG_DEBUG, "Setting interface at position %g (mesh index: %d)", pos, interface);
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
        if (interface == size_t(-1))
            throw BadInput(this->getId(), "No interface position set");
        if (interface == 0 || interface >= stack.size())
            throw BadInput(this->getId(), "Wrong interface position %1% (min: 1, max: %2%)", interface, stack.size()-1);
    }

    /// Get discontinuity matrix determinant for the current parameters
    dcomplex getDeterminant() {
        this->initCalculation();
        initTransfer(getExpansion(), false);
        return transfer->determinant();
    }

    /// Get solver expansion
    virtual Expansion& getExpansion() = 0;

#ifndef NDEBUG
    void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH);
#endif
    
  protected:

    /**
     * Return number of determined modes
     */
    virtual size_t nummodes() const = 0;

    /**
     * Get refractive index after expansion
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<const Tensor3<dcomplex>> getRefractiveIndexProfile(const shared_ptr<const MeshD<BaseT::SpaceType::DIM>>& dst_mesh,
                                                                  InterpolationMethod interp=INTERPOLATION_DEFAULT);

    /**
     * Compute electric field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    virtual LazyData<Vec<3,dcomplex>> getE(size_t num, shared_ptr<const MeshD<BaseT::SpaceType::DIM>> dst_mesh, InterpolationMethod method) = 0;

    /**
     * Compute magnetic field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    virtual LazyData<Vec<3,dcomplex>> getH(size_t num, shared_ptr<const MeshD<BaseT::SpaceType::DIM>> dst_mesh, InterpolationMethod method) = 0;

    /**
     * Compute normalized electric field intensity 1/2 E conj(E) / P
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    virtual LazyData<double> getMagnitude(size_t num, shared_ptr<const MeshD<BaseT::SpaceType::DIM>> dst_mesh, InterpolationMethod method) = 0;

};


}}} // namespace

#endif // PLASK__SOLVER_SLAB_SLABBASE_H

