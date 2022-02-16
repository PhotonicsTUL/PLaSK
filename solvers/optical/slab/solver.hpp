#ifndef PLASK__SOLVER_SLAB_SOLVER_H
#define PLASK__SOLVER_SLAB_SOLVER_H

#include <plask/plask.hpp>
#include "solverbase.hpp"
#include "expansion.hpp"
#include "rootdigger.hpp"
#include "transfer.hpp"

#undef interface

namespace plask { namespace optical { namespace slab {

/**
 * Base class for all slab solvers
 */
template <typename BaseT> class PLASK_SOLVER_API SlabSolver : public BaseT, public SlabBase {
    /// Reset structure if input is changed
    void onInputChanged(ReceiverBase&, ReceiverBase::ChangeReason) {
        this->clearModes();
        this->recompute_integrals = true;
    }

    /// Reset structure if input is changed
    void onGainChanged(ReceiverBase&, ReceiverBase::ChangeReason reason) {
        if (reason == ReceiverBase::ChangeReason::REASON_VALUE) {
            this->clearModes();
            this->recompute_gain_integrals = true;
        } else {
            this->invalidate();
        }
    }

  protected:
    void onInitialize() override { setupLayers(); }

    void onGeometryChange(const Geometry::Event& evt) override {
        BaseT::onGeometryChange(evt);
        if (this->geometry) {
            if (evt.flags() == 0) {
                auto objects = this->geometry->getObjectsWithRole("interface");
                if (objects.size() > 1) {
                    Solver::writelog(LOG_WARNING, "More than one object with 'interface' role: interface not set");
                } else if (objects.size() == 1) {
                    setInterfaceOn(objects[0]);
                }
            }
        } else {
            vbounds->clear();
        }
    }

    /// Parse common configuration from XML file
    void parseCommonSlabConfiguration(XMLReader& reader, Manager& manager);

    /// Compute layer boundaries and detect layer sets
    void setupLayers();

  public:
    /// Smoothing coefficient
    double smooth;

    /// Receiver for the temperature
    ReceiverFor<Temperature, typename BaseT::SpaceType> inTemperature;

    /// Receiver for the gain
    ReceiverFor<Gain, typename BaseT::SpaceType> inGain;

    /// Receiver for carriers concentration
    ReceiverFor<CarriersConcentration, typename BaseT::SpaceType> inCarriersConcentration;

    /// Provider of the refractive index
    typename ProviderFor<RefractiveIndex, typename BaseT::SpaceType>::Delegate outRefractiveIndex;

    /// Provider for computed resonant wavelength
    typename ProviderFor<ModeWavelength>::Delegate outWavelength;

    /// Provider of the optical field intensity
    typename ProviderFor<ModeLightMagnitude, typename BaseT::SpaceType>::Delegate outLightMagnitude;

    /// Provider of the optical electric field
    typename ProviderFor<ModeLightE, typename BaseT::SpaceType>::Delegate outLightE;

    /// Provider of the optical magnetic field
    typename ProviderFor<ModeLightH, typename BaseT::SpaceType>::Delegate outLightH;

    /// Provider of the optical electric field propagating upwards
    typename ProviderFor<ModeLightE, typename BaseT::SpaceType>::Delegate outUpwardsLightE;

    /// Provider of the optical magnetic field propagating upwards
    typename ProviderFor<ModeLightH, typename BaseT::SpaceType>::Delegate outUpwardsLightH;

    /// Provider of the optical electric field propagating downwards
    typename ProviderFor<ModeLightE, typename BaseT::SpaceType>::Delegate outDownwardsLightE;

    /// Provider of the optical magnetic field propagating downwards
    typename ProviderFor<ModeLightH, typename BaseT::SpaceType>::Delegate outDownwardsLightH;

    SlabSolver(const std::string& name = "");

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

    /// Getter for max_temp_diff
    double getMaxTempDiff() const { return max_temp_diff; }

    /// Setter for max_temp_diff
    void setMaxTempDiff(double value) {
        bool changed = max_temp_diff != value;
        max_temp_diff = value;
        if (changed) this->invalidate();
    }

    /// Getter for temp_dist
    double getTempDist() const { return temp_dist; }

    /// Setter for temp_dist
    void setTempDist(double value) {
        bool changed = temp_dist != value;
        temp_dist = value;
        if (changed) this->invalidate();
    }

    /// Getter for temp_dist
    double getTempLayer() const { return temp_layer; }

    /// Setter for temp_dist
    void setTempLayer(double value) {
        bool changed = temp_layer != value;
        temp_layer = value;
        if (changed) this->invalidate();
    }

    std::string getId() const override { return Solver::getId(); }

    bool initCalculation() override { return Solver::initCalculation(); }

    /**
     * Get the position of the matching interface.
     * \return index of the vertical mesh, where interface is set
     */
    inline size_t getInterface() {
        Solver::initCalculation();
        return interface;
    }

    /**
     * Set the position of the matching interface.
     * \param pos vertical position close to the point where interface will be set
     */
    inline void setInterfaceAt(double pos) {
        if (pos != interface_position) {
            this->invalidate();
            interface_position = pos;
            Solver::writelog(LOG_DEBUG, "Setting interface at position {:g}um", interface_position);
        }
    }

    /**
     * Set the position of the matching interface at the bottom of the provided geometry object
     * \param object where the interface should  be set on
     * \param path path specifying object in the geometry
     */
    void setInterfaceOn(const shared_ptr<const GeometryObject>& object, const PathHints* path = nullptr) {
        auto boxes = this->geometry->getObjectBoundingBoxes(object, path);
        if (boxes.size() != 1) throw NotUniqueObjectException();
        if (interface_position != boxes[0].lower.vert()) {
            this->invalidate();
            interface_position = boxes[0].lower.vert();
            Solver::writelog(LOG_DEBUG, "Setting interface on an object at position {:g}um", interface_position);
        }
    }

    /**
     * Set the position of the matching interface at the bottom of the provided geometry object
     * \param object where the interface should  be set on
     * \param path path specifying object in the geometry
     */
    void setInterfaceOn(const shared_ptr<const GeometryObject>& object, const PathHints& path) { setInterfaceOn(object, &path); }

    /// Get discontinuity matrix determinant for the current parameters
    dcomplex getDeterminant() {
        initCalculation();
        ensureInterface();
        if (!transfer) initTransfer(getExpansion(), false);
        return transfer->determinant();
    }

    void prepareExpansionIntegrals(Expansion* expansion, const shared_ptr<MeshD<BaseT::SpaceType::DIM>>& mesh,
                                   double lam, double glam) {
        expansion->temperature = inTemperature(mesh);
        expansion->gain_connected = inGain.hasProvider();
        if (expansion->gain_connected) {
            if (isnan(glam)) glam = lam;
            expansion->gain = inGain(mesh, glam);
        }
        expansion->carriers = inCarriersConcentration.hasProvider() ? inCarriersConcentration(CarriersConcentration::MAJORITY, mesh)
                                                                    : LazyData<double>(mesh->size(), 0.);
    }


  protected:
    /**
     * Return number of determined modes
     */
    virtual size_t nummodes() const = 0;

    /**
     * Apply mode number n
     * \return Mode power
     */
    virtual double applyMode(size_t n) = 0;

    /**
     * Get refractive index after expansion
     * \param dst_mesh target mesh
     * \param method interpolation method
     */
    DataVector<const Tensor3<dcomplex>> getRefractiveIndexProfile(const shared_ptr<const MeshD<BaseT::SpaceType::DIM>>& dst_mesh,
                                                                  InterpolationMethod interp = INTERPOLATION_DEFAULT);

    /**
     * Compute electric field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    template <PropagationDirection part = PROPAGATION_TOTAL>
    LazyData<Vec<3, dcomplex>> getLightE(size_t num,
                                         shared_ptr<const MeshD<BaseT::SpaceType::DIM>> dst_mesh,
                                         InterpolationMethod method);

    /**
     * Compute magnetic field
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    template <PropagationDirection part = PROPAGATION_TOTAL>
    LazyData<Vec<3, dcomplex>> getLightH(size_t num,
                                         shared_ptr<const MeshD<BaseT::SpaceType::DIM>> dst_mesh,
                                         InterpolationMethod method);

    /**
     * Compute normalized electric field intensity 1/2 E conj(E) / P
     * \param num mode number
     * \param dst_mesh destination mesh
     * \param method interpolation method
     */
    LazyData<double> getLightMagnitude(size_t num,
                                       shared_ptr<const MeshD<BaseT::SpaceType::DIM>> dst_mesh,
                                       InterpolationMethod method);

    /**
     * Return mode wavelength
     * \param n mode number
     */
    virtual double getWavelength(size_t n) = 0;
};

}}}  // namespace plask::optical::slab

#endif  // PLASK__SOLVER_SLAB_SOLVER_H
