/**
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__SOLVER_GAIN_FERMI_H
#define PLASK__SOLVER_GAIN_FERMI_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace fermi {

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename GeometryType>
struct FermiGainSolver: public SolverOver<GeometryType>
{
    /// Structure containing information about each active region
    struct ActiveRegionInfo {
        shared_ptr<StackContainer<2>> layers;   ///< Stack containing all layers in the active region
        Vec<2> origin;                          ///< Location of the active region stack origin
        std::vector<bool> isQW;                 ///< Flags indicating which layers are quantum wells
        shared_ptr<Material> bottom;            ///< Material below the active region
        shared_ptr<Material> top;               ///< Material above the active region
        ActiveRegionInfo(Vec<2> origin): layers(make_shared<StackContainer<2>>()), origin(origin) {}
        /// \return material of \p n-th layer
        shared_ptr<Material> getLayerMaterial(size_t n) const {
            auto block = static_pointer_cast<Block<2>>(static_pointer_cast<Translation<2>>(layers->getChildNo(n))->getChild());
            return block->material;
        }
        /// \return translated block of \p n-th layer
        Box2D getLayerBox(size_t n) const {
            return static_pointer_cast<GeometryObjectD<2>>(layers->getChildNo(n))->getBoundingBox() + origin;
        }
        /// \return bounding box of the whole active region
        Box2D getBoundingBox() const {
            return layers->getBoundingBox() + origin;
        }
    };

    std::vector<ActiveRegionInfo> regions;  ///< List of active regions

    /**
     * Our own implementation of receivers.
     * As this solver does not have explicit compute function, we must trigger outGain change
     * when any input changes. Receivers of this class do it automatically.
     */
    template <typename Property>
    struct MyReceiverFor: ReceiverFor<Property, GeometryType>
    {
        FermiGainSolver<GeometryType>* parent;

        MyReceiverFor(FermiGainSolver<GeometryType>* parent): parent(parent) {}

        template <typename T> MyReceiverFor& operator=(const T& rhs) {
            ReceiverFor<Property, GeometryType>::operator=(rhs); return *this;
        }

        template <typename T> MyReceiverFor& operator=(T&& rhs) {
            ReceiverFor<Property, GeometryType>::operator=(std::forward<T>(rhs)); return *this;
        }

        virtual void onChange() {
            parent->outGain.fireChanged();  // the input changed, so we inform the world that everybody should get the new gain
        }
    };

    /// Receiver for temperature.
    MyReceiverFor<Temperature> inTemperature;

    /// Receiver for carriers concentration in the active region
    MyReceiverFor<CarriersConcentration> inCarriersConcentration;

    /// Provider for gain distribution
    typename ProviderFor<Gain, GeometryType>::Delegate outGain;

    FermiGainSolver(const std::string& name="");

    virtual std::string getClassName() const;

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);

    /// Main computation function TODO: is this necessary in this solver?
    void compute();

  protected:

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the gain
    virtual void onInvalidate();

    /**
     * Detect active regions.
     * Store information about them in the \p regions field.
     */
    void detectActiveRegions();

    /**
     * Compute gain at given point. This method is called multiple times when the gain is being provided
     * \param point point to compute gain at
     * \param wavelenght wavelenght to compute gain for
     * \return computed gain
     */
    double computeGain(const Vec<2>& point, double wavelenght);

    /**
     * Method computing the gain on the mesh (called by gain provider)
     * \param dst_mesh destination mesh
     * \param wavelenght wavelenght to compute gain for
     * \return gain distribution
     */
    const DataVector<const double> getGain(const MeshD<2>& dst_mesh, double wavelenght, InterpolationMethod=DEFAULT_INTERPOLATION);

};


}}} // namespace

#endif

