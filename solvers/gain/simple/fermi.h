/**
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__SOLVER_GAIN_FERMI_H
#define PLASK__SOLVER_GAIN_FERMI_H

#include <plask/plask.hpp>
#include "gainQW.h"

namespace plask { namespace solvers { namespace fermi {

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename GeometryType>
struct FermiGainSolver: public SolverOver<GeometryType>
{
    /// Structure containing information about each active region
    struct ActiveRegionInfo
    {
        shared_ptr<StackContainer<2>> layers;   ///< Stack containing all layers in the active region
        Vec<2> origin;                          ///< Location of the active region stack origin
        ActiveRegionInfo(Vec<2> origin): layers(make_shared<StackContainer<2>>()), origin(origin) {}

        /// \return number of layers in the active region with surrounding barriers
        size_t size() const
        {
            return layers->getChildrenCount();
        }

        /// \return material of \p n-th layer
        shared_ptr<Material> getLayerMaterial(size_t n) const
        {
            auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layers->getChildNo(n).get())->getChild().get());
            return block->material;
        }

        /// \return translated bounding box of \p n-th layer
        Box2D getLayerBox(size_t n) const
        {
            return static_cast<GeometryObjectD<2>*>(layers->getChildNo(n).get())->getBoundingBox() + origin;
        }

        /// \return \p true if given layer is quantum well
        bool isQW(size_t n) const
        {
            return static_cast<Translation<2>*>(layers->getChildNo(n).get())->getChild()->hasRole("QW");
        }

        /// \return bounding box of the whole active region
        Box2D getBoundingBox() const
            {
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

        MyReceiverFor(FermiGainSolver<GeometryType>* par): parent(par) {}

        template <typename T> MyReceiverFor& operator=(const T& rhs)
        {
            ReceiverFor<Property, GeometryType>::operator=(rhs); return *this;
        }

        template <typename T> MyReceiverFor& operator=(T&& rhs)
        {
            ReceiverFor<Property, GeometryType>::operator=(std::forward<T>(rhs)); return *this;
        }

        virtual void onChange()
        {
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
    void determineLevels(double T, double n);

  protected:

    /// External gain module (Michal Wasiak)
    QW::gain gainModule;

//    double lambda_start;
//    double lambda_stop;
//    double lambda;

    void setParameters(double wavelength, double T, double n, ActiveRegionInfo active);
    double nm_to_eV(double wavelength);

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
     * Compute width of the box
     * \param materialBox box to compute the width of
     * \return width of the box
     */
    double determineBoxWidth(plask::Box2D materialBox)
    {
        return  materialBox.upper[1] - materialBox.lower[1];
    }

    /**
     * Method computing the gain on the mesh (called by gain provider)
     * \param dst_mesh destination mesh
     * \param wavelenght wavelenght to compute gain for
     * \return gain distribution
     */
    const DataVector<double> getGain(const MeshD<2>& dst_mesh, double wavelength, InterpolationMethod=DEFAULT_INTERPOLATION);

};


}}} // namespace

#endif

