#include "python_globals.h"
#include "python_property.h"

#include <plask/properties/thermal.h>
#include <plask/properties/electrical.h>
#include <plask/properties/gain.h>
#include <plask/properties/optical.h>

namespace plask { namespace python {

/**
 * Register standard properties to Python.
 *
 * Add new properties here
 */
void register_standard_properties()
{
    // Thermal
    registerProperty<Temperature>();
    registerProperty<HeatFlux2D>();
    registerProperty<HeatFlux3D>();
    registerProperty<HeatDensity>();

    // Electrical
    registerProperty<Potential>();
    registerProperty<CurrentDensity2D>();
    registerProperty<CurrentDensity3D>();
    registerProperty<CarriersConcentration>();
    registerProperty<ElectronsConcentration>();
    registerProperty<HolesConcentration>();

    // Gain
    registerProperty<Gain>();

    // Optical
    registerProperty<OpticalIntensity>();
    registerProperty<Wavelength>();
    registerProperty<ModalLoss>();
    registerProperty<PropagationConstant>();
    registerProperty<EffectiveIndex>();
}

}} // namespace plask>();