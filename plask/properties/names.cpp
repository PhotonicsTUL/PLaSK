#include "thermal.h"
#include "electrical.h"
#include "gain.h"
#include "optical.h"

namespace plask {

constexpr const char* Temperature::NAME;                       constexpr const char* Temperature::UNIT;
constexpr const char* HeatFlux::NAME;                          constexpr const char* HeatFlux::UNIT;
constexpr const char* Heat::NAME;                              constexpr const char* Heat::UNIT;
constexpr const char* ThermalConductivity::NAME;               constexpr const char* ThermalConductivity::UNIT;

constexpr const char* Voltage::NAME;                           constexpr const char* Voltage::UNIT;
constexpr const char* Potential::NAME;                         constexpr const char* Potential::UNIT;
constexpr const char* CurrentDensity::NAME;                    constexpr const char* CurrentDensity::UNIT;
constexpr const char* CarriersConcentration::NAME;             constexpr const char* CarriersConcentration::UNIT;
constexpr const char* ElectronsConcentration::NAME;            constexpr const char* ElectronsConcentration::UNIT;
constexpr const char* HolesConcentration::NAME;                constexpr const char* HolesConcentration::UNIT;
constexpr const char* Conductivity::NAME;                      constexpr const char* Conductivity::UNIT;
constexpr const char* QuasiFermiEnergyLevelForElectrons::NAME; constexpr const char* QuasiFermiEnergyLevelForElectrons::UNIT;
constexpr const char* QuasiFermiEnergyLevelForHoles::NAME;     constexpr const char* QuasiFermiEnergyLevelForHoles::UNIT;
constexpr const char* ConductionBandEdge::NAME;                constexpr const char* ConductionBandEdge::UNIT;
constexpr const char* ValenceBandEdge::NAME;                   constexpr const char* ValenceBandEdge::UNIT;

constexpr const char* Gain::NAME;                              constexpr const char* Gain::UNIT;
constexpr const char* Luminescence::NAME;                      constexpr const char* Luminescence::UNIT;
constexpr const char* GainOverCarriersConcentration::NAME;     constexpr const char* GainOverCarriersConcentration::UNIT;

constexpr const char* RefractiveIndex::NAME;                   constexpr const char* RefractiveIndex::UNIT;
constexpr const char* LightMagnitude::NAME;                    constexpr const char* LightMagnitude::UNIT;
constexpr const char* LightE::NAME;                            constexpr const char* LightE::UNIT;
constexpr const char* LightH::NAME;                            constexpr const char* LightH::UNIT;
constexpr const char* Wavelength::NAME;                        constexpr const char* Wavelength::UNIT;
constexpr const char* ModalLoss::NAME;                         constexpr const char* ModalLoss::UNIT;
constexpr const char* PropagationConstant::NAME;               constexpr const char* PropagationConstant::UNIT;
constexpr const char* EffectiveIndex::NAME;                    constexpr const char* EffectiveIndex::UNIT;

}
