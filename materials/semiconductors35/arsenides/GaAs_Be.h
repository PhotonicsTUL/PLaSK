#ifndef PLASK__GaAs_Be_H
#define PLASK__GaAs_Be_H

/** @file
This file contains Be-doped GaAs
*/

#include <plask/material/material.h>
#include "GaAs.h"

namespace plask { namespace materials {

/**
 * Represent Be-doped GaAs, its physical properties.
 */
struct GaAs_Be: public GaAs {

    static constexpr const char* NAME = "GaAs:Be";

    GaAs_Be(double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

private:
    double NA,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__GaAs_Be_H
