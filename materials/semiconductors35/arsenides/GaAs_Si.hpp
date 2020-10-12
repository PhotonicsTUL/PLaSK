#ifndef PLASK__GaAs_Si_H
#define PLASK__GaAs_Si_H

/** @file
This file contains Si-doped GaAs
*/

#include "plask/material/material.hpp"
#include "GaAs.hpp"

namespace plask { namespace materials {

/**
 * Represent Si-doped GaAs, its physical properties.
 */
struct GaAs_Si: public GaAs {

    static constexpr const char* NAME = "GaAs:Si";

    GaAs_Si(double Val);
    std::string name() const override;
    std::string str() const override;
    double EactA(double T) const override;
    double EactD(double T) const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    double Na() const override;
    double Nd() const override;
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__GaAs_Si_H
