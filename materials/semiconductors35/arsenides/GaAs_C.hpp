#ifndef PLASK__GaAs_C_H
#define PLASK__GaAs_C_H

/** @file
This file contains C-doped GaAs
*/

#include "plask/material/material.hpp"
#include "GaAs.hpp"

namespace plask { namespace materials {

/**
 * Represent C-doped GaAs, its physical properties.
 */
struct GaAs_C: public GaAs {

    static constexpr const char* NAME = "GaAs:C";

    GaAs_C(double Val);
    std::string name() const override;
    std::string str() const override;
    double EactA(double T) const override;
    double EactD(double T) const override; // will be removed
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
    double NA,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__GaAs_C_H
