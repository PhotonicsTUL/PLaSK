#ifndef PLASK__InP_Zn_H
#define PLASK__InP_Zn_H

/** @file
This file contains Zn-doped InP
*/

#include "plask/material/material.hpp"
#include "InP.hpp"

namespace plask { namespace materials {

/**
 * Represent Zn-doped InP, its physical properties.
 */
struct InP_Zn: public InP {

    static constexpr const char* NAME = "InP:Zn";

    InP_Zn(double Val);
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

#endif	//PLASK__InP_Zn_H
