#ifndef PLASK__AlN_Mg_H
#define PLASK__AlN_Mg_H

/** @file
This file contains Mg-doped AlN
*/

#include "plask/material/material.hpp"
#include "AlN.hpp"

namespace plask { namespace materials {

/**
 * Represent Mg-doped AlN, its physical properties.
 */
struct AlN_Mg: public AlN {

    static constexpr const char* NAME = "AlN:Mg";

    AlN_Mg(double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override;
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

private:
    double NA,
           Nf_RT,
           mob_RT,
           cond_RT;

};


}} // namespace plask::materials

#endif	//PLASK__AlN_Mg_H
