#ifndef PLASK__AlN_Si_H
#define PLASK__AlN_Si_H

/** @file
This file contains Si-doped AlN
*/

#include "plask/material/material.hpp"
#include "AlN.hpp"

namespace plask { namespace materials {

/**
 * Represent Si-doped AlN, its physical properties.
 */
struct AlN_Si: public AlN {

    static constexpr const char* NAME = "AlN:Si";

    AlN_Si(double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO change to cm^(-3)
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

#endif	//PLASK__AlN_Si_H
