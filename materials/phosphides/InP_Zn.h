#ifndef PLASK__InP_Zn_H
#define PLASK__InP_Zn_H

/** @file
This file contains Zn-doped InP
*/

#include <plask/material/material.h>
#include "InP.h"

namespace plask { namespace materials {

/**
 * Represent Zn-doped InP, its physical properties.
 */
struct InP_Zn: public InP {

    static constexpr const char* NAME = "InP:Zn";

    InP_Zn(DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
    virtual double Nf(double T) const; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

private:
    double NA,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__InP_Zn_H
