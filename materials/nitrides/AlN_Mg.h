#ifndef PLASK__AlN_Mg_H
#define PLASK__ALN_Mg_H

/** @file
This file includes Mg-doped AlN
*/

#include <plask/material/material.h>
#include "AlN.h"

namespace plask {

/**
 * Represent Mg-doped AlN, its physical properties.
 */
struct AlN_Mg: public AlN {

    static constexpr const char* NAME = "AlN:Mg";

    AlN_Mg(DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    //virtual std::pair<double,double> mob(double T) const;
    //virtual double Nf(double T) const; //TODO change to cm^(-3)
    //virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual double absp(double wl, double T) const;

private:
    double NA,
           Nf_RT,
           mob_RT,
           cond_RT;

};


} // namespace plask

#endif	//PLASK__AlN_Mg_H
