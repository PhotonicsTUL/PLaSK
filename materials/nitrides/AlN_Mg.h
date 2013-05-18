#ifndef PLASK__AlN_Mg_H
#define PLASK__ALN_Mg_H

/** @file
This file contains Mg-doped AlN
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
    virtual Tensor2<double> mob(double T) const;
    virtual double Nf(double T) const;
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

private:
    double NA,
           Nf_RT,
           mob_RT,
           cond_RT;

};


} // namespace plask

#endif	//PLASK__AlN_Mg_H
