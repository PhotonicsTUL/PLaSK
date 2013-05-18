#ifndef PLASK__GaInAs_Zn_H
#define PLASK__GaInAs_Zn_H

/** @file
This file contains Zn-doped GaInAs
*/

#include <plask/material/material.h>
#include "GaInAs.h"
//#include "GaAs_Zn.h"
//#include "InAs_Zn.h"

namespace plask {

/**
 * Represent Zn-doped GaInAs, its physical properties.
 */
struct GaInAs_Zn: public GaInAs {

    static constexpr const char* NAME = "GaInAs:Zn";

    GaInAs_Zn(const Material::Composition& Comp, DopingAmountType Type, double Val);
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

    //GaAs_Zn mGaAs_Zn;
    //InAs_Zn mInAs_Zn;
};

} // namespace plask

#endif	//PLASK__GaInAs_Zn_H
