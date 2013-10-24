#ifndef PLASK__AlAsSb_Te_H
#define PLASK__AlAsSb_Te_H

/** @file
This file contains Te-doped AlAsSb
*/

#include <plask/material/material.h>
#include "AlAsSb.h"
//#include "AlAs_Te.h"
//#include "AlSb_Te.h"

namespace plask { namespace materials {

/**
 * Represent Te-doped AlAsSb, its physical properties.
 */
struct AlAsSb_Te: public AlAsSb {

    static constexpr const char* NAME = "AlAsSb:Te";

    AlAsSb_Te(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> mob(double T) const;
    virtual double Nf(double T) const; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const;
    virtual double nr(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

private:
    double ND,
           Nf_RT,
           mob_RT;

    //AlAs_Te mAlAs_Te;
    //AlSb_Te mAlSb_Te;
};

}} // namespace plask::materials

#endif	//PLASK__AlAsSb_Te_H
