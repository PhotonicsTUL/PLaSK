#ifndef PLASK__AlGaAs_C_H
#define PLASK__AlGaAs_C_H

/** @file
This file includes C-doped AlGaAs
*/

#include <plask/material/material.h>
#include "AlGaAs.h"
#include "GaAs_C.h"
#include "AlAs_C.h"

namespace plask {

/**
 * Represent C-doped AlGaAs, its physical properties.
 */
struct AlGaAs_C: public AlGaAs {

    static constexpr const char* NAME = "AlGaAs:C";

    AlGaAs_C(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const;    
    virtual std::string str() const;
    virtual std::pair<double,double> mob(double T) const;
    virtual double Nf(double T) const;
    virtual double Dop() const;
    virtual std::pair<double,double> cond(double T) const;

private:
    double NA,
           Nf_RT;

    GaAs_C mGaAs_C;
    AlAs_C mAlAs_C;
};

} // namespace plask

#endif	//PLASK__AlGaAs_C_H
