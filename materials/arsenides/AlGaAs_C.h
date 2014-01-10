#ifndef PLASK__AlGaAs_C_H
#define PLASK__AlGaAs_C_H

/** @file
This file contains C-doped AlGaAs
*/

#include <plask/material/material.h>
#include "AlGaAs.h"
#include "GaAs_C.h"
#include "AlAs_C.h"

namespace plask { namespace materials {

/**
 * Represent C-doped AlGaAs, its physical properties.
 */
struct AlGaAs_C: public AlGaAs {

    static constexpr const char* NAME = "AlGaAs:C";

    AlGaAs_C(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double NA,
           Nf_RT,
           mob_RT;

    GaAs_C mGaAs_C;
    AlAs_C mAlAs_C;
};

}} // namespace plask::materials

#endif	//PLASK__AlGaAs_C_H
