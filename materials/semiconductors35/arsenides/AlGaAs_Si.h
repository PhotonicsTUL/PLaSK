#ifndef PLASK__AlGaAs_Si_H
#define PLASK__AlGaAs_Si_H

/** @file
This file contains Si-doped AlGaAs
*/

#include <plask/material/material.h>
#include "AlGaAs.h"
#include "GaAs_Si.h"
#include "AlAs_Si.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped AlGaAs, its physical properties.
 */
struct AlGaAs_Si: public AlGaAs {

    static constexpr const char* NAME = "AlGaAs:Si";

    AlGaAs_Si(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;
    virtual double EactA(double T) const override;
    virtual double EactD(double T) const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    virtual double Na() const override;
    virtual double Nd() const override;
    virtual double Dop() const;
    virtual Tensor2<double> cond(double T) const override;
    virtual ConductivityType condtype() const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

    GaAs_Si mGaAs_Si;
    AlAs_Si mAlAs_Si;
};

}} // namespace plask::materials

#endif	//PLASK__AlGaAs_Si_H
