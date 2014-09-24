#ifndef PLASK__AlInAs_Si_H
#define PLASK__AlInAs_Si_H

/** @file
This file contains Si-doped AlInAs
*/

#include <plask/material/material.h>
#include "AlInAs.h"
#include "AlAs_Si.h"
#include "InAs_Si.h"

namespace plask { namespace materials {

/**
 * Represent Si-doped AlInAs, its physical properties.
 */
struct AlInAs_Si: public AlInAs {

    static constexpr const char* NAME = "AlInAs:Si";

    AlInAs_Si(const Material::Composition& Comp, DopingAmountType Type, double Val);
    virtual std::string name() const override;
    virtual std::string str() const override;

protected:
    virtual bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

    AlAs_Si mAlAs_Si;
    InAs_Si mInAs_Si;
};

}} // namespace plask::materials

#endif	//PLASK__AlInAs_Si_H
