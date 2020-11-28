#ifndef PLASK__AlInAs_Si_H
#define PLASK__AlInAs_Si_H

/** @file
This file contains Si-doped AlInAs
*/

#include "plask/material/material.hpp"
#include "AlInAs.hpp"
#include "AlAs_Si.hpp"
#include "InAs_Si.hpp"

namespace plask { namespace materials {

/**
 * Represent Si-doped AlInAs, its physical properties.
 */
struct AlInAs_Si: public AlInAs {

    static constexpr const char* NAME = "AlInAs:Si";

    AlInAs_Si(const Material::Composition& Comp, double Val);
    std::string name() const override;
    std::string str() const override;
    double doping() const override;
    ConductivityType condtype() const override;

protected:
    bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

    AlAs_Si mAlAs_Si;
    InAs_Si mInAs_Si;
};

}} // namespace plask::materials

#endif	//PLASK__AlInAs_Si_H