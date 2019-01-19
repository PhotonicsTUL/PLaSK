#ifndef PLASK__Cu_H
#define PLASK__Cu_H

/** @file
This file contains Cu
*/

#include "metal.h"

namespace plask { namespace materials {

/**
 * Represent Cu, its physical properties.
 */
struct Cu: public LorentzDrudeMetal {

    Cu();

    static constexpr const char* NAME = "Cu";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double t) const override;

protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif    //PLASK__Cu_H
