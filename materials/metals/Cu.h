#ifndef PLASK__Cu_H
#define PLASK__Cu_H

/** @file
This file contains Cu
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent Cu, its physical properties.
 */
struct Cu: public Metal {

    static constexpr const char* NAME = "Cu";

    virtual std::string name() const;
    virtual Tensor2<double> cond(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;
};


} // namespace plask

#endif	//PLASK__Cu_H
