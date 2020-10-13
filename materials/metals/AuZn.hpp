#ifndef PLASK__AuZn_H
#define PLASK__AuZn_H

/** @file
This file contains AuZn
*/

#include "metal.hpp"

namespace plask { namespace materials {

/**
 * Represent AuZn, its physical properties.
 */
struct AuZn: public Metal {

    static constexpr const char* NAME = "AuZn";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double t) const override;
//     virtual double nr(double lam, double T, double n=0.) const override;
//     virtual double absp(double lam, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif    //PLASK__AuZn_H
