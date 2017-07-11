#ifndef PLASK__InGaN_H
#define PLASK__InGaN_H

/** @file
This file contains undoped InGaN
*/

#include <plask/material/material.h>
#include "GaN.h"
#include "InN.h"

namespace plask { namespace materials {

/**
 * Represent undoped InGaN, its physical properties.
 */
struct InGaN: public Semiconductor {

    static constexpr const char* NAME = "InGaN";

    InGaN(const Material::Composition& Comp);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> thermk(double T, double t) const override;
    double nr(double wl, double T, double n=0.) const override;
    double absp(double wl, double T) const override;
    double Eg(double T, double e, char point) const override;
    //double VB(double T, double e, char point, char hole) const override;
    double Dso(double T, double e) const override;
    Tensor2<double> Me(double T, double e, char point) const override;
    Tensor2<double> Mhh(double T, double e) const override;
    Tensor2<double> Mlh(double T, double e) const override;
    double CB(double T, double e, char point) const override;
    double VB(double T, double e, char point, char hole) const override;
    double lattC(double T, char x) const override;
    virtual ConductivityType condtype() const override;

protected:
    bool isEqual(const Material& other) const override;

protected:
    double In,
           Ga;

    GaN mGaN;
    InN mInN;

};


}} // namespace plask::materials

#endif	//PLASK__InGaN_H
