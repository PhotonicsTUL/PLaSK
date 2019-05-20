#ifndef PLASK__AlGaN_H
#define PLASK__AlGaN_H

/** @file
This file contains undoped AlGaN
*/

#include <plask/material/material.h>
#include "GaN.h"
#include "AlN.h"

namespace plask { namespace materials {

/**
 * Represent undoped AlGaN, its physical properties.
 */
struct AlGaN: public Semiconductor {

    static constexpr const char* NAME = "AlGaN";

    AlGaN(const Material::Composition& Comp);
    std::string name() const override;
    std::string str() const override;
    Composition composition() const override;
    Tensor2<double> thermk(double T, double t) const override;
    double nr(double lam, double T, double n=0.) const override;
    double absp(double lam, double T) const override;
    double Eg(double T, double e, char point) const override;
    double VB(double T, double e, char point, char hole) const override;
    double Dso(double T, double e) const override;
    Tensor2<double> Me(double T, double e, char point) const override;
    Tensor2<double> Mhh(double T, double e) const override;
    Tensor2<double> Mlh(double T, double e) const override;
    double lattC(double T, char x) const override;
    ConductivityType condtype() const override;

protected:
    bool isEqual(const Material& other) const override;

protected:
    double Al,
           Ga;

    GaN mGaN;
    AlN mAlN;

};


}} // namespace plask::materials

#endif	//PLASK__AlGaN_H