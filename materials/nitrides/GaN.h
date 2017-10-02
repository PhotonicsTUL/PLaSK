#ifndef PLASK__GaN_H
#define PLASK__GaN_H

/** @file
This file contains undoped GaN
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent undoped GaN, its physical properties.
 */
struct GaN: public Semiconductor {

    static constexpr const char* NAME = "GaN";

    std::string name() const override;
    Tensor2<double> cond(double T) const override;
    Tensor2<double> thermk(double T, double t) const override;
    double nr(double lam, double T, double n=0.) const override;
    double absp(double lam, double T) const override;
    double lattC(double T, char x) const override;
    double Eg(double T, double e=0, char point='*') const override;
    //double VB(double T, double e, char point, char hole) const override;
    double Dso(double T, double e) const override;
    Tensor2<double> Me(double T, double e, char point) const override;
    Tensor2<double> Mhh(double T, double e) const override;
    Tensor2<double> Mlh(double T, double e) const override;
    double CB(double T, double e, char point) const override;
    double VB(double T, double e, char point, char hole) const override;
    virtual ConductivityType condtype() const override;

protected:
    bool isEqual(const Material& other) const override;

};

/**
 * Represent undoped bulk (substrate) GaN, its physical properties.
 */
struct GaN_bulk: public GaN {

    static constexpr const char* NAME = "GaN_bulk";

    virtual Tensor2<double> thermk(double T, double t) const override;

};

}} // namespace plask::materials

#endif	//PLASK__GaN_H
