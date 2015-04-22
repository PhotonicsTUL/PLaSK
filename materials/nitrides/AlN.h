#ifndef PLASK__AlN_H
#define PLASK__AlN_H

/** @file
This file contains undoped AlN
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent undoped AlN, its physical properties.
 */
struct AlN: public Semiconductor {

    static constexpr const char* NAME = "AlN";

	std::string name() const override;
    Tensor2<double> thermk(double T, double t) const override;
    double nr(double wl, double T, double n=0.) const override;
    double absp(double wl, double T) const override;
    double lattC(double T, char x) const override;
    double Eg(double T, double e, char point) const override;
    double VB(double T, double e, char point, char hole) const override;
    double Dso(double T, double e) const override;
    Tensor2<double> Me(double T, double e, char point) const override;
    Tensor2<double> Mhh(double T, double e) const override;
    Tensor2<double> Mlh(double T, double e) const override;
/*TODO
    double Mhh(double T, double e, char point) const override;
    double Mhh_l(double T, char point) const override;
    double Mhh_v(double T, char point) const override;
    double Mlh(double T, double e, char point) const override;
    double Mlh_l(double T, char point) const override;
    double Mlh_v(double T, char point) const override;
*/

protected:
    virtual bool isEqual(const Material& other) const override;

};


}} // namespace plask::materials

#endif	//PLASK__AlN_H
