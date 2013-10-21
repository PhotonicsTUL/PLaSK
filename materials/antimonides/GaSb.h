#ifndef PLASK__GaSb_H
#define PLASK__GaSb_H

/** @file
This file contains undoped GaSb
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent undoped GaSb, its physical properties.
 */
struct GaSb: public Semiconductor {

    static constexpr const char* NAME = "GaSb";

    virtual std::string name() const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, double e, char point) const;
    virtual double Dso(double T, double e) const;
    virtual Tensor2<double> Me(double T, double e, char point) const;
    virtual Tensor2<double> Mhh(double T, double e) const;
    virtual Tensor2<double> Mlh(double T, double e) const;
    virtual double CB(double T, double e, char point) const;
    virtual double VB(double T, double e, char point, char hole='H') const;
    virtual double ac(double T) const;
    virtual double av(double T) const;
    virtual double b(double T) const;
    virtual double d(double T) const;
    virtual double c11(double T) const;
    virtual double c12(double T) const;
    virtual double c44(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;

    //TODO qwak: commented out because there is no impl. in .cpp file
    //virtual double absp(double wl, double T) const;
    //virtual Tensor2<double> cond(double T) const;

protected:
    virtual bool isEqual(const Material& other) const;
};

}} // namespace plask::materials

#endif	//PLASK__GaSb_H
