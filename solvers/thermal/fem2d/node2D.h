#ifndef PLASK__MODULE_THERMAL_NODE2D_H
#define PLASK__MODULE_THERMAL_NODE2D_H

namespace plask { namespace solvers { namespace thermal {

class Node2D // 2D node in finite element method
{
public:
    Node2D(int iNo, const double* ipX, const double* ipY, double iT, bool ifTConst = false); // constructor (set no, coordinates pointers, temperature, temperature flag)

    // getters
    int getNo() const; // return node number
    double getX() const; // return x-coordinate
    double getY() const; // return y-coordinate
    double getT() const; // return temperature
    bool ifTConst() const; // true if temperature is constant
    const double* getXPtr() const; // return pointer to x from x-axis
    const double* getYPtr() const; // return pointer to y from y-axis

    // setters
    void setT(double iT); // set temperature

protected:
    int mNo; // number
    const double *mpX, *mpY; // coordinates pointers
    double mT; // temperature
    bool mfTConst; // true if temperature is constant
};

}}} // namespace plask::solvers::thermal
#endif
