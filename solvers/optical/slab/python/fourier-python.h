#ifndef PLASK__SOLVER__OPTICAL__SLAB_FOURIER_PYTHON_H
#define PLASK__SOLVER__OPTICAL__SLAB_FOURIER_PYTHON_H

#include <cmath>
#include <plask/python.hpp>
#include <util/ufunc.h>
using namespace plask::python;

#include "../expansion.h"
using namespace plask::solvers::slab;

namespace plask { namespace solvers { namespace slab { namespace python {
    
template <typename SolverT>
shared_ptr<typename SolverT::Reflected> FourierSolver_getReflected(SolverT* parent,
                                                                  double wavelength,
                                                                  Expansion::Component polarization,
                                                                  Transfer::IncidentDirection side)
{
    return plask::make_shared<typename SolverT::Reflected>(parent, wavelength, polarization, side);
}


    
}}}} // # namespace plask::solvers::slab::python

#endif // PLASK__SOLVER__OPTICAL__SLAB_FOURIER_PYTHON_H
