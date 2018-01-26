#ifndef PLASK__SOLVER__OPTICAL__SLAB_FOURIER_PYTHON_H
#define PLASK__SOLVER__OPTICAL__SLAB_FOURIER_PYTHON_H

#include <cmath>
#include <plask/python.hpp>
#include <plask/python_util/ufunc.h>
using namespace plask::python;

#include "../expansion.h"
using namespace plask::optical::slab;

namespace plask { namespace optical { namespace slab { namespace python {

template <typename SolverT>
shared_ptr<typename SolverT::Reflected> FourierSolver_getReflected(SolverT* parent,
                                                                  double wavelength,
                                                                  Expansion::Component polarization,
                                                                  Transfer::IncidentDirection side)
{
    return plask::make_shared<typename SolverT::Reflected>(parent, wavelength, polarization, side);
}

}}}} // # namespace plask::optical::slab::python

#endif // PLASK__SOLVER__OPTICAL__SLAB_FOURIER_PYTHON_H
