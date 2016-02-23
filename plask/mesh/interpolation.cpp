#include "interpolation.h"

namespace plask {

const char* interpolationMethodNames[__ILLEGAL_INTERPOLATION_METHOD__+1] = {
    "DEFAULT",
    "NEAREST",
    "LINEAR",
    "SPLINE",
    "SMOOTH_SPLINE",
    "PERIODIC_SPLINE",
    "FOURIER",
    // ...attach new interpolation algorithm names here...
    "ILLEGAL"
};

}   // namespace plask
