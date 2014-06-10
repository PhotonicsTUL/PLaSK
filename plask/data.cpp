#include "data.h"

namespace plask {

template struct DataVector<double>;
template struct DataVector<const double>;
template struct DataVector<std::complex<double>>;
template struct DataVector<const std::complex<double>>;



}   // namespace plask


