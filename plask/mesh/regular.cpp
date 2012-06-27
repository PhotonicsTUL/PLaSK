#include "regular.h"

#include "rectangular2d_impl.h"
#include "rectangular3d_impl.h"

namespace plask {

template class RectangularMesh2D<RegularMesh1D>;
template class RectangularMesh3D<RegularMesh1D>;

} // namespace plask
