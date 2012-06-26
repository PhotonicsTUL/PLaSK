#include "regular.h"

#include "rectangular2d_impl.h"
#include "rectangular3d_impl.h"

namespace plask {

template class RectangularMesh2d<RegularMesh1d>;
template class RectangularMesh3d<RegularMesh1d>;

} // namespace plask
