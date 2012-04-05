#include "calculation_space.h"

namespace plask {

shared_ptr< GeometryElementD<2> > Space2DCartesian::getChild() const {
    return extrusion->getChild();
}

}   // namespace plask
