#include "regular1d.h"

namespace plask {

void RegularAxis::writeXML(XMLElement &object) const {
    object.attr("type", "regular").attr("start", first()).attr("stop", last()).attr("num", size());
}

bool RegularAxis::isIncreasing() const
{
    return step() >= 0;
}

shared_ptr<RectangularMesh<1> > RegularAxis::getMidpointsMesh() const
{
    if (this->points_count == 0) return make_shared<RegularMesh1D>(*this);
    return make_shared<RegularMesh1D>(this->first() + this->step() * 0.5, this->last() - this->step() * 0.5, this->points_count - 1);
}

shared_ptr<RegularMesh1D> readRegularMesh1D(XMLReader& reader) {
    double start = reader.requireAttribute<double>("start");
    double stop = reader.requireAttribute<double>("stop");
    size_t count = reader.requireAttribute<size_t>("num");
    reader.requireTagEnd();
    return make_shared<RegularMesh1D>(start, stop, count);
}

RegisterMeshReader regularmesh1d_reader("regular", readRegularMesh1D);

}
