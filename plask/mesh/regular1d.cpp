#include "regular1d.h"

namespace plask {

void RegularAxis::writeXML(XMLElement &object) const {
    object.attr("type", "regular").attr("start", first()).attr("stop", last()).attr("num", size());
}

bool RegularAxis::isIncreasing() const
{
    return step() >= 0;
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
