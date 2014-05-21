#include "regular1d.h"

#include "../log/log.h"

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
    beforeCalcMidpointMesh();
    auto result = make_shared<RegularMesh1D>(*this);
    //if (this->points_count > 0) { //beforeCalcMidpointMesh() throws exception if this is not true
        --result->points_count;
        result->lo += _step * 0.5;
    //}
    return result;
    //return make_shared<RegularMesh1D>(this->first() + this->step() * 0.5, this->last() - this->step() * 0.5, this->points_count - 1);
}

shared_ptr<RegularMesh1D> readRegularMeshAxis(XMLReader& reader) {
    double start = reader.requireAttribute<double>("start");
    double stop = reader.requireAttribute<double>("stop");
    size_t count = reader.requireAttribute<size_t>("num");
    reader.requireTagEnd();
    return make_shared<RegularMesh1D>(start, stop, count);
}

shared_ptr<RegularMesh1D> readRegularMesh1D(XMLReader& reader) {
    reader.requireTag("axis");
    auto result = readRegularMeshAxis(reader);
    reader.requireTagEnd();
    return result;
}

RegisterMeshReader regularmesh_reader("regular", readRegularMesh1D);


shared_ptr<RegularMesh1D> readRegularMesh1D_obsolete(XMLReader& reader) {
    writelog(LOG_WARNING, "Mesh type \"%1%\" is obsolete (will not work in future versions of PLaSK), use \"regular\" instead.", reader.requireAttribute("type"));
    return readRegularMesh1D(reader);
}
RegisterMeshReader regularmesh1d_reader("regular1d", readRegularMesh1D_obsolete);

}
