#include "rectangular1d.h"
#include "rectilinear1d.h"

#include "../utils/stl.h"

namespace plask {

shared_ptr<RectangularMesh<1> > RectangularMesh<1>::clone() const {
    //return make_shared<MidpointsMesh>(wrapped);
    return make_shared<RectilinearAxis>(*this);
}

std::size_t RectangularMesh<1>::findIndex(double to_find) const {
    return std::lower_bound(begin(), end(), to_find).index;
}

std::size_t RectangularMesh<1>::findNearestIndex(double to_find) const {
    return find_nearest_binary(begin(), end(), to_find).index;
}

shared_ptr<RectangularMesh<1> > RectangularMesh<1>::getMidpointsMesh() const {
    return make_shared<MidpointsMesh>(*this)->clone();  //TODO clone() to generate rectangular mesh
}


/*shared_ptr<RectangularMesh<1> > MidpointsMesh::getWrapped() const {
    return wrapped;
}

void MidpointsMesh::setWrapped(shared_ptr<RectangularMesh<1> > value) {
    wrapped = value;
}

shared_ptr<RectangularMesh<1> > MidpointsMesh::clone() const {
    return make_shared<MidpointMesh>(wrapped->clone());
}

std::size_t MidpointsMesh::size() const {
    if (!wrapped) return 0;
    std::size_t wrapped_size = wrapped->size();
    return wrapped_size ? wrapped_size - 1 : 0;
}

double MidpointsMesh::at(std::size_t index) const {
    return (wrapped->at(index) + wrapped->at(index+1)) * 0.5;
}*/

std::size_t MidpointsMesh::size() const {
    //if (!wrapped) return 0;
    std::size_t wrapped_size = wrapped.size();
    return wrapped_size ? wrapped_size - 1 : 0;
}

double MidpointsMesh::at(std::size_t index) const {
    return (wrapped.at(index) + wrapped.at(index+1)) * 0.5;
}

bool MidpointsMesh::isIncreasing() const {
    return wrapped.isIncreasing();
}







}



/*template <>
void RectangularMesh<1,RectilinearAxis>::writeXML(XMLElement& object) const {
    object.attr("type", "rectilinear1d");
    auto tag = object.addTag("axis");
    tag.indent();
    for (auto x: axis) {
        tag.writeText(x);
        tag.writeText(" ");
    }
    tag.writeText("\n");
}


template <>
void RectangularMesh<1,RegularAxis>::writeXML(XMLElement& object) const {
    object.attr("type", "regular1d");
    object.addTag("axis").attr("start", axis.first()).attr("stop", axis.last()).attr("num", axis.size());
}


static shared_ptr<Mesh> readRectilinearMesh1D(XMLReader& reader)
{
    RectilinearAxis axis;

    reader.requireTag();
    std::string node = reader.getNodeName();

    if (node != "axis") throw XMLUnexpectedElementException(reader, "<axis>");

    if (reader.hasAttribute("start")) {
        double start = reader.requireAttribute<double>("start");
        double stop = reader.requireAttribute<double>("stop");
        size_t count = reader.requireAttribute<size_t>("num");
        axis.addPointsLinear(start, stop, count);
        reader.requireTagEnd();
    } else {
        std::string data = reader.requireTextInCurrentTag();
        for (auto point: boost::tokenizer<>(data)) {
            try {
                double val = boost::lexical_cast<double>(point);
                axis.addPoint(val);
            } catch (boost::bad_lexical_cast) {
                throw XMLException(reader, format("Value '%1%' cannot be converted to float", point));
            }
        }
    }
    reader.requireTagEnd();

    return make_shared<RectangularMesh<1,RectilinearAxis>>(std::move(axis));
}


static shared_ptr<Mesh> readRegularMesh1D(XMLReader& reader)
{
    double start, stop;
    size_t count;

    reader.requireTag();

    if (reader.getNodeName() != "axis") throw XMLUnexpectedElementException(reader, "<axis>");

    start = reader.requireAttribute<double>("start");
    stop = reader.requireAttribute<double>("stop");
    count = reader.requireAttribute<size_t>("num");

    reader.requireTagEnd();

    reader.requireTagEnd();

    return make_shared<RectangularMesh<1,RegularAxis>>(start, stop, count);
}

static RegisterMeshReader rectilinearmesh1d_reader("rectilinear1d", readRectilinearMesh1D);
static RegisterMeshReader regularmesh1d_reader("regular1d", readRegularMesh1D);*/




