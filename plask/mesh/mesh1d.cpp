#include "mesh1d.h"

namespace plask {
    
template <>
void Mesh1D<RectilinearAxis>::writeXML(XMLElement& object) const {
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
void Mesh1D<RegularAxis>::writeXML(XMLElement& object) const {
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

    return make_shared<Mesh1D<RectilinearAxis>>(std::move(axis));
}


static shared_ptr<Mesh> readRegularMesh1D(XMLReader& reader)
{
    double start, stop;
    size_t count;

    for (int i = 0; i < 2; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();

        if (node != "axis") throw XMLUnexpectedElementException(reader, "<axis>");

        start = reader.requireAttribute<double>("start");
        stop = reader.requireAttribute<double>("stop");
        count = reader.requireAttribute<size_t>("num");

        reader.requireTagEnd();
    }
    reader.requireTagEnd();

    return make_shared<Mesh1D<RegularAxis>>(start, stop, count);
}

static RegisterMeshReader rectilinearmesh1d_reader("rectilinear1d", readRectilinearMesh1D);
static RegisterMeshReader regularmesh1d_reader("regular1d", readRegularMesh1D);
    
} // namespace plask
