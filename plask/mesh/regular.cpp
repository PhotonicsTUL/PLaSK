#include "regular.h"

namespace plask {

template<>
shared_ptr<RegularMesh2D> RegularMesh2D::getMidpointsMesh() {
    if (this->midpoints_cache) return this->midpoints_cache;

    this->midpoints_cache = make_shared<RegularMesh2D>(
        RegularMesh1D(this->axis0.getFirst() + 0.5*this->axis0.getStep(), this->axis0.getLast() - 0.5*this->axis0.getStep(), this->axis0.size()-1),
        RegularMesh1D(this->axis1.getFirst() + 0.5*this->axis1.getStep(), this->axis1.getLast() - 0.5*this->axis1.getStep(), this->axis1.size()-1)
    );
    return this->midpoints_cache;
}

template<>
shared_ptr<RegularMesh3D> RegularMesh3D::getMidpointsMesh() {
    if (this->midpoints_cache) return this->midpoints_cache;

    this->midpoints_cache = make_shared<RegularMesh3D>(
        RegularMesh1D(this->axis0.getFirst() + 0.5*this->axis0.getStep(), this->axis0.getLast() - 0.5*this->axis0.getStep(), this->axis0.size()-1),
        RegularMesh1D(this->axis1.getFirst() + 0.5*this->axis1.getStep(), this->axis1.getLast() - 0.5*this->axis1.getStep(), this->axis1.size()-1),
        RegularMesh1D(this->axis2.getFirst() + 0.5*this->axis2.getStep(), this->axis2.getLast() - 0.5*this->axis2.getStep(), this->axis2.size()-1)
    );
    return this->midpoints_cache;
}


template <>
void RegularMesh2D::writeXML(XMLElement& object) const {
    object.attr("type", "regular2d");
    object.addTag("axis0").attr("start", axis0.getFirst()).attr("end", axis0.getLast()).attr("count", axis0.size());
    object.addTag("axis1").attr("start", axis1.getFirst()).attr("end", axis1.getLast()).attr("count", axis1.size());
}

template <>
void RegularMesh3D::writeXML(XMLElement& object) const {
    object.attr("type", "regular3d");
    object.addTag("axis0").attr("start", axis0.getFirst()).attr("end", axis0.getLast()).attr("count", axis0.size());
    object.addTag("axis1").attr("start", axis1.getFirst()).attr("end", axis1.getLast()).attr("count", axis1.size());
    object.addTag("axis2").attr("start", axis2.getFirst()).attr("end", axis2.getLast()).attr("count", axis2.size());
}


static shared_ptr<Mesh> readRegularMesh2D(XMLReader& reader)
{
    std::map<std::string,std::tuple<double,double,size_t>> axes;

    for (int i = 0; i < 2; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();

        if (node != "axis0" && node != "axis1") throw XMLUnexpectedElementException(reader, "<axis0> or <axis1>");
        if (axes.find(node) != axes.end()) throw XMLDuplicatedElementException(std::string("<mesh>"), "tag <" + node + ">");

        double start = reader.requireAttribute<double>("start");
        double end = reader.requireAttribute<double>("end");
        size_t count = reader.requireAttribute<size_t>("count");
        axes[node] = std::make_tuple(start, end, count);

        reader.requireTagEnd();
    }
    reader.requireTagEnd();

    return make_shared<RegularMesh2D>(RegularMesh1D(std::get<0>(axes["axis0"]), std::get<1>(axes["axis0"]), std::get<2>(axes["axis0"])),
                                      RegularMesh1D(std::get<0>(axes["axis1"]), std::get<1>(axes["axis1"]), std::get<2>(axes["axis1"])));
}

static shared_ptr<Mesh> readRegularMesh3D(XMLReader& reader)
{
    std::map<std::string,std::tuple<double,double,size_t>> axes;

    for (int i = 0; i < 3; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();

        if (node != "axis0" && node != "axis1") throw XMLUnexpectedElementException(reader, "<axis0>, <axis1>, or <axis2>");
        if (axes.find(node) != axes.end()) throw XMLDuplicatedElementException(std::string("<mesh>"), "tag <" + node + ">");

        double start = reader.requireAttribute<double>("start");
        double end = reader.requireAttribute<double>("end");
        size_t count = reader.requireAttribute<size_t>("count");
        axes[node] = std::make_tuple(start, end, count);

        reader.requireTagEnd();
    }
    reader.requireTagEnd();

    return make_shared<RegularMesh3D>(RegularMesh1D(std::get<0>(axes["axis0"]), std::get<1>(axes["axis0"]), std::get<2>(axes["axis0"])),
                                      RegularMesh1D(std::get<0>(axes["axis1"]), std::get<1>(axes["axis1"]), std::get<2>(axes["axis1"])),
                                      RegularMesh1D(std::get<0>(axes["axis2"]), std::get<1>(axes["axis2"]), std::get<2>(axes["axis2"])));
}

static RegisterMeshReader regularmesh2d_reader("regular2d", readRegularMesh2D);
static RegisterMeshReader regularmesh3d_reader("regular3d", readRegularMesh3D);


} // namespace plask
