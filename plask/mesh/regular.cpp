#include "regular.h"

namespace plask {

template<>
shared_ptr<RegularMesh2D> RegularMesh2D::getMidpointsMesh() {
    if (this->midpoints_cache) return this->midpoints_cache;

    this->midpoints_cache = make_shared<RegularMesh2D>(
        RegularAxis(this->axis0.first() + 0.5*this->axis0.step(), this->axis0.last() - 0.5*this->axis0.step(), this->axis0.size()-1),
        RegularAxis(this->axis1.first() + 0.5*this->axis1.step(), this->axis1.last() - 0.5*this->axis1.step(), this->axis1.size()-1)
    );
    return this->midpoints_cache;
}

template<>
shared_ptr<RegularMesh3D> RegularMesh3D::getMidpointsMesh() {
    if (this->midpoints_cache) return this->midpoints_cache;

    this->midpoints_cache = make_shared<RegularMesh3D>(
        RegularAxis(this->axis0.first() + 0.5*this->axis0.step(), this->axis0.last() - 0.5*this->axis0.step(), this->axis0.size()-1),
        RegularAxis(this->axis1.first() + 0.5*this->axis1.step(), this->axis1.last() - 0.5*this->axis1.step(), this->axis1.size()-1),
        RegularAxis(this->axis2.first() + 0.5*this->axis2.step(), this->axis2.last() - 0.5*this->axis2.step(), this->axis2.size()-1)
    );
    return this->midpoints_cache;
}


template <>
void RegularMesh2D::writeXML(XMLElement& object) const {
    object.attr("type", "regular2d");
    object.addTag("axis0").attr("start", axis0.first()).attr("stop", axis0.last()).attr("num", axis0.size());
    object.addTag("axis1").attr("start", axis1.first()).attr("stop", axis1.last()).attr("num", axis1.size());
}

template <>
void RegularMesh3D::writeXML(XMLElement& object) const {
    object.attr("type", "regular3d");
    object.addTag("axis0").attr("start", axis0.first()).attr("stop", axis0.last()).attr("num", axis0.size());
    object.addTag("axis1").attr("start", axis1.first()).attr("stop", axis1.last()).attr("num", axis1.size());
    object.addTag("axis2").attr("start", axis2.first()).attr("stop", axis2.last()).attr("num", axis2.size());
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
        double stop = reader.requireAttribute<double>("stop");
        size_t count = reader.requireAttribute<size_t>("num");
        axes[node] = std::make_tuple(start, stop, count);

        reader.requireTagEnd();
    }
    reader.requireTagEnd();

    return make_shared<RegularMesh2D>(RegularAxis(std::get<0>(axes["axis0"]), std::get<1>(axes["axis0"]), std::get<2>(axes["axis0"])),
                                      RegularAxis(std::get<0>(axes["axis1"]), std::get<1>(axes["axis1"]), std::get<2>(axes["axis1"])));
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
        double stop = reader.requireAttribute<double>("stop");
        size_t count = reader.requireAttribute<size_t>("num");
        axes[node] = std::make_tuple(start, stop, count);

        reader.requireTagEnd();
    }
    reader.requireTagEnd();

    return make_shared<RegularMesh3D>(RegularAxis(std::get<0>(axes["axis0"]), std::get<1>(axes["axis0"]), std::get<2>(axes["axis0"])),
                                      RegularAxis(std::get<0>(axes["axis1"]), std::get<1>(axes["axis1"]), std::get<2>(axes["axis1"])),
                                      RegularAxis(std::get<0>(axes["axis2"]), std::get<1>(axes["axis2"]), std::get<2>(axes["axis2"])));
}

static RegisterMeshReader regularmesh2d_reader("regular2d", readRegularMesh2D);
static RegisterMeshReader regularmesh3d_reader("regular3d", readRegularMesh3D);


} // namespace plask
