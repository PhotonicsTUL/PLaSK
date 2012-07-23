#include "regular.h"

#include "rectangular2d_impl.h"
#include "rectangular3d_impl.h"

namespace plask {

template class RectangularMesh2D<RegularMesh1D>;
template class RectangularMesh3D<RegularMesh1D>;

template<>
RegularMesh2D RegularMesh2D::getMidpointsMesh() const {
    return RegularMesh2D(
        RegularMesh1D(c0.getFirst() + 0.5*c0.getStep(), c0.getLast() - 0.5*c0.getStep(), c0.size()-1),
        RegularMesh1D(c1.getFirst() + 0.5*c1.getStep(), c1.getLast() - 0.5*c1.getStep(), c1.size()-1)
    );
}

template<>
RegularMesh3D RegularMesh3D::getMidpointsMesh() const {
    return RegularMesh3D(
        RegularMesh1D(c0.getFirst() + 0.5*c0.getStep(), c0.getLast() - 0.5*c0.getStep(), c0.size()-1),
        RegularMesh1D(c1.getFirst() + 0.5*c1.getStep(), c1.getLast() - 0.5*c1.getStep(), c1.size()-1),
        RegularMesh1D(c2.getFirst() + 0.5*c2.getStep(), c2.getLast() - 0.5*c2.getStep(), c2.size()-1)
    );
}


template <>
void RegularMesh2D::serialize(XMLWriter& writer, const std::string name) const {
    auto mesh = writer.addTag("mesh");
    mesh.attr("type", "regular2d").attr("name", name);
    mesh.addTag("axis0").attr("start", c0.getFirst()).attr("end", c0.getLast()).attr("count", c0.size());
    mesh.addTag("axis1").attr("start", c1.getFirst()).attr("end", c1.getLast()).attr("count", c1.size());
}

template <>
void RegularMesh3D::serialize(XMLWriter& writer, const std::string name) const {
    auto mesh = writer.addTag("mesh");
    mesh.attr("type", "regular3d").attr("name", name);
    mesh.addTag("axis0").attr("start", c0.getFirst()).attr("end", c0.getLast()).attr("count", c0.size());
    mesh.addTag("axis1").attr("start", c1.getFirst()).attr("end", c1.getLast()).attr("count", c1.size());
    mesh.addTag("axis2").attr("start", c2.getFirst()).attr("end", c2.getLast()).attr("count", c2.size());
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
