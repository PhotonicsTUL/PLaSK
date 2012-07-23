#include "regular.h"

#include "rectangular2d_impl.h"
#include "rectangular3d_impl.h"

namespace plask {

template class RectangularMesh2D<RegularMesh1D>;
template class RectangularMesh3D<RegularMesh1D>;

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
