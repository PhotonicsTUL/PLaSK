#include <deque>
#include <boost/algorithm/string.hpp>

#include "rectilinear.h"

namespace plask {

template<>
shared_ptr<RectilinearMesh2D> RectilinearMesh2D::getMidpointsMesh() {

    if (this->midpoints_cache) return this->midpoints_cache;

    if (this->axis0.size() < 2 || this->axis1.size() < 2)
        throw BadMesh("getMidpointsMesh", "at least two points in each direction are required");

    RectilinearMesh1D line0;
    for (auto a = this->axis0.begin(), b = this->axis0.begin()+1; b != this->axis0.end(); ++a, ++b)
        line0.addPoint((*a + *b) / 2.0);    // "/ 2.0" is better than "* 0.5", because 2.0 has accurate representation

    RectilinearMesh1D line1;
    for (auto a = this->axis1.begin(), b = this->axis1.begin()+1; b != this->axis1.end(); ++a, ++b)
        line1.addPoint((*a + *b) / 2.0);

    this->midpoints_cache = make_shared<RectilinearMesh2D>(line0, line1, this->getIterationOrder());
    return this->midpoints_cache;
}

template<>
shared_ptr<RectilinearMesh3D> RectilinearMesh3D::getMidpointsMesh() {

    if (this->midpoints_cache) return this->midpoints_cache;

    if (this->axis0.size() < 2 || this->axis1.size() < 2 || this->axis2.size() < 2)
        throw BadMesh("getMidpointsMesh", "at least two points in each direction are required");

    RectilinearMesh1D line0;
    for (auto a = this->axis0.begin(), b = this->axis0.begin()+1; b != this->axis0.end(); ++a, ++b)
        line0.addPoint((*a + *b) / 2.0);

    RectilinearMesh1D line1;
    for (auto a = this->axis1.begin(), b = this->axis1.begin()+1; b != this->axis1.end(); ++a, ++b)
        line1.addPoint((*a + *b) / 2.0);

    RectilinearMesh1D line2;
    for (auto a = this->axis2.begin(), b = this->axis2.begin()+1; b != this->axis2.end(); ++a, ++b)
        line2.addPoint((*a + *b) / 2.0);

    this->midpoints_cache = make_shared<RectilinearMesh3D>(line0, line1, line2, this->getIterationOrder());
    return this->midpoints_cache;
}

template <>
void RectilinearMesh2D::writeXML(XMLElement& object) const {
    object.attr("type", "rectilinear2d");
    for (size_t n = 0; n != 2; ++n) {
        auto axis = object.addTag("axis"+boost::lexical_cast<std::string>(n));
        axis.indent();
        for (auto x: this->axis(n)) {
            axis.writeText(x);
            axis.writeText(" ");
        }
        axis.writeText("\n");
    }
}

template <>
void RectilinearMesh3D::writeXML(XMLElement& object) const {
    object.attr("type", "rectilinear3d");
    for (size_t n = 0; n != 3; ++n) {
        auto axis = object.addTag("axis"+boost::lexical_cast<std::string>(n));
        axis.indent();
        for (auto x: this->axis(n)) {
            axis.writeText(x);
            axis.writeText(" ");
        }
        axis.writeText("\n");
    }
}



static shared_ptr<Mesh> readRectilinearMesh2D(XMLReader& reader)
{
    std::map<std::string,std::vector<double>> axes;

    for (int i = 0; i < 2; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();

        if (node != "axis0" && node != "axis1") throw XMLUnexpectedElementException(reader, "<axis0> or <axis1>");
        if (axes.find(node) != axes.end()) throw XMLDuplicatedElementException(std::string("<mesh>"), "tag <" + node + ">");

        std::string data = reader.requireText();
        for (auto point: boost::tokenizer<>(data))
            axes[node].push_back(boost::lexical_cast<double>(point));

        reader.requireTagEnd();
    }
    reader.requireTagEnd();

    return make_shared<RectilinearMesh2D>(std::move(axes["axis0"]), std::move(axes["axis1"]));
}

static shared_ptr<Mesh> readRectilinearMesh3D(XMLReader& reader)
{
    std::map<std::string,std::vector<double>> axes;

    for (int i = 0; i < 3; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();

        if (node != "axis0" && node != "axis1" && node != "axis2") throw XMLUnexpectedElementException(reader, "<axis0>, <axis1>, or <axis2>");
        if (axes.find(node) != axes.end()) throw XMLDuplicatedElementException(std::string("<mesh>"), "tag <" + node + ">");

        std::string data = reader.requireText();
        for (auto point: boost::tokenizer<>(data))
            axes[node].push_back(boost::lexical_cast<double>(point));

        reader.requireTagEnd();
    }
    reader.requireTagEnd();

    return make_shared<RectilinearMesh3D>(std::move(axes["axis0"]), std::move(axes["axis1"]), std::move(axes["axis2"]));
}

static RegisterMeshReader rectilinearmesh2d_reader("rectilinear2d", readRectilinearMesh2D);
static RegisterMeshReader rectilinearmesh3d_reader("rectilinear3d", readRectilinearMesh3D);

} // namespace plask
