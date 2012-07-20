#include <deque>
#include <boost/algorithm/string.hpp>

#include "rectilinear.h"

#include "rectangular2d_impl.h"
#include "rectangular3d_impl.h"

namespace plask {

template class RectangularMesh2D<RectilinearMesh1D>;
template class RectangularMesh3D<RectilinearMesh1D>;



template<>
RectilinearMesh2D RectilinearMesh2D::getMidpointsMesh() const {

    if (c0.size() < 2 || c1.size() < 2) throw BadMesh("getMidpointsMesh", "at least two points in each direction are required");

    RectilinearMesh1D line0;
    for (auto a = c0.begin(), b = c0.begin()+1; b != c0.end(); ++a, ++b)
        line0.addPoint(0.5 * (*a + *b));

    RectilinearMesh1D line1;
    for (auto a = c1.begin(), b = c1.begin()+1; b != c1.end(); ++a, ++b)
        line1.addPoint(0.5 * (*a + *b));

    return RectilinearMesh2D(line0, line1, getIterationOrder());
}

template<>
RectilinearMesh3D RectilinearMesh3D::getMidpointsMesh() const {

    if (c0.size() < 2 || c1.size() < 2 || c2.size() < 2) throw BadMesh("getMidpointsMesh", "at least two points in each direction are required");

    RectilinearMesh1D line0;
    for (auto a = c0.begin(), b = c0.begin()+1; b != c0.end(); ++a, ++b)
        line0.addPoint(0.5 * (*a + *b));

    RectilinearMesh1D line1;
    for (auto a = c1.begin(), b = c1.begin()+1; b != c1.end(); ++a, ++b)
        line1.addPoint(0.5 * (*a + *b));

    RectilinearMesh1D line2;
    for (auto a = c2.begin(), b = c2.begin()+1; b != c2.end(); ++a, ++b)
        line2.addPoint(0.5 * (*a + *b));

    return RectilinearMesh3D(line0, line1, line2, getIterationOrder());
}



static shared_ptr<Mesh> readRectilinearMesh2D(XMLReader& reader)
{
    std::map<std::string,std::vector<double>> axes;

    for (int i = 0; i < 2; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();

        if (node != "axis0" && node != "axis1") throw XMLUnexpectedElementException("<axis0> or <axis1>");
        if (axes.find(node) != axes.end()) XMLUnexpectedElementException("non-repeated axis");

        reader.read();
        if (reader.getNodeType() != XMLReader::NODE_TEXT) throw XMLUnexpectedElementException("axis specification");

        std::deque<std::string> points;
        const char* data = reader.getNodeDataC();
        boost::split(points, data, boost::is_any_of(", \t\r\n"), boost::token_compress_on);

        axes[node].reserve(points.size());
        for (auto point: points) axes[node].push_back(boost::lexical_cast<double>(point));

        reader.requireTagEnd();
    }
    reader.requireTagEnd();

    return make_shared<RectilinearMesh2D>(axes["axis0"], axes["axis1"]);
}

static shared_ptr<Mesh> readRectilinearMesh3D(XMLReader& reader)
{
    std::map<std::string,std::vector<double>> axes;

    for (int i = 0; i < 3; ++i) {
        reader.requireTag();
        std::string node = reader.getNodeName();

        if (node != "axis0" && node != "axis1" && node != "axis2") throw XMLUnexpectedElementException("<axis0>, <axis1>, or <axis2>");
        if (axes.find(node) != axes.end()) XMLUnexpectedElementException("non-repeated axis");

        reader.read();
        if (reader.getNodeType() != XMLReader::NODE_TEXT) throw XMLUnexpectedElementException("axis specification");

        std::deque<std::string> points;
        const char* data = reader.getNodeDataC();
        boost::split(points, data, boost::is_any_of(", \n"), boost::token_compress_on);

        axes[node].reserve(points.size());
        for (auto point: points) axes[node].push_back(boost::lexical_cast<double>(point));

        reader.requireTagEnd();
    }
    reader.requireTagEnd();

    return make_shared<RectilinearMesh3D>(axes["axis0"], axes["axis1"], axes["axis2"]);
}

static RegisterMeshReader rectilinearmesh2d_reader("rectilinear2d", readRectilinearMesh2D);
static RegisterMeshReader rectilinearmesh3d_reader("rectilinear3d", readRectilinearMesh3D);

} // namespace plask
