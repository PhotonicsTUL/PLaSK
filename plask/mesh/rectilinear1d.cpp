#include "rectilinear1d.h"

#include "../utils/stl.h"
#include "../log/log.h"

namespace plask {

void RectilinearAxis::sortPointsAndRemoveNonUnique()
{
    std::sort(this->points.begin(), this->points.end());
    auto almost_equal = [](const double& x, const double& y) -> bool { return std::abs(x-y) < MIN_DISTANCE; };
    this->points.erase(std::unique(this->points.begin(), this->points.end(), almost_equal), this->points.end());
}

RectilinearAxis::RectilinearAxis(std::initializer_list<PointType> points): points(points) {
    sortPointsAndRemoveNonUnique();
}

RectilinearAxis::RectilinearAxis(const std::vector<PointType>& points): points(points) {
    sortPointsAndRemoveNonUnique();
}

RectilinearAxis::RectilinearAxis(std::vector<PointType>&& points): points(std::move(points)) {
    sortPointsAndRemoveNonUnique();
}

bool RectilinearAxis::operator==(const plask::RectilinearAxis& to_compare) const {
    return points == to_compare.points;
}

void RectilinearAxis::writeXML(XMLElement &object) const {
    object.attr("type", "rectilinear");
    object.indent();
    for (auto x: this->points) {
        object.writeText(x);
        object.writeText(" ");
    }
    object.writeText("\n");
}



RectilinearAxis::native_const_iterator RectilinearAxis::find(double to_find) const {
    return std::lower_bound(points.begin(), points.end(), to_find);
}
//
RectilinearAxis::native_const_iterator RectilinearAxis::findNearest(double to_find) const {
    return find_nearest_binary(points.begin(), points.end(), to_find);
}

bool RectilinearAxis::addPoint(double new_node_cord) {
    auto where = std::lower_bound(points.begin(), points.end(), new_node_cord);
    if (where == points.end()) {
        if (points.size() == 0 || new_node_cord - points.back() > MIN_DISTANCE) {
            points.push_back(new_node_cord);
            fireResized();
            return true;
        }
    } else {
        if (*where - new_node_cord > MIN_DISTANCE && (where == points.begin() || new_node_cord - *(where-1) > MIN_DISTANCE)) {
            points.insert(where, new_node_cord);
            fireResized();
            return true;
        }
    }
    return false;
}

void RectilinearAxis::addPointsLinear(double first, double last, std::size_t points_count) {
    if (points_count == 0) return;
    --points_count;
    double len = last - first;
    auto get_el = [&](std::size_t i) { return first + i * len / points_count; };
    addOrderedPoints(makeFunctorIndexedIterator(get_el, 0), makeFunctorIndexedIterator(get_el, points_count+1), points_count+1);
    fireResized();
}

void RectilinearAxis::removePoint(std::size_t index) {
    points.erase(points.begin() + index);
    fireResized();
}


void RectilinearAxis::clear() {
    points.clear();
    fireResized();
}

shared_ptr<RectangularAxis> RectilinearAxis::clone() const {
    return make_shared<RectilinearAxis>(*this);
}

shared_ptr<RectilinearMesh1D> readRectilinearMeshAxis(XMLReader& reader) {
    auto result = make_shared<RectilinearMesh1D>();
    if (reader.hasAttribute("start")) {
         double start = reader.requireAttribute<double>("start");
         double stop = reader.requireAttribute<double>("stop");
         size_t count = reader.requireAttribute<size_t>("num");
         result->addPointsLinear(start, stop, count);
         reader.requireTagEnd();
    } else {
         std::string data = reader.requireTextInCurrentTag();
         for (auto point: boost::tokenizer<>(data)) {
             try {
                 double val = boost::lexical_cast<double>(point);
                 result->addPoint(val);
             } catch (boost::bad_lexical_cast) {
                 throw XMLException(reader, format("Value '%1%' cannot be converted to float", point));
             }
         }
    }
    return result;
}

shared_ptr<RectilinearMesh1D> readRectilinearMesh1D(XMLReader& reader) {
    reader.requireTag("axis");
    auto result = readRectilinearMeshAxis(reader);
    reader.requireTagEnd();
    return result;
}

static RegisterMeshReader rectilinearmesh_reader("rectilinear", readRectilinearMesh1D);


// deprecated:

shared_ptr<RectilinearMesh1D> readRectilinearMesh1D_deprecated(XMLReader& reader) {
    writelog(LOG_WARNING, "Mesh type \"%1%\" is deprecated (will not work in future versions of PLaSK), use \"rectilinear\" instead.", reader.requireAttribute("type"));
    return readRectilinearMesh1D(reader);
}
RegisterMeshReader rectilinearmesh1d_reader("rectilinear1d", readRectilinearMesh1D_deprecated);




}   // namespace plask
