#include "unordered1d.h"

#include "../utils/stl.h"
#include "../log/log.h"

namespace plask {

UnorderedAxis::UnorderedAxis(std::initializer_list<PointType> points): points(points) {}

UnorderedAxis::UnorderedAxis(const std::vector<PointType>& points): points(points) {}

UnorderedAxis::UnorderedAxis(std::vector<PointType>&& points): points(std::move(points)) {}

bool UnorderedAxis::operator==(const plask::UnorderedAxis& to_compare) const {
    return points == to_compare.points;
}

void UnorderedAxis::writeXML(XMLElement &object) const {
    object.attr("type", "unordered");
    //object.indent();
    for (auto x: this->points) {
        object.writeText(x);
        object.writeText(" ");
    }
    //object.writeText("\n");
}


UnorderedAxis::native_const_iterator UnorderedAxis::findNearest(double to_find) const {
    double dist = std::numeric_limits<double>::max();
    UnorderedAxis::native_const_iterator found;
    for (auto first = begin(), last = end(); first != last; ++first) {
        double d = abs(*first - to_find);
        if (d < dist) {
            dist = d;
            found = first;
        }
    }
    return end();
}

UnorderedAxis &UnorderedAxis::operator=(const RectangularAxis &src) {
    bool resized = size() != src.size();
    points.clear();
    points.reserve(src.size());
    for (auto i: src) points.push_back(i);
    if (resized) fireResized(); else fireChanged();
    return *this;
}

UnorderedAxis &UnorderedAxis::operator=(UnorderedAxis &&src) {
    bool resized = size() != src.size();
    this->points = std::move(src.points);
    if (resized) fireResized(); else fireChanged();
    return *this;
}

UnorderedAxis &UnorderedAxis::operator=(const UnorderedAxis &src) {
    bool resized = size() != src.size();
    points = src.points;
    if (resized) fireResized(); else fireChanged();
    return *this;
}

void UnorderedAxis::appendPoint(double new_node_cord) {
    points.push_back(new_node_cord);
    fireResized();
}

void UnorderedAxis::insertPoint(size_t index, double new_node_cord) {
    points.insert(points.begin() + index, new_node_cord);
    fireResized();
}

void UnorderedAxis::removePoint(std::size_t index) {
    points.erase(points.begin() + index);
    fireResized();
}


void UnorderedAxis::clear() {
    points.clear();
    fireResized();
}

shared_ptr<RectangularAxis> UnorderedAxis::clone() const {
    return make_shared<UnorderedAxis>(*this);
}

// shared_ptr<OrderedMesh1D> readUnorderedMeshAxis(XMLReader& reader) {
//     auto result = make_shared<UnorderedMesh1D>();
//     std::string data = reader.requireTextInCurrentTag();
//     for (auto point: boost::tokenizer<>(data)) {
//         try {
//             double val = boost::lexical_cast<double>(point);
//             result->appendPoint(val);
//         } catch (boost::bad_lexical_cast) {
//             throw XMLException(reader, format("Value '%1%' cannot be converted to float", point));
//         }
//     }
//     return result;
// }
// 
// shared_ptr<OrderedMesh1D> readUnorderedMesh1D(XMLReader& reader) {
//     reader.requireTag("axis");
//     auto result = readUnorderedMeshAxis(reader);
//     reader.requireTagEnd();
//     return result;
// }
// 
// static RegisterMeshReader rectilinearmesh_reader("unordered", readUnorderedMesh1D);


}   // namespace plask
