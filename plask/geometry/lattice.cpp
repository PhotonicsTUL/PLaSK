#include "lattice.h"
#include "reader.h"


//used by lattice
#include <map>
#include <set>
#include <utility>



namespace plask {

template <int dim>
typename ArrangeContainer<dim>::Box ArrangeContainer<dim>::getBoundingBox() const {
    Box bbox;
    if (!_child) {
        bbox.makeInvalid();
    } else {
        Box box = _child->getBoundingBox();
        for (int i = 0; i != dim; ++i) {
            if (translation[i] >= 0.) {
                bbox.lower[i] = box.lower[i];
                bbox.upper[i] = box.upper[i] + (int(repeat_count)-1) * translation[i];
            } else {
                bbox.lower[i] = box.lower[i] + (int(repeat_count)-1) * translation[i];
                bbox.upper[i] = box.upper[i];
            }
        }
    }
    return bbox;
}

template <int dim>
typename ArrangeContainer<dim>::Box ArrangeContainer<dim>::getRealBoundingBox() const {
    return getChild()->getBoundingBox();
}

template <int dim>
void ArrangeContainer<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate,
                                        std::vector<ArrangeContainer<dim>::Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    if (repeat_count == 0 || !_child) return;
    std::size_t old_size = dest.size();
    _child->getBoundingBoxesToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (unsigned r = 1; r < repeat_count; ++r)
        for (std::size_t i = old_size; i < new_size; ++i)
            dest.push_back(dest[i] + translation * r);
}

template <int dim>
void ArrangeContainer<dim>::getObjectsToVec(const GeometryObject::Predicate& predicate,
                                  std::vector<shared_ptr<const GeometryObject>>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->shared_from_this());
        return;
    }
    if (repeat_count == 0 || !_child) return;
    std::size_t old_size = dest.size();
    _child->getObjectsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (unsigned r = 1; r < repeat_count; ++r)
        for (std::size_t i = old_size; i < new_size; ++i)
            dest.push_back(dest[i]);
}

template <int dim>
void ArrangeContainer<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate,
                                    std::vector<typename ArrangeContainer<dim>::DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    if (repeat_count == 0 || !_child) return;
    std::size_t old_size = dest.size();
    _child->getPositionsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (unsigned r = 1; r < repeat_count; ++r)
        for (std::size_t i = old_size; i < new_size; ++i)
            dest.push_back(dest[i] + translation * r);
}

template <int dim>
bool ArrangeContainer<dim>::contains(const ArrangeContainer<dim>::DVec& p) const {
    if (!_child) return false;
    auto lohi = bounds(p);
    for (int i = lohi.second; i >= lohi.first; --i)
        if (_child->contains(p - i * translation)) return true;
    return false;
}

template <int dim>
shared_ptr<Material> ArrangeContainer<dim>::getMaterial(const typename ArrangeContainer<dim>::DVec& p) const {
    if (!_child) return shared_ptr<Material>();
    auto lohi = bounds(p);
    for (int i = lohi.second; i >= lohi.first; --i)
        if (auto material = _child->getMaterial(p - i * translation)) return material;
    return shared_ptr<Material>();
}

template <int dim>
std::size_t ArrangeContainer<dim>::getChildrenCount() const {
    if (!_child) return 0;
    return repeat_count;
}

template <int dim>
shared_ptr<GeometryObject> ArrangeContainer<dim>::getChildNo(std::size_t child_no) const {
    if (child_no >= getChildrenCount())
        throw OutOfBoundsException("getChildNo", "child_no", child_no, 0, getChildrenCount()-1);
    return make_shared<Translation<dim>>(_child, child_no * translation);
}

template <int dim>
std::size_t ArrangeContainer<dim>::getRealChildrenCount() const {
    return GeometryObjectTransform<dim>::getChildrenCount();
}

template <int dim>
shared_ptr<GeometryObject> ArrangeContainer<dim>::getRealChildNo(std::size_t child_no) const {
    return GeometryObjectTransform<dim>::getRealChildNo(child_no);
}

template <int dim>
GeometryObject::Subtree ArrangeContainer<dim>::getPathsAt(const typename ArrangeContainer<dim>::DVec& point, bool all) const {
    if (!_child) return GeometryObject::Subtree();
    GeometryObject::Subtree result;
    auto lohi = bounds(point);
    if (all) {
        for (int i = lohi.first; i <= lohi.second; --i) {
            GeometryObject::Subtree child_path = _child->getPathsAt(point - i * translation, true);
            if (!child_path.empty())
                result.children.push_back(std::move(child_path));
        }
    } else {
        for (int i = lohi.second; i >= lohi.first; --i) {
            GeometryObject::Subtree child_path = _child->getPathsAt(point - i * translation, true);
            if (!child_path.empty()) {
                result.children.push_back(std::move(child_path));
                break;
            }
        }
    }
    if (!result.children.empty())
        result.object = this->shared_from_this();
    return result;
}

template <int dim>
shared_ptr<GeometryObjectTransform<dim>> ArrangeContainer<dim>::shallowCopy() const {
    return make_shared<ArrangeContainer<dim>>(_child, translation, repeat_count);
}

template <int dim>
typename ArrangeContainer<dim>::Box ArrangeContainer<dim>::fromChildCoords(const typename ArrangeContainer<dim>::ChildType::Box& child_bbox) const {
    return child_bbox;
}

template <>
void ArrangeContainer<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    if (translation.tran() != 0.) dest_xml_object.attr("d"+axes.getNameForTran(), translation.tran());
    if (translation.vert() != 0.) dest_xml_object.attr("d"+axes.getNameForVert(), translation.vert());
    dest_xml_object.attr("count", repeat_count);
    if (warn_overlapping) dest_xml_object.attr("warning", "false");
}

template <>
void ArrangeContainer<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    if (translation.lon() != 0.) dest_xml_object.attr("d"+axes.getNameForLong(), translation.lon());
    if (translation.tran() != 0.) dest_xml_object.attr("d"+axes.getNameForTran(), translation.tran());
    if (translation.vert() != 0.) dest_xml_object.attr("d"+axes.getNameForVert(), translation.vert());
    dest_xml_object.attr("count", repeat_count);
    if (warn_overlapping) dest_xml_object.attr("warning", "false");
}

shared_ptr<GeometryObject> read_arrange2d(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    typename ArrangeContainer<2>::DVec vec;
    vec.tran() = reader.source.getAttribute("d"+reader.getAxisTranName(), 0.);
    vec.vert() = reader.source.getAttribute("d"+reader.getAxisVertName(), 0.);
    unsigned repeat = reader.source.requireAttribute<unsigned>("count");
    bool warn = reader.source.getAttribute("warning", true);
    auto child = reader.readExactlyOneChild<typename ArrangeContainer<2>::ChildType>();
    return make_shared<ArrangeContainer<2>>(child, vec, repeat, warn);
}

shared_ptr<GeometryObject> read_arrange3d(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    typename ArrangeContainer<3>::DVec vec;
    vec.lon() = reader.source.getAttribute("d"+reader.getAxisLongName(), 0.);
    vec.tran() = reader.source.getAttribute("d"+reader.getAxisTranName(), 0.);
    vec.vert() = reader.source.getAttribute("d"+reader.getAxisVertName(), 0.);
    unsigned repeat = reader.source.requireAttribute<unsigned>("count");
    bool warn = reader.source.getAttribute("warning", true);
    auto child = reader.readExactlyOneChild<typename ArrangeContainer<3>::ChildType>();
    return make_shared<ArrangeContainer<3>>(child, vec, repeat, warn);
}

static GeometryReader::RegisterObjectReader arrange2d_reader(ArrangeContainer<2>::NAME, read_arrange2d);
static GeometryReader::RegisterObjectReader arrange3d_reader(ArrangeContainer<3>::NAME, read_arrange3d);

template struct PLASK_API ArrangeContainer<2>;
template struct PLASK_API ArrangeContainer<3>;


struct IntPoint { int x, y; };
typedef std::pair<IntPoint, IntPoint> IntSegment;

struct YEnds {
    // map: y -> 2*x, each (x, y) is point where shape begins or ends
    // odd 2*x are for each points between integer coordinates
    std::map<int, std::set<int>> coords;

    void add_d(int dbl_x, int y) {
        std::set<int>& dst = coords[y];
        auto ins_ans = dst.insert(dbl_x);
        if (!ins_ans.second) {    //element was already included
            dst.erase(ins_ans.first);   //we remove it
            if (dst.empty()) coords.erase(y);
        }
    }

    //void add(int x, int y) { add_d(2*x, y); }
    void add(IntPoint p) { add_d(2*p.x, p.y); }
};


/**
 * Find all points lied on sides and inside of the poligon described by segments.
 * @param segments in any order, without intersections (boost geometry can produce this)
 * @return coordinates of all (x, y) points inside the poligon in the map: y -> set of x
 */
std::map<int, std::set<int>> calcLatticePoints(const std::vector<IntSegment>& segments) {
    std::map<int, std::set<int>> result;
    YEnds ends;
    for (const IntSegment& segment: segments) {
        if (segment.first.y == segment.second.y) {
            std::set<int>& dst = result[segment.first.y];
            for (int x = segment.first.x; x <= segment.second.x; ++x)
                dst.insert(x);  // we imedietly add all points which lie on side
        } else {
            result[segment.first.y].insert(segment.first.x);    // we imedietly add all vertexes
            result[segment.second.y].insert(segment.second.x);  // we imedietly add all vertexes

            IntPoint low_y, hi_y;
            if (segment.first.y > segment.second.y) {
                low_y = segment.second;
                hi_y = segment.first;
            } else {
                low_y = segment.first;
                hi_y = segment.second;
            }
            int dx = hi_y.x - low_y.x;
            int dy = hi_y.y - low_y.y;  //dy > 0
            for (int y = low_y.y; y < hi_y.y; ++y) {
                // x = l/m + low_y.x
                int l = dx * (y - low_y.y);
                int x = l / dy + low_y.x;
                int rem = l % dy;
                if (rem == 0) {        //x, y is exactly on side
                    result[y].insert(x);    //so we imedietly add it
                    ends.add_d(2*x, y);
                } else {
                    // here: real x = x + rem / dy and dy>0
                    ends.add_d(rem > 0 ? 2*x+1 : 2*x-1, y);
                }
            }
            ends.add(hi_y); //add point with higher y to ends
        }
    }
    for (std::pair<const int, std::set<int>>& line: ends.coords) {
        assert(line.second.size() % 2 == 0);
        std::set<int>& dst = result[line.first];
        for (std::set<int>::iterator dblx_it = line.second.begin(); dblx_it != line.second.end(); ++dblx_it) {
            // we can exlude ends of the segments as eventualy already included
            int beg = (*dblx_it+1) / 2;
            ++dblx_it;  // this is ok because of line.second has even length
            int end = (*dblx_it+1) / 2;
            // add all points from range [beg, end)
            while (beg != end) {
                dst.insert(beg);
                ++beg;
            }
        }
    }
    return result;
}

}
