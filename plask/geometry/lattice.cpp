#include "lattice.h"
#include "reader.h"
#include "../manager.h"

//used by lattice
#include <map>
#include <set>
#include <utility>



namespace plask {

template <int dim>
std::pair<int, int> ArrangeContainer<dim>::bounds(const ArrangeContainer<dim>::DVec &vec) const {
    if (!this->hasChild() || repeat_count == 0) return std::make_pair(1, 0);
    auto box = _child->getBoundingBox();
    int hi = repeat_count - 1, lo = 0;
    for (int i = 0; i != dim; ++i) {
        if (translation[i] > 0.) {
            lo = max(1 + int(std::floor((vec[i] - box.upper[i]) / translation[i])), lo);
            hi = min(int(std::floor((vec[i] - box.lower[i]) / translation[i])), hi);
        } else if (translation[i] < 0.) {
            lo = max(1 + int(std::floor((vec[i] - box.lower[i]) / translation[i])), lo);
            hi = min(int(std::floor((vec[i] - box.upper[i]) / translation[i])), hi);
        } else if (vec[i] < box.lower[i] || box.upper[i] < vec[i]) {
            return std::make_pair(1, 0);
        }
    }
    return std::make_pair(lo, hi);
}

template <int dim>
void ArrangeContainer<dim>::warmOverlaping() const
{
    if (warn_overlapping && this->hasChild()) {
        Box box = this->_child->getBoundingBox();
        box -= box.lower;
        if (box.intersects(box + translation))
            writelog(LOG_WARNING, "Arrange: item bboxes overlap");
    }
}

template <int dim>
typename ArrangeContainer<dim>::Box ArrangeContainer<dim>::getBoundingBox() const {
    Box bbox;
    if (!this->hasChild()) {
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
    if (repeat_count == 0 || !this->hasChild()) return;
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
    if (repeat_count == 0 || !this->hasChild()) return;
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
    if (repeat_count == 0 || !this->hasChild()) return;
    std::size_t old_size = dest.size();
    _child->getPositionsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (unsigned r = 1; r < repeat_count; ++r)
        for (std::size_t i = old_size; i < new_size; ++i)
            dest.push_back(dest[i] + translation * r);
}

template <int dim>
bool ArrangeContainer<dim>::contains(const ArrangeContainer<dim>::DVec& p) const {
    if (!this->hasChild()) return false;
    auto lohi = bounds(p);
    for (int i = lohi.second; i >= lohi.first; --i)
        if (_child->contains(p - i * translation)) return true;
    return false;
}

template <int dim>
shared_ptr<Material> ArrangeContainer<dim>::getMaterial(const typename ArrangeContainer<dim>::DVec& p) const {
    if (!this->hasChild()) return shared_ptr<Material>();
    auto lohi = bounds(p);
    for (int i = lohi.second; i >= lohi.first; --i)
        if (auto material = _child->getMaterial(p - i * translation)) return material;
    return shared_ptr<Material>();
}

template <int dim>
std::size_t ArrangeContainer<dim>::getChildrenCount() const {
    return this->hasChild() ? repeat_count : 0;
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
    if (!this->hasChild()) return GeometryObject::Subtree();
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
    auto child = reader.readExactlyOneChild<typename ArrangeContainer<2>::ChildType>(!reader.manager.draft);
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
    auto child = reader.readExactlyOneChild<typename ArrangeContainer<3>::ChildType>(!reader.manager.draft);
    return make_shared<ArrangeContainer<3>>(child, vec, repeat, warn);
}

static GeometryReader::RegisterObjectReader arrange2d_reader(ArrangeContainer<2>::NAME, read_arrange2d);
static GeometryReader::RegisterObjectReader arrange3d_reader(ArrangeContainer<3>::NAME, read_arrange3d);

template struct PLASK_API ArrangeContainer<2>;
template struct PLASK_API ArrangeContainer<3>;



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
    void add(Vec<2, int> p) { add_d(2*p.c0, p.c1); }
};

/**
 * Iterate over segments.
 * SegmentsIterator it(segments);
 * while (it.next()) {
 *  //use it.first, it.second
 * }
 */
struct SegmentsIterator {

    const std::vector< std::vector<Vec<2, int>> >& segments;

    Vec<2, int> first, second;

    int seg_nr, point_nr;

    /**
     * Construct iterator. After first next() call iterator will point to the first point.
     * @param segments vector of closed polygons, each consist of number of successive verticles, one side is between last and first vertex.
     *  These polygons are xored. Sides must not cross each other.
     */
    SegmentsIterator(const std::vector< std::vector<Vec<2, int>> >& segments): segments(segments), seg_nr(0), point_nr(-1) {}

    /**
     * Go to next segment. Set new values of first, second.
     * @return @c true only if next segment exists.
     */
    bool next() {
        if (seg_nr == segments.size()) return false;
        ++point_nr;
        if (point_nr == segments[seg_nr].size()) {  //end of segment?
            point_nr = 0;   //go to next segment
            do {
                ++seg_nr;
                if (seg_nr == segments.size()) return false;
            } while (segments[seg_nr].empty());  //loop skips empty segments
        }
        first = segments[seg_nr][point_nr];
        second = segments[seg_nr][(point_nr+1) % segments[seg_nr].size()];
        return true;
    }

};



void Lattice::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    container->forEachChild([&](const Translation<3> &child) { child.getBoundingBoxesToVec(predicate, dest, path); }, path);
}

void Lattice::getObjectsToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObject> >& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->shared_from_this());
        return;
    }
    container->forEachChild([&](const Translation<3> &child) { child.getObjectsToVec(predicate, dest, path); }, path);
}

void Lattice::getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<3>::ZERO_VEC);
        return;
    }
    container->forEachChild([&](const Translation<3> &child) { child.getPositionsToVec(predicate, dest, path); }, path);
}

GeometryObject::Subtree Lattice::getPathsTo(const GeometryObject& el, const PathHints* path) const {
    if (this == &el) return this->shared_from_this();
    GeometryObject::Subtree result = container->getPathsTo(el, path);
    if (result.object) result.object = this->shared_from_this();
    return result;
}

void Lattice::refillContainer()
{
    container->clear();

    std::map<int, std::set<int>> result;    //coordinates of all (x, y) points inside the poligon in the map: y -> set of x
    YEnds ends;
    SegmentsIterator segment(this->segments);
    while (segment.next()) {
        if (segment.first.c1 == segment.second.c1) {
            std::set<int>& dst = result[segment.first.c1];           
            if (segment.first.c0 > segment.second.c0)
                std::swap(segment.first.c0, segment.second.c0);
            for (int x = segment.first.c0; x <= segment.second.c0; ++x)
                dst.insert(x);  // we imedietly add all points which lie on side
        } else {
            result[segment.first.c1].insert(segment.first.c0);    // we imedietly add all vertexes
            result[segment.second.c1].insert(segment.second.c0);  // we imedietly add all vertexes

            Vec<2, int> low_y, hi_y;
            if (segment.first.c1 > segment.second.c1) {
                low_y = segment.second;
                hi_y = segment.first;
            } else {
                low_y = segment.first;
                hi_y = segment.second;
            }
            int dx = hi_y.c0 - low_y.c0;
            int dy = hi_y.c1 - low_y.c1;  //dy > 0
            for (int y = low_y.c1+1; y < hi_y.c1; ++y) {
                // x = l/m + low_y.c0
                int l = dx * (y - low_y.c1);
                int x = l / dy + low_y.c0;
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

    for (auto& p: result) { //p is pair: set of x's, and one y
        for (auto x: p.second) {
            container->addUnsafe(this->_child, x * vec0 + p.first * vec1);
        }
    }
}




void Lattice::writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const {
    if (vec0.lon() != 0.)  dest_xml_object.attr("a"+axes.getNameForLong(), vec0.lon());
    if (vec0.tran() != 0.) dest_xml_object.attr("a"+axes.getNameForTran(), vec0.tran());
    if (vec0.vert() != 0.) dest_xml_object.attr("a"+axes.getNameForVert(), vec0.vert());
    if (vec1.lon() != 0.)  dest_xml_object.attr("b"+axes.getNameForLong(), vec1.lon());
    if (vec1.tran() != 0.) dest_xml_object.attr("b"+axes.getNameForTran(), vec1.tran());
    if (vec1.vert() != 0.) dest_xml_object.attr("b"+axes.getNameForVert(), vec1.vert());
}

#define LATTICE_XML_SEGMENTS_TAG_NAME "segments"

void Lattice::writeXMLChildren(XMLWriter::Element &dest_xml_object, GeometryObject::WriteXMLCallback &write_cb, const AxisNames &axes) const {
    {   // write <segments>
        XMLElement segments_tag(dest_xml_object, LATTICE_XML_SEGMENTS_TAG_NAME);
        bool first = true;
        for (const std::vector<Vec<2, int>>& s: segments) {
            if (!first) segments_tag.writeText(" ^\n");
            first = false;
            bool first_point = true;
            for (Vec<2, int> p: s) {
                if (!first_point) { segments_tag.writeText("; "); }
                first_point = false;
                segments_tag.writeText(p[0]).writeText(' ').writeText(p[1]);
            }
        }
    }   // </segments>
    // write child:
    GeometryObjectTransform<3>::writeXML(dest_xml_object, write_cb, axes);
}

void Lattice::setSegments(std::vector< std::vector<Vec<2, int>> > new_segments) {
    this->segments = std::move(new_segments);
    refillContainer();
}

shared_ptr<GeometryObject> read_lattice(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    auto result = make_shared<Lattice>();
    result->vec0.lon() =  reader.source.getAttribute("a"+reader.getAxisLongName(), 0.);
    result->vec0.tran() = reader.source.getAttribute("a"+reader.getAxisTranName(), 0.);
    result->vec0.vert() = reader.source.getAttribute("a"+reader.getAxisVertName(), 0.);
    result->vec1.lon() =  reader.source.getAttribute("b"+reader.getAxisLongName(), 0.);
    result->vec1.tran() = reader.source.getAttribute("b"+reader.getAxisTranName(), 0.);
    result->vec1.vert() = reader.source.getAttribute("b"+reader.getAxisVertName(), 0.);
    reader.source.requireTag(LATTICE_XML_SEGMENTS_TAG_NAME);
    std::string segments = reader.source.requireTextInCurrentTag();
    boost::tokenizer<boost::char_separator<char> > tokens(segments, boost::char_separator<char>(" \t\n\r", ";^"));
    result->segments.emplace_back();
    int cords_in_current_point = 0;
    for (const std::string& t: tokens) {
        if (t == ";" || t == "^") { //end of point or segment
            if (cords_in_current_point != 2) throw Exception("Each point must have two coordinates.");
            cords_in_current_point = 0;
            if (t == "^")   //end of segment, add new one
                result->segments.emplace_back();
        } else {    //end of point coordinate
            if (cords_in_current_point == 2) throw Exception("End of point (\";\") or segment (\"^\") was expected, but got \"%1%\".", t);
            if (cords_in_current_point == 0) result->segments.back().emplace_back();
            result->segments.back().back()[cords_in_current_point++] = boost::lexical_cast<double>(t);
        }
    }
    result->setChild(reader.readExactlyOneChild<typename Lattice::ChildType>(!reader.manager.draft));
    result->refillContainer();
    return result;
}

static GeometryReader::RegisterObjectReader lattice_reader(Lattice::NAME, read_lattice);


}
