#include "clip.h"

#include "reader.h"

#include <limits>

namespace plask {

template <int dim>
GeometryObject::Subtree Clip<dim>::getPathsAt(const Clip<dim>::DVec &point, bool all) const
{
        if (clipBox.contains(point))
            return GeometryObject::Subtree::extendIfNotEmpty(this, getChild()->getPathsAt(point, all));
        else
            return GeometryObject::Subtree();
}

template <int dim>
void Clip<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    std::vector<Box> result = getChild()->getBoundingBoxes(predicate, path);
    dest.reserve(dest.size() + result.size());
    for (Box& r: result) {
        r.makeIntersection(clipBox);
        dest.push_back(r);
    }
}

template <int dim>
void Clip<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    getChild()->getPositionsToVec(predicate, dest, path);
}

template <typename ClipBoxType>
inline static void writeClip2D3D(XMLWriter::Element& dest_xml_object, const ClipBoxType& clipBox) {
    dest_xml_object.attr("left", clipBox.left());
    dest_xml_object.attr("right", clipBox.right());
    dest_xml_object.attr("top", clipBox.top());
    dest_xml_object.attr("bottom", clipBox.bottom());
}

template <>
void Clip<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    writeClip2D3D(dest_xml_object, clipBox);
}

template <>
void Clip<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    writeClip2D3D(dest_xml_object, clipBox);
    dest_xml_object.attr("back", clipBox.back());
    dest_xml_object.attr("front", clipBox.front());
}

template <typename ClipType>
inline static void setupClip2D3D(GeometryReader& reader, ClipType& clip) {
    clip.clipBox.left() = reader.source.getAttribute<double>("left", - std::numeric_limits<double>::infinity());
    clip.clipBox.right() = reader.source.getAttribute<double>("right", std::numeric_limits<double>::infinity());
    clip.clipBox.top() = reader.source.getAttribute<double>("top", std::numeric_limits<double>::infinity());
    clip.clipBox.bottom() = reader.source.getAttribute<double>("bottom", - std::numeric_limits<double>::infinity());
    clip.setChild(reader.readExactlyOneChild<typename ClipType::ChildType>());
}

shared_ptr<GeometryObject> read_Clip2D(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    shared_ptr< Clip<2> > clip(new Clip<2>());
    setupClip2D3D(reader, *clip);
    return clip;
}

shared_ptr<GeometryObject> read_Clip3D(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    shared_ptr< Clip<3> > clip(new Clip<3>());
    clip->clipBox.back() = reader.source.getAttribute<double>("back", - std::numeric_limits<double>::infinity());
    clip->clipBox.front() = reader.source.getAttribute<double>("front", std::numeric_limits<double>::infinity());
    setupClip2D3D(reader, *clip);
    return clip;
}

static GeometryReader::RegisterObjectReader Clip2D_reader(Clip<2>::NAME, read_Clip2D);
static GeometryReader::RegisterObjectReader Clip3D_reader(Clip<3>::NAME, read_Clip3D);

template struct PLASK_API Clip<2>;
template struct PLASK_API Clip<3>;

}   // namespace plask
