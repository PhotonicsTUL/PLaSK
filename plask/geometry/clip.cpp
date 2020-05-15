#include "clip.h"

#include "../manager.h"
#include "reader.h"

#include <limits>

#define PLASK_CLIP2D_NAME ("clip" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D)
#define PLASK_CLIP3D_NAME ("clip" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D)

namespace plask {

template <int dim> const char* Clip<dim>::NAME = dim == 2 ? PLASK_CLIP2D_NAME : PLASK_CLIP3D_NAME;

template <int dim> shared_ptr<Material> Clip<dim>::getMaterial(const typename Clip<dim>::DVec& p) const {
    return (this->hasChild() && clipBox.contains(p)) ? this->_child->getMaterial(p) : shared_ptr<Material>();
}

template <int dim> bool Clip<dim>::contains(const typename Clip<dim>::DVec& p) const {
    return this->hasChild() && clipBox.contains(p) && this->_child->contains(p);
}

template <int dim>
GeometryObject::Subtree Clip<dim>::getPathsAt(const typename Clip<dim>::DVec& point, bool all) const {
    if (this->hasChild() && clipBox.contains(point))
        return GeometryObject::Subtree::extendIfNotEmpty(this, this->_child->getPathsAt(point, all));
    else
        return GeometryObject::Subtree();
}

template <int dim>
typename Clip<dim>::Box Clip<dim>::fromChildCoords(const typename Clip<dim>::ChildType::Box& child_bbox) const {
    return child_bbox.intersection(clipBox);
}

template <int dim>
void Clip<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate,
                                  std::vector<DVec>& dest,
                                  const PathHints* path) const {
    this->_getNotChangedPositionsToVec(this, predicate, dest, path);
}

template <int dim> shared_ptr<GeometryObject> Clip<dim>::shallowCopy() const { return copyShallow(); }

template <int dim>
void Clip<dim>::addPointsAlong(std::set<double>& points,
                               Primitive<3>::Direction direction,
                               unsigned max_steps,
                               double min_step_size) const {
    if (this->_child) {
        std::set<double> child_points;
        this->_child->addPointsAlong(child_points, direction, this->max_steps ? this->max_steps : max_steps,
                               this->min_step_size ? this->min_step_size : min_step_size);
        auto bbox = this->getBoundingBox();
        points.insert(bbox.lower[int(direction) - (3 - dim)]);
        for (double p : child_points) {
            if (clipBox.lower[int(direction) - (3 - dim)] <= p && p <= clipBox.upper[int(direction) - (3 - dim)])
                points.insert(p);
        }
        points.insert(bbox.upper[int(direction) - (3 - dim)]);
    }
}

template <int dim>
void Clip<dim>::addLineSegmentsToSet(std::set<typename GeometryObjectD<dim>::LineSegment>& segments,
                                     unsigned max_steps,
                                     double min_step_size) const {
    if (this->_child) {
        std::set<typename GeometryObjectD<dim>::LineSegment> child_segments;
        this->_child->addLineSegmentsToSet(child_segments, this->max_steps ? this->max_steps : max_steps,
                                     this->min_step_size ? this->min_step_size : min_step_size);
        for (const auto& s : child_segments) {
            bool in0 = clipBox.contains(s.p0()), in1 = clipBox.contains(s.p1());
            if (in0 && in1) segments.insert(s);
            else if (in0 || in1) {
                // TODO clip segments
            }
        }
    }
}

template <typename ClipBoxType>
inline static void writeClip2D3D(XMLWriter::Element& dest_xml_object, const ClipBoxType& clipBox) {
    if (clipBox.left() > -std::numeric_limits<double>::infinity()) dest_xml_object.attr("left", clipBox.left());
    if (clipBox.right() < std::numeric_limits<double>::infinity()) dest_xml_object.attr("right", clipBox.right());
    if (clipBox.bottom() > -std::numeric_limits<double>::infinity()) dest_xml_object.attr("bottom", clipBox.bottom());
    if (clipBox.top() < std::numeric_limits<double>::infinity()) dest_xml_object.attr("top", clipBox.top());
}

template <> void Clip<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    GeometryObjectTransform<2>::writeXMLAttr(dest_xml_object, axes);
    writeClip2D3D(dest_xml_object, clipBox);
}

template <> void Clip<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    GeometryObjectTransform<3>::writeXMLAttr(dest_xml_object, axes);
    writeClip2D3D(dest_xml_object, clipBox);
    if (clipBox.back() > -std::numeric_limits<double>::infinity()) dest_xml_object.attr("back", clipBox.back());
    if (clipBox.front() < std::numeric_limits<double>::infinity()) dest_xml_object.attr("front", clipBox.front());
}

template <typename ClipType> inline static void setupClip2D3D(GeometryReader& reader, ClipType& clip) {
    clip.clipBox.left() = reader.source.getAttribute<double>("left", -std::numeric_limits<double>::infinity());
    clip.clipBox.right() = reader.source.getAttribute<double>("right", std::numeric_limits<double>::infinity());
    clip.clipBox.top() = reader.source.getAttribute<double>("top", std::numeric_limits<double>::infinity());
    clip.clipBox.bottom() = reader.source.getAttribute<double>("bottom", -std::numeric_limits<double>::infinity());
    clip.setChild(reader.readExactlyOneChild<typename ClipType::ChildType>(!reader.manager.draft));
}

shared_ptr<GeometryObject> read_Clip2D(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    shared_ptr<Clip<2>> clip(new Clip<2>());
    setupClip2D3D(reader, *clip);
    return clip;
}

shared_ptr<GeometryObject> read_Clip3D(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    shared_ptr<Clip<3>> clip(new Clip<3>());
    clip->clipBox.back() = reader.source.getAttribute<double>("back", -std::numeric_limits<double>::infinity());
    clip->clipBox.front() = reader.source.getAttribute<double>("front", std::numeric_limits<double>::infinity());
    setupClip2D3D(reader, *clip);
    return clip;
}

static GeometryReader::RegisterObjectReader Clip2D_reader(PLASK_CLIP2D_NAME, read_Clip2D);
static GeometryReader::RegisterObjectReader Clip3D_reader(PLASK_CLIP3D_NAME, read_Clip3D);

template struct PLASK_API Clip<2>;
template struct PLASK_API Clip<3>;

}  // namespace plask
