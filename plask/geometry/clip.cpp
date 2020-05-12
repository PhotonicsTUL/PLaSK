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
void Clip<dim>::addPointsAlongToSet(std::set<double>& points,
                                    Primitive<3>::Direction direction,
                                    unsigned max_steps,
                                    double min_step_size) const {
    if (this->_child) {
        std::set<double> child_points;
        this->_child->addPointsAlongToSet(child_points, direction, this->max_steps ? this->max_steps : max_steps,
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
            if (in0 && in1)
                segments.insert(s);
            else
                addClippedSegment(segments, s.p0(), s.p1());
        }
    }
}

inline static void limitT(double& t0, double& t1, double tl[], double th[], int n, int x = -1) {
    t0 = 0.;
    t1 = 1.;
    for (size_t i = 0; i < n; ++i) {
        if (i == x) continue;
        double l = tl[i], h = th[i];
        if (l > h) std::swap(l, h);
        if (l > t0) t0 = l;
        if (h < t1) t1 = h;
    }
    if (t0 > 1.) t0 = 1.;
    if (t1 < 0.) t1 = 0.;
}

template <int dim>
void Clip<dim>::addClippedSegment(std::set<typename GeometryObjectD<dim>::LineSegment>& segments,
                                  DVec p0,
                                  DVec p1) const {
    DVec dp = p1 - p0;
    double tl[2], tu[2];
    for (size_t i = 0; i < 2; ++i) {
        if (dp[i] == 0) {
            if (clipBox.lower[i] <= p0[i] && p0[0] <= clipBox.upper[i]) {
                tl[i] = 0.;
                tu[i] = 1.;
            } else {
                tl[i] = tu[i] = 0.5;
            }
        } else {
            tl[i] = (clipBox.lower[i] - p0[i]) / dp[i];
            tu[i] = (clipBox.upper[i] - p0[i]) / dp[i];
        }
    }
    for (size_t i = 0; i <= dim; ++i) {
        double t0, t1;
        limitT(t0, t1, tl, tu, dim, i);
        if (t1 <= t0) continue;
        DVec q0 = p0 + dp * t0, q1 = p0 + dp * t1;
        if (i != dim) {
            if (dp[i] == 0.) {
                if (p0[i] < clipBox.lower[i]) {
                    q0[i] = q1[i] = clipBox.lower[i];
                    if (q0 != q1) segments.insert(typename GeometryObjectD<dim>::LineSegment(q0, q1));
                } else if (p0[i] > clipBox.upper[i]) {
                    q0[i] = q1[i] = clipBox.upper[i];
                    if (q0 != q1) segments.insert(typename GeometryObjectD<dim>::LineSegment(q0, q1));
                }
            } else {    
                DVec q0l = (q0[i] <= clipBox.lower[i]) ? q0 : p0 + dp * clamp(tl[i], t0, t1),
                     q1l = (q1[i] <= clipBox.lower[i]) ? q1 : p0 + dp * clamp(tl[i], t0, t1);
                q0l[i] = q1l[i] = clipBox.lower[i];
                if (q0l != q1l) segments.insert(typename GeometryObjectD<dim>::LineSegment(q0l, q1l));
                DVec q0u = (q0[i] >= clipBox.upper[i]) ? q0 : p0 + dp * clamp(tu[i], t0, t1),
                     q1u = (q1[i] >= clipBox.upper[i]) ? q1 : p0 + dp * clamp(tu[i], t0, t1);
                q0u[i] = q1u[i] = clipBox.upper[i];
                if (q0u != q1u) segments.insert(typename GeometryObjectD<dim>::LineSegment(q0u, q1u));
            }
        } else {
            if (q0 != q1) segments.insert(typename GeometryObjectD<dim>::LineSegment(q0, q1));
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
