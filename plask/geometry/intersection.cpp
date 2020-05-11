#include "intersection.h"

#include "../manager.h"
#include "reader.h"

#define PLASK_INTERSECTION2D_NAME ("intersection" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D)
#define PLASK_INTERSECTION3D_NAME ("intersection" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D)

namespace plask {

template <int dim>
const char* Intersection<dim>::NAME = dim == 2 ? PLASK_INTERSECTION2D_NAME : PLASK_INTERSECTION3D_NAME;

template <int dim>
shared_ptr<Material> Intersection<dim>::getMaterial(const typename Intersection<dim>::DVec& p) const {
    return (this->hasChild() && this->inEnvelope(p)) ? this->_child->getMaterial(p) : shared_ptr<Material>();
}

template <int dim> bool Intersection<dim>::contains(const typename Intersection<dim>::DVec& p) const {
    return this->hasChild() && (inEnvelope(p)) && this->_child->contains(p);
}

template <int dim>
GeometryObject::Subtree Intersection<dim>::getPathsAt(const typename Intersection<dim>::DVec& point, bool all) const {
    if (this->hasChild() && inEnvelope(point))
        return GeometryObject::Subtree::extendIfNotEmpty(this, this->_child->getPathsAt(point, all));
    else
        return GeometryObject::Subtree();
}

template <int dim>
typename Intersection<dim>::Box Intersection<dim>::fromChildCoords(
    const typename Intersection<dim>::ChildType::Box& child_bbox) const {
    if (envelope) return envelope->getBoundingBox().intersection(child_bbox);
    else
        return child_bbox;
}

template <int dim>
void Intersection<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate,
                                              std::vector<Box>& dest,
                                              const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->getBoundingBox());
        return;
    }
    if (!this->hasChild()) return;
    std::vector<Box> result = this->_child->getBoundingBoxes(predicate, path);
    dest.reserve(dest.size() + result.size());
    if (envelope) {
        Box clipBox = envelope->getBoundingBox();
        for (Box& r : result) {
            r.makeIntersection(clipBox);
            dest.push_back(r);
        }
    } else {
        for (Box& r : result) dest.push_back(r);
    }
}

template <int dim>
void Intersection<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate,
                                          std::vector<DVec>& dest,
                                          const PathHints* path) const {
    this->_getNotChangedPositionsToVec(this, predicate, dest, path);
}

template <int dim> shared_ptr<GeometryObject> Intersection<dim>::shallowCopy() const { return copyShallow(); }

template <int dim>
void Intersection<dim>::addPointsAlong(std::set<double>& points,
                                       Primitive<3>::Direction direction,
                                       unsigned max_steps,
                                       double min_step_size) const {
    if (this->_child) {
        if (!envelope) {
            this->_child->addPointsAlong(points, direction, this->max_steps ? this->max_steps : max_steps,
                                         this->min_step_size ? this->min_step_size : min_step_size);
            return;
        }
        std::set<double> child_points;
        this->_child->addPointsAlong(child_points, direction, this->max_steps ? this->max_steps : max_steps,
                                     this->min_step_size ? this->min_step_size : min_step_size);
        auto clipbox = envelope->getBoundingBox();
        auto bbox = this->getBoundingBox();
        points.insert(bbox.lower[int(direction) - (3 - dim)]);
        for (double p : child_points) {
            if (clipbox.lower[int(direction) - (3 - dim)] <= p && p <= clipbox.upper[int(direction) - (3 - dim)])
                points.insert(p);
        }
        points.insert(bbox.upper[int(direction) - (3 - dim)]);
    }
}

template <int dim>
void Intersection<dim>::addLineSegmentsToSet(std::set<typename GeometryObjectD<dim>::LineSegment>& segments,
                                             unsigned max_steps,
                                             double min_step_size) const {
    if (this->_child) {
        if (!this->envelope) {
            this->_child->addLineSegmentsToSet(segments, this->max_steps ? this->max_steps : max_steps,
                                               this->min_step_size ? this->min_step_size : min_step_size);
            return;
        }
        std::set<typename GeometryObjectD<dim>::LineSegment> child_segments;
        this->_child->addLineSegmentsToSet(child_segments, this->max_steps ? this->max_steps : max_steps,
                                           this->min_step_size ? this->min_step_size : min_step_size);
        for (const auto& s : child_segments) {
            if (envelope->contains(s.p0()) || envelope->contains(s.p1())) segments.insert(s);
        }
    }
}

template <int dim>
void Intersection<dim>::writeXMLChildren(XMLWriter::Element& dest_xml_object,
                                         GeometryObject::WriteXMLCallback& write_cb,
                                         const AxisNames& axes) const {
    if (this->hasChild()) {
        this->_child->writeXML(dest_xml_object, write_cb, axes);
        if (envelope) envelope->writeXML(dest_xml_object, write_cb, axes);
    }
}

template <int dim> shared_ptr<GeometryObject> read_Intersection(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(
        reader, dim == 2 ? PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D : PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    shared_ptr<Intersection<dim>> intersection = plask::make_shared<Intersection<dim>>();
    if (reader.source.requireNext(reader.manager.draft ? (XMLReader::NODE_ELEMENT | XMLReader::NODE_ELEMENT_END)
                                                       : XMLReader::NODE_ELEMENT) == XMLReader::NODE_ELEMENT) {
        intersection->setChild(reader.readObject<typename Intersection<dim>::ChildType>());
        if (reader.source.requireTagOrEnd()) {
            GeometryReader::RevertMaterialsAreRequired enableShapeOnlyMode(reader, false);
            intersection->envelope = reader.readObject<typename Intersection<dim>::ChildType>();
            reader.source.requireTagEnd();
        }
    }
    return intersection;
}

static GeometryReader::RegisterObjectReader Intersection2D_reader(PLASK_INTERSECTION2D_NAME, read_Intersection<2>);
static GeometryReader::RegisterObjectReader Intersection3D_reader(PLASK_INTERSECTION3D_NAME, read_Intersection<3>);

template struct PLASK_API Intersection<2>;
template struct PLASK_API Intersection<3>;

}  // namespace plask
