#ifndef PLASK__GEOMETRY_INTERSECTION_H
#define PLASK__GEOMETRY_INTERSECTION_H

#include "transform.h"

namespace plask {

/**
 * Represent geometry object equal to intersection of the children.
 * First child is a source of materials.
 * @ingroup GEOMETRY_OBJ
 */
template <int dim>
struct PLASK_API Intersection: public GeometryObjectTransform<dim> {

    static constexpr const char* NAME = dim == 2 ?
                ("intersection" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D) :
                ("intersection" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);

    virtual std::string getTypeName() const override { return NAME; }

    typedef typename GeometryObjectTransform<dim>::ChildType ChildType;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<dim>::DVec DVec;

    /// Box type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<dim>::Box Box;

    using GeometryObjectTransform<dim>::getChild;

    /**
     * Cliping shape.
     */
    shared_ptr<ChildType> envelope;

    shared_ptr<ChildType> getEnvelope() const { return envelope; }

    void setEnvelope(shared_ptr<ChildType> clipShape) {
        if (this->envelope == clipShape) return;
        this->envelope = clipShape;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    //Intersection(const Intersection<dim>& tocpy) = default;

    /**
     * @param child child geometry object to Intersection
     * @param clipShape shape to which the child will be cliped, can have undefined materials in leafs
     */
    explicit Intersection(shared_ptr< GeometryObjectD<dim> > child = shared_ptr< GeometryObjectD<dim> >(), shared_ptr< GeometryObjectD<dim> > clipShape = shared_ptr< GeometryObjectD<dim> >())
        : GeometryObjectTransform<dim>(child), envelope(clipShape) {}

    /**
     * @param child child geometry object to Intersection
     * @param clipShape shape to which the child will be clipped, can have undefined materials in leafs
     */
    explicit Intersection(GeometryObjectD<dim>& child, shared_ptr< GeometryObjectD<dim> > clipShape = shared_ptr< GeometryObjectD<dim> >())
        : GeometryObjectTransform<dim>(child), envelope(clipShape) {}

    virtual Box getBoundingBox() const override;

    virtual shared_ptr<Material> getMaterial(const DVec& p) const override;

    virtual bool contains(const DVec& p) const override;

    using GeometryObjectTransform<dim>::getPathsTo;

    GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const override;

    virtual Box fromChildCoords(const typename ChildType::Box& child_bbox) const override;

    virtual void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const override;

    virtual void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path = 0) const override;

    /**
     * Get shallow copy of this.
     * @return shallow copy of this
     */
    shared_ptr<Intersection<dim>> copyShallow() const {
         return make_shared<Intersection<dim>>(getChild(), envelope);
    }

    virtual shared_ptr<GeometryObjectTransform<dim>> shallowCopy() const override;

    /**
     * Get shallow copy of this with diffrent clipping shape.
     * @param clipShape shape to which the child will of result will be clipped, can have undefined materials in leafs
     * @return shallow copy of this with clipping shape equals to @p clipShape
     */
    shared_ptr<Intersection<dim>> copyShallow(shared_ptr< GeometryObjectD<dim> > clipShape) const {
        return make_shared<Intersection<dim>>(getChild(), clipShape);
    }

  protected:

    void writeXMLChildren(XMLWriter::Element& dest_xml_object, GeometryObject::WriteXMLCallback& write_cb, const AxisNames &axes) const override;

};

PLASK_API_EXTERN_TEMPLATE_STRUCT(Intersection<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(Intersection<3>)

}   // namespace plask

#endif // PLASK__GEOMETRY_INTERSECTION_H
