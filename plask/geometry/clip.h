#ifndef PLASK__GEOMETRY_CLIP_H
#define PLASK__GEOMETRY_CLIP_H

#include "transform.h"

namespace plask {

/**
 * Represent geometry object equal to its child clipped to given box.
 * @ingroup GEOMETRY_OBJ
 */
template <int dim>
struct PLASK_API Clip: public GeometryObjectTransform<dim> {

    static const char* NAME;

    virtual std::string getTypeName() const override { return NAME; }

    typedef typename GeometryObjectTransform<dim>::ChildType ChildType;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<dim>::DVec DVec;

    /// Box type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<dim>::Box Box;

    using GeometryObjectTransform<dim>::getChild;

    /**
     * Clip box.
     */
    Box clipBox;

    //Clip(const Clip<dim>& tocpy) = default;

    /**
     * @param child child geometry object to clip
     * @param Clip Clip
     */
    explicit Clip(shared_ptr< GeometryObjectD<dim> > child = shared_ptr< GeometryObjectD<dim> >(), const Box& clipBox = Primitive<dim>::INF_BOX)
        : GeometryObjectTransform<dim>(child), clipBox(clipBox) {}

    explicit Clip(GeometryObjectD<dim>& child, const Box& clipBox = Primitive<dim>::INF_BOX)
        : GeometryObjectTransform<dim>(child), clipBox(clipBox) {}

    virtual shared_ptr<Material> getMaterial(const DVec& p) const override;

    virtual bool contains(const DVec& p) const override;

    using GeometryObjectTransform<dim>::getPathsTo;

    GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const override;

    virtual Box fromChildCoords(const typename ChildType::Box& child_bbox) const override;

    virtual void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path = 0) const override;

    /**
     * Get shallow copy of this.
     * @return shallow copy of this
     */
    shared_ptr<Clip<dim>> copyShallow() const {
         return plask::make_shared<Clip<dim>>(getChild(), clipBox);
    }

    virtual shared_ptr<GeometryObject> shallowCopy() const override;

    /**
     * Get shallow copy of this with diffrent clip box.
     * @param new_clip clip box for new Clip object
     * @return shallow copy of this with clip box equal to @p new_clip
     */
    shared_ptr<Clip<dim>> copyShallow(const Box& new_clip) const {
        return plask::make_shared<Clip<dim>>(getChild(), new_clip);
    }

    protected:

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

};

template <> void Clip<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;
template <> void Clip<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

PLASK_API_EXTERN_TEMPLATE_STRUCT(Clip<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(Clip<3>)

}   // namespace plask

#endif // PLASK__GEOMETRY_CLIP_H
