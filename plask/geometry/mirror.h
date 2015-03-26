#ifndef PLASK__GEOMETRY_MIRROR_H
#define PLASK__GEOMETRY_MIRROR_H

#include "transform.h"

//Flip - odbicie
//Mirror - odbicie i klonowanie

namespace plask {

/**
 * Represent geometry object equal to mirror reflection of its child.
 * @tparam dim
 * @ingroup GEOMETRY_OBJ
 */
template <int dim>
struct PLASK_API Flip: public GeometryObjectTransform<dim> {

    static constexpr const char* NAME = dim == 2 ?
                ("flip" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D) :
                ("flip" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);

    virtual std::string getTypeName() const override;

    typedef typename GeometryObjectTransform<dim>::ChildType ChildType;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<dim>::DVec DVec;

    /// Box type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<dim>::Box Box;

    using GeometryObjectTransform<dim>::getChild;

    /**
     * Constructor.
     * @param flipDir flip direction, 2D or 3D axis number
     * @param child child geometry object, object to reflect
     */
    explicit Flip(typename Primitive<dim>::Direction flipDir, shared_ptr< GeometryObjectD<dim> > child = shared_ptr< GeometryObjectD<dim> >())
        : GeometryObjectTransform<dim>(child), flipDir(flipDir) {}

    /// 2D or 3D axis number
    typename Primitive<dim>::Direction flipDir;

    virtual Box getBoundingBox() const override;

    /**
     * Get fliped version of @p v.
     * @param v vector
     * @return fliped version of @p v
     */
    DVec fliped(DVec v) const { return v.fliped(flipDir); }

    Box fliped(Box res) const { return res.fliped(flipDir); }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const override;

    virtual bool contains(const DVec& p) const override;

    using GeometryObjectTransform<dim>::getPathsTo;

    virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const override;

    virtual Box fromChildCoords(const typename ChildType::Box& child_bbox) const override;

    virtual void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path = 0) const override;

    /**
     * Get shallow copy of this.
     * @return shallow copy of this
     */
    shared_ptr<Flip<dim>> copyShallow() const {
         return shared_ptr<Flip<dim>>(new Flip<dim>(flipDir, getChild()));
    }

    virtual shared_ptr<GeometryObjectTransform<dim>> shallowCopy() const override;

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

};

PLASK_API_EXTERN_TEMPLATE_STRUCT(Flip<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(Flip<3>)

/**
 * Represent geometry object equal to its child with mirror reflection.
 * @tparam dim
 * @ingroup GEOMETRY_OBJ
 */
//TODO add checking of coordinates
template <int dim>
struct PLASK_API Mirror: public GeometryObjectTransform<dim> {

    static constexpr const char* NAME = dim == 2 ?
                ("mirror" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D) :
                ("mirror" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);

    virtual std::string getTypeName() const override;

    typedef typename GeometryObjectTransform<dim>::ChildType ChildType;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<dim>::DVec DVec;

    /// Box type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<dim>::Box Box;

    using GeometryObjectTransform<dim>::getChild;

    /**
     * @param flipDir
     * @param child child geometry object, object to reflect
     */
    explicit Mirror(typename Primitive<dim>::Direction flipDir, shared_ptr< GeometryObjectD<dim> > child = shared_ptr< GeometryObjectD<dim> >())
        : GeometryObjectTransform<dim>(child), flipDir(flipDir) {}

    /// 2D or 3D axis number
    typename Primitive<dim>::Direction flipDir;

    virtual Box getBoundingBox() const override;

    virtual Box getRealBoundingBox() const override;

    DVec flipedIfNeg(DVec v) const {
        return v[flipDir] >= 0 ? v : v.fliped(flipDir);
    }

    void extend(Box& toExt) const {
        toExt.lower[flipDir] = - toExt.upper[flipDir];
    }

    Box extended(Box res) const {
        extend(res);
        return res;
    }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const override;

    virtual bool contains(const DVec& p) const override;

    using GeometryObjectTransform<dim>::getPathsTo;

    virtual Box fromChildCoords(const typename ChildType::Box& child_bbox) const override;

    virtual void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const override;

    virtual void getObjectsToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObject> >& dest, const PathHints* path = 0) const override;

    virtual void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path = 0) const override;

    virtual GeometryObject::Subtree getPathsTo(const GeometryObject& el, const PathHints* path = 0) const override;

    virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const override;

    virtual std::size_t getChildrenCount() const override;

    virtual shared_ptr<GeometryObject> getChildNo(std::size_t child_no) const override;

    virtual std::size_t getRealChildrenCount() const override;

    virtual shared_ptr<GeometryObject> getRealChildNo(std::size_t child_no) const override;

    /**
     * Get shallow copy of this.
     * @return shallow copy of this
     */
    shared_ptr<Mirror<dim>> copyShallow() const {
         return shared_ptr<Mirror<dim>>(new Mirror<dim>(flipDir, getChild()));
    }

    virtual shared_ptr<GeometryObjectTransform<dim>> shallowCopy() const override;

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

};

PLASK_API_EXTERN_TEMPLATE_STRUCT(Mirror<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(Mirror<3>)

}   // namespace plask

#endif // MIRROR_H
