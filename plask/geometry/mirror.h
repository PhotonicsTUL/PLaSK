#ifndef PLASK__GEOMETRY_MIRROR_H
#define PLASK__GEOMETRY_MIRROR_H

#include "transform.h"

//MirrorReflection - odbicie
//MirrorSymetry - odbicie i klonowanie

namespace plask {

/**
 * Represent geometry object equal to its child with mirror reflection.
 * @tparam dim
 * @ingroup GEOMETRY_OBJ
 */
template <int dim>
struct MirrorReflection: public GeometryObjectTransform<dim> {

    static constexpr const char* NAME = dim == 2 ?
                ("mirrorReflection" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D) :
                ("mirrorReflection" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);

    virtual std::string getTypeName() const { return NAME; }

    typedef typename GeometryObjectTransform<dim>::ChildType ChildType;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<dim>::DVec DVec;

    /// Box type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<dim>::Box Box;

    using GeometryObjectTransform<dim>::getChild;

    /**
     * @param child child geometry object, object to reflect
     */
    explicit MirrorReflection(typename Primitive<dim>::Direction flipDir, shared_ptr< GeometryObjectD<dim> > child = shared_ptr< GeometryObjectD<dim> >())
        : GeometryObjectTransform<dim>(child), flipDir(flipDir) {}

    /// 2D or 3D axis number
    typename Primitive<dim>::Direction flipDir;

    virtual Box getBoundingBox() const {
        return fliped(getChild()->getBoundingBox());
    }

    /**
     * Get fliped version of @p v.
     * @param v vector
     * @return fliped version of @p v
     */
    DVec fliped(DVec v) const { return v.fliped(flipDir); }

    Box fliped(Box res) const { return res.fliped(flipDir); }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        return getChild()->getMaterial(fliped(p));
    }

    virtual bool includes(const DVec& p) const {
        return getChild()->includes(fliped(p));
    }

    using GeometryObjectTransform<dim>::getPathsTo;

    virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const {
        return GeometryObject::Subtree::extendIfNotEmpty(this, getChild()->getPathsAt(fliped(point), all));
    }

    virtual void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const;

    virtual void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path = 0) const;

    /**
     * Get shallow copy of this.
     * @return shallow copy of this
     */
    shared_ptr<MirrorReflection<dim>> copyShallow() const {
         return shared_ptr<MirrorReflection<dim>>(new MirrorReflection<dim>(flipDir, getChild()));
    }

    virtual shared_ptr<GeometryObjectTransform<dim>> shallowCopy() const { return copyShallow(); }

    virtual shared_ptr<const GeometryObject> changedVersion(const GeometryObject::Changer& changer, Vec<3, double>* translation = 0) const;

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

};

/**
 * We suppose that getBoundingBoxes().upper[flipDir] > 0
 */
//TODO add checking of coordinates
template <int dim>
struct MirrorSymetry: public GeometryObjectTransform<dim> {

    static constexpr const char* NAME = dim == 2 ?
                ("mirrorSymetry" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D) :
                ("mirrorSymetry" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);

    virtual std::string getTypeName() const { return NAME; }

    typedef typename GeometryObjectTransform<dim>::ChildType ChildType;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<dim>::DVec DVec;

    /// Box type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<dim>::Box Box;

    using GeometryObjectTransform<dim>::getChild;

    /**
     * @param child child geometry object, object to reflect
     */
    explicit MirrorSymetry(typename Primitive<dim>::Direction flipDir, shared_ptr< GeometryObjectD<dim> > child = shared_ptr< GeometryObjectD<dim> >())
        : GeometryObjectTransform<dim>(child), flipDir(flipDir) {}

    /// 2D or 3D axis number
    typename Primitive<dim>::Direction flipDir;

    virtual Box getBoundingBox() const {
        return extended(getChild()->getBoundingBox());
    }

    virtual Box getRealBoundingBox() const {
        return getChild()->getBoundingBox();
    }

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

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        return getChild()->getMaterial(flipedIfNeg(p));
    }

    virtual bool includes(const DVec& p) const {
        return getChild()->includes(flipedIfNeg(p));
    }

    using GeometryObjectTransform<dim>::getPathsTo;

    virtual void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const;

    virtual void getObjectsToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObject> >& dest, const PathHints* path = 0) const;

    virtual void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path = 0) const;

    virtual GeometryObject::Subtree getPathsTo(const GeometryObject& el, const PathHints* path = 0) const;

    virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const;

    virtual std::size_t getChildrenCount() const;

    virtual shared_ptr<GeometryObject> getChildNo(std::size_t child_no) const;

    virtual std::size_t getRealChildrenCount() const {
        return GeometryObjectTransform<dim>::getChildrenCount();
    }

    virtual shared_ptr<GeometryObject> getRealChildNo(std::size_t child_no) const {
        return GeometryObjectTransform<dim>::getChildNo(child_no);
    }

    /**
     * Get shallow copy of this.
     * @return shallow copy of this
     */
    shared_ptr<MirrorSymetry<dim>> copyShallow() const {
         return shared_ptr<MirrorSymetry<dim>>(new MirrorSymetry<dim>(flipDir, getChild()));
    }

    virtual shared_ptr<GeometryObjectTransform<dim>> shallowCopy() const { return copyShallow(); }

    virtual shared_ptr<const GeometryObject> changedVersion(const GeometryObject::Changer& changer, Vec<3, double>* translation = 0) const;

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

};

}   // namespace plask

#endif // MIRROR_H
