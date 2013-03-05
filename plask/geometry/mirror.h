#ifndef PLASK__GEOMETRY_MIRROR_H
#define PLASK__GEOMETRY_MIRROR_H

#include "transform.h"

//MirrorReflection - odbicie
//MirrorSymetry - odbicie i klonowanie

namespace plask {

/**
 * Represent geometry object equal to its child with mirror reflection.
 * @tparam dim
 * @tparam flipDir 2D or 3D axis number
 * @ingroup GEOMETRY_OBJ
 */
template <int dim, typename Primitive<dim>::Direction flipDir>
class MirrorReflection: public GeometryObjectTransform<dim> {
    
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
    explicit MirrorReflection(shared_ptr< GeometryObjectD<dim> > child = shared_ptr< GeometryObjectD<dim> >())
        : GeometryObjectTransform<dim>(child) {}

    virtual Box getBoundingBox() const {
        return fliped(getChild()->getBoundingBox());
    }
    
    /**
     * Get fliped version of @p v.
     * @param v vector
     * @return fliped version of @p v
     */
    static DVec fliped(DVec v) { v[flipDir] = -v[flipDir]; return v; }
    
    static Box fliped(Box res) {
        double temp = res.lower[flipDir];
        res.lower[flipDir] = - res.upper[flipDir];
        res.upper[flipDir] = - temp;
        return res;
    }

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
    shared_ptr<MirrorReflection<dim, flipDir>> copyShallow() const {
         return shared_ptr<MirrorReflection<dim, flipDir>>(new MirrorReflection<dim, flipDir>(getChild()));
    }

    virtual shared_ptr<GeometryObjectTransform<dim>> shallowCopy() const { return copyShallow(); }

    virtual shared_ptr<const GeometryObject> changedVersion(const GeometryObject::Changer& changer, Vec<3, double>* translation = 0) const;

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

};


}   // namespace plask

#endif // MIRROR_H
