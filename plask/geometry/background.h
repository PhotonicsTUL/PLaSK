#ifndef PLASK__GEOMETRY_BACKGROUND_H
#define PLASK__GEOMETRY_BACKGROUND_H

#include "transform.h"

namespace plask {

/**
 * \brief Container which provides material outside of its bounding box.
 *
 * This container holds only one element. However all for material queries (through \ref getMaterial)
 * it considers the points outside of its bounding box as if they were ocated exactly at the edges
 * of the bounding box. This allows to create infinite egde areas filled with some particular material.
 */
template <int dim>
struct Background: public GeometryElementTransform<dim>{

    typedef typename GeometryElementTransform<dim>::ChildType ChildType;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementTransform<dim>::DVec DVec;

    /// Box type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementTransform<dim>::Box Box;

    using GeometryElementTransform<dim>::getChild;

    /// Type of extensions for infinite stacks
    enum ExtendType {
        EXTEND_NONE = 0,
        EXTEND_VERTICAL = 1,
        EXTEND_TRAN = 2,
        EXTEND_LON = 4,
        EXTEND_HORIZONTAL = 6,
        EXTEND_ALL = 7
    };

  private:

      ExtendType _extend;

  public:

    /**
     * \param extend direction of extension
     */
    explicit Background(ExtendType extend) :
        GeometryElementTransform<dim>(shared_ptr<GeometryElementD<dim>>()) {
        setExtend(extend);
    }

    /**
     * \param child hold geometry element
     * \param extend direction of extension
     */
    explicit Background(shared_ptr<ChildType> child, ExtendType extend) :
        GeometryElementTransform<dim>(child) {
        setExtend(extend);
    }

    /**
     * \param child hold geometry element
     * \param extend direction of extension
     */
    explicit Background(ChildType& child, ExtendType extend) :
        GeometryElementTransform<dim>(child) {
        setExtend(extend);
    }

    /// \return direction of the extension
    inline ExtendType getExtend() { return _extend; }

    /** Set the extend
     * \param extend new extend
     */
    inline void setExtend(ExtendType extend);


    virtual bool inside(const DVec& p) const { return getChild()->inside(p); }

    virtual bool intersect(const Box& area) const { return getChild()->intersect(area); }

    virtual Box getBoundingBox() const { return getChild()->getBoundingBox(); } //shouldn't this be infinite?

    virtual void getLeafsBoundingBoxesToVec(std::vector<Box>& dest, const PathHints* path = 0) const {
        getChild()->getLeafsBoundingBoxesToVec(dest, path);
    }

    virtual std::vector<std::tuple<shared_ptr<const GeometryElement>, DVec>> getLeafsWithTranslations() const {
        return getChild()->getLeafsWithTranslations();
    }

    virtual shared_ptr<GeometryElementTransform<dim>> shallowCopy() const {
        return make_shared<Background<dim>>(getChild(), _extend);
    }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const;

};

template<>
inline void Background<2>::setExtend(typename Background<2>::ExtendType extend) {
    if (extend == EXTEND_LON) throw BadInput("Background<2>", "EXTEND_LON not allowed for 2D background.");
    _extend = extend;
}

template<>
inline void Background<3>::setExtend(typename Background<3>::ExtendType extend) {
    _extend = extend;
}


} // namespace plask
#endif // PLASK__GEOMETRY_BACKGROUND_H