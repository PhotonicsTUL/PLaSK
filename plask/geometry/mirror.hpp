#ifndef PLASK__GEOMETRY_MIRROR_H
#define PLASK__GEOMETRY_MIRROR_H

#include "transform.hpp"

// Flip - odbicie
// Mirror - odbicie i klonowanie

namespace plask {

/**
 * Represent geometry object equal to mirror reflection of its child.
 * @tparam dim
 * @ingroup GEOMETRY_OBJ
 */
template <int dim> struct PLASK_API Flip : public GeometryObjectTransform<dim> {
    typedef GeometryObjectTransform<dim> BaseClass;

    static const char* NAME;

    std::string getTypeName() const override;

    typedef typename BaseClass::ChildType ChildType;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename BaseClass::DVec DVec;

    /// Box type in space on this, rectangle in space with dim number of dimensions.
    typedef typename BaseClass::Box Box;

    using BaseClass::getChild;

    /**
     * Constructor.
     * @param flipDir flip direction, 2D or 3D axis number
     * @param child child geometry object, object to reflect
     */
    explicit Flip(typename Primitive<dim>::Direction flipDir,
                  shared_ptr<GeometryObjectD<dim>> child = shared_ptr<GeometryObjectD<dim>>())
        : BaseClass(child), flipDir(flipDir) {}

    /// 2D or 3D axis number
    typename Primitive<dim>::Direction flipDir;

    /**
     * Get flipped version of @p v.
     * @param v vector
     * @return flipped version of @p v
     */
    DVec flipped(DVec v) const { return v.flipped(flipDir); }

    Box flipped(Box res) const { return res.flipped(flipDir); }

    shared_ptr<Material> getMaterial(const DVec& p) const override;

    bool contains(const DVec& p) const override;

    using GeometryObjectTransform<dim>::getPathsTo;

    GeometryObject::Subtree getPathsAt(const DVec& point, bool all = false) const override;

    Box fromChildCoords(const typename ChildType::Box& child_bbox) const override;

    void getPositionsToVec(const GeometryObject::Predicate& predicate,
                           std::vector<DVec>& dest,
                           const PathHints* path = 0) const override;

    /**
     * Get shallow copy of this.
     * @return shallow copy of this
     */
    shared_ptr<Flip<dim>> copyShallow() const { return shared_ptr<Flip<dim>>(new Flip<dim>(flipDir, getChild())); }

    shared_ptr<GeometryObject> shallowCopy() const override;

    void addPointsAlongToSet(std::set<double>& points,
                             Primitive<3>::Direction direction,
                             unsigned max_steps,
                             double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<dim>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;
};

PLASK_API_EXTERN_TEMPLATE_STRUCT(Flip<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(Flip<3>)

/**
 * Represent geometry object equal to its child with mirror reflection.
 * @tparam dim
 * @ingroup GEOMETRY_OBJ
 */
// TODO add checking of coordinates
template <int dim> struct PLASK_API Mirror : public GeometryObjectTransform<dim> {
    typedef GeometryObjectTransform<dim> BaseClass;

    static const char* NAME;

    std::string getTypeName() const override;

    typedef typename BaseClass::ChildType ChildType;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename BaseClass::DVec DVec;

    /// Box type in space on this, rectangle in space with dim number of dimensions.
    typedef typename BaseClass::Box Box;

    using BaseClass::getChild;

    /**
     * @param flipDir
     * @param child child geometry object, object to reflect
     */
    explicit Mirror(typename Primitive<dim>::Direction flipDir,
                    shared_ptr<GeometryObjectD<dim>> child = shared_ptr<GeometryObjectD<dim>>())
        : BaseClass(child), flipDir(flipDir) {}

    /// 2D or 3D axis number
    typename Primitive<dim>::Direction flipDir;

    Box getBoundingBox() const override;

    Box getRealBoundingBox() const override;

    DVec flipped(DVec v) const { return v.flipped(flipDir); }

    DVec flippedIfNeg(DVec v) const { return v[flipDir] >= 0 ? v : v.flipped(flipDir); }

    void extend(Box& toExt) const { toExt.lower[flipDir] = -toExt.upper[flipDir]; }

    Box extended(Box res) const {
        extend(res);
        return res;
    }

    shared_ptr<Material> getMaterial(const DVec& p) const override;

    bool contains(const DVec& p) const override;

    using GeometryObjectTransform<dim>::getPathsTo;

    Box fromChildCoords(const typename ChildType::Box& child_bbox) const override;

    void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate,
                               std::vector<Box>& dest,
                               const PathHints* path = 0) const override;

    void getObjectsToVec(const GeometryObject::Predicate& predicate,
                         std::vector<shared_ptr<const GeometryObject>>& dest,
                         const PathHints* path = 0) const override;

    void getPositionsToVec(const GeometryObject::Predicate& predicate,
                           std::vector<DVec>& dest,
                           const PathHints* path = 0) const override;

    GeometryObject::Subtree getPathsTo(const GeometryObject& el, const PathHints* path = 0) const override;

    GeometryObject::Subtree getPathsAt(const DVec& point, bool all = false) const override;

    std::size_t getChildrenCount() const override;

    shared_ptr<GeometryObject> getChildNo(std::size_t child_no) const override;

    std::size_t getRealChildrenCount() const override;

    shared_ptr<GeometryObject> getRealChildNo(std::size_t child_no) const override;

    /**
     * Get shallow copy of this.
     * @return shallow copy of this
     */
    shared_ptr<Mirror<dim>> copyShallow() const {
        return shared_ptr<Mirror<dim>>(new Mirror<dim>(flipDir, this->_child));
    }

    void addPointsAlongToSet(std::set<double>& points,
                             Primitive<3>::Direction direction,
                             unsigned max_steps,
                             double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<dim>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;

    shared_ptr<GeometryObject> shallowCopy() const override;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;
};

PLASK_API_EXTERN_TEMPLATE_STRUCT(Mirror<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(Mirror<3>)

}  // namespace plask

#endif  // MIRROR_H
