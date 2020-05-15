#ifndef PLASK__GEOMETRY_LATTICE_H
#define PLASK__GEOMETRY_LATTICE_H

#include <cmath>

#include "../log/log.h"
#include "transform.h"

#include "translation_container.h"  // used by lattiece

namespace plask {

/// Sequence container that repeats its child over a line shifted by a vector.
/// You can consider this as a one-dimentional lattice
template <int dim> struct PLASK_API ArrangeContainer : public GeometryObjectTransform<dim> {
    typedef GeometryObjectTransform<dim> BaseClass;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename BaseClass::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename BaseClass::Box Box;

    /// Type of this child.
    typedef typename BaseClass::ChildType ChildType;

    using BaseClass::getChild;

  protected:
    using BaseClass::_child;

    /// Translation vector for each repetition
    DVec translation;

    /// Number of repetitions
    unsigned repeat_count;

    /// Reduce vector to the first repetition
    std::pair<int, int> bounds(const DVec& vec) const;

    /// Warm if warn_overlapping is set and item bboxes overlap.
    void warmOverlaping() const;

  public:
    /// Should the user be warned about overlapping bounding boxes?
    bool warn_overlapping;

    ArrangeContainer()
        : GeometryObjectTransform<dim>(shared_ptr<ChildType>()),
          translation(Primitive<dim>::ZERO_VEC),
          repeat_count(0),
          warn_overlapping(true) {}

    /// Create a repeat object.
    /// \param item Object to repeat.
    /// \param step Vector, by which each repetition is shifted from the previous one.
    /// \param count Number of repetitions.
    ArrangeContainer(const shared_ptr<ChildType>& child, const DVec& step, unsigned repeat, bool warn = true)
        : GeometryObjectTransform<dim>(child), translation(step), repeat_count(repeat), warn_overlapping(warn) {
        warmOverlaping();
    }

    static const char* NAME;

    std::string getTypeName() const override { return NAME; }

    Box getBoundingBox() const override;

    Box getRealBoundingBox() const override;

    void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate,
                               std::vector<Box>& dest,
                               const PathHints* path = 0) const override;

    void getObjectsToVec(const GeometryObject::Predicate& predicate,
                         std::vector<shared_ptr<const GeometryObject>>& dest,
                         const PathHints* path = 0) const override;

    void getPositionsToVec(const GeometryObject::Predicate& predicate,
                           std::vector<DVec>& dest,
                           const PathHints* path = 0) const override;

    bool contains(const DVec& p) const override;

    shared_ptr<Material> getMaterial(const DVec& p) const override;

    std::size_t getChildrenCount() const override;

    shared_ptr<GeometryObject> getChildNo(std::size_t child_no) const override;

    std::size_t getRealChildrenCount() const override;

    shared_ptr<GeometryObject> getRealChildNo(std::size_t child_no) const override;

    GeometryObject::Subtree getPathsAt(const DVec& point, bool all = false) const override;

    shared_ptr<GeometryObject> shallowCopy() const override;

    Box fromChildCoords(const typename ChildType::Box& child_bbox) const override;

    unsigned getRepeatCount() const { return repeat_count; }

    void setRepeatCount(unsigned new_repeat_count) {
        if (repeat_count == new_repeat_count) return;
        repeat_count = new_repeat_count;
        this->fireChildrenChanged();
    }

    DVec getTranslation() const { return translation; }

    void setTranslation(DVec new_translation) {
        if (translation == new_translation) return;
        translation = new_translation;
        warmOverlaping();
        this->fireChildrenChanged();
    }

    void addPointsAlong(std::set<double>& points,
                        Primitive<3>::Direction direction,
                        unsigned max_steps,
                        double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<dim>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;
};

template <> void ArrangeContainer<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;
template <> void ArrangeContainer<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

PLASK_API_EXTERN_TEMPLATE_STRUCT(ArrangeContainer<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(ArrangeContainer<3>)

/// Lattice container that arranges its children in two-dimensional lattice.
struct PLASK_API Lattice : public GeometryObjectTransform<3> {
    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<3>::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectTransform<3>::Box Box;

    /// Type of this child.
    typedef typename GeometryObjectTransform<3>::ChildType ChildType;

    using GeometryObjectTransform<3>::getChild;

    static const char* NAME;

    /// Basis vectors
    DVec vec0, vec1;

    shared_ptr<TranslationContainer<3>> container;

    /**
     * Vector of closed polygons, each consist of number of successive verticles, one side is between last and first
     * vertex. These polygons are xored. Sides must not cross each other.
     */
    std::vector<std::vector<Vec<2, int>>> segments;  // TODO checking somewhere if sides do not cross each other

    std::string getTypeName() const override { return NAME; }

    /**
     * Create a lattice object.
     * @param child Object to repeat.
     * @param vec0, vec1 basis vectors
     */
    Lattice(const shared_ptr<ChildType>& child = shared_ptr<ChildType>(),
            const DVec& vec0 = Primitive<3>::ZERO_VEC,
            const DVec& vec1 = Primitive<3>::ZERO_VEC)
        : GeometryObjectTransform<3>(child),
          vec0(vec0),
          vec1(vec1),
          container(plask::make_shared<TranslationContainer<3>>()) {}

    shared_ptr<Material> getMaterial(const DVec& p) const override { return container->getMaterial(p); }

    bool contains(const DVec& p) const override { return container->contains(p); }

    // some methods must be overwrite to invalidate cache:
    void onChildChanged(const GeometryObject::Event& evt) override {
        // if (evt.isResize()) invalidateCache();
        container->onChildChanged(evt);  // force early cache rebuilding
        GeometryObjectTransform<3>::onChildChanged(evt);
    }

    /*void rebuildCache() {
        //TODO
    }*/

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

    void writeXMLChildren(XMLWriter::Element& dest_xml_object,
                          WriteXMLCallback& write_cb,
                          const AxisNames& axes) const override;

    Box getBoundingBox() const override { return container->getBoundingBox(); }

    Box getRealBoundingBox() const override { return container->getRealBoundingBox(); }

    // using GeometryObjectTransform<3>::getPathsTo;

    // Box fromChildCoords(const typename ChildType::Box& child_bbox) const override;

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

    // GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const override;

    GeometryObject::Subtree getPathsAt(const DVec& point, bool all = false) const override {
        return container->ensureHasCache()->getPathsAt(this->shared_from_this(), point, all);
    }

    std::size_t getChildrenCount() const override { return container->getChildrenCount(); }

    shared_ptr<GeometryObject> getChildNo(std::size_t child_no) const override {
        return container->getChildNo(child_no);
    }

    std::size_t getRealChildrenCount() const override { return GeometryObjectTransform<3>::getRealChildrenCount(); }

    shared_ptr<GeometryObject> getRealChildNo(std::size_t child_no) const override {
        return GeometryObjectTransform<3>::getRealChildNo(child_no);
    }

    shared_ptr<Lattice> copyShallow() const {
        auto result = plask::make_shared<Lattice>(*this);
        result->container = plask::make_shared<TranslationContainer<3>>(*result->container);
        return result;
    }

    shared_ptr<GeometryObject> shallowCopy() const override { return copyShallow(); }

    void addPointsAlong(std::set<double>& points,
                        Primitive<3>::Direction direction,
                        unsigned max_steps,
                        double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;

    // probably unused
    Box fromChildCoords(const typename ChildType::Box& child_bbox) const override { return child_bbox; }

    void setSegments(std::vector<std::vector<Vec<2, int>>> new_segments);

    // protected:

    /// Use segments, vec0, vec1 to refill container.
    void refillContainer();
};

}  // namespace plask

#endif  // PLASK__GEOMETRY_LATTICE_H
