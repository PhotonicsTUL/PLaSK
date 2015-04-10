// #ifndef PLASK__GEOMETRY_LATTICE_H
// #define PLASK__GEOMETRY_LATTICE_H
//
// #include "container.h"
// #include "container.h"
//
// namespace plask {
//
// /// Sequence container that repeats its children over a line shifted by a vector
// template <int dim>
// struct PLASK_API Repeat: public GeometryObjectTransform<3> {
//
//     /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
//     typedef typename GeometryObjectTransform<3>::DVec DVec;
//
//     /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
//     typedef typename GeometryObjectTransform<3>::Box Box;
//
//     /// Type of this child.
//     typedef typename GeometryObjectTransform<3>::ChildType ChildType;
//
//     /// Type of translation geometry element in space of this.
//     typedef typename GeometryObjectTransform<3>::TranslationT TranslationT;
//
//     /// Type of the vector holding container children
//     typedef typename GeometryObjectTransform<3>::TranslationVector TranslationVector;
//
//     using GeometryObjectTransform<dim>::getChild;
//
//     /// Translation vector for each repetition
//     DVec shift;
//
//     /// Number of repetitions
//     unsigned repeat_count;
//
//     virtual Box getBoundingBox() const override;
//
//     virtual Box getRealBoundingBox() const override;
//
//     virtual void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const override;
//
//     virtual void getObjectsToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObject> >& dest, const PathHints* path = 0) const override;
//
//     virtual void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path = 0) const override;
//
//     // void extractToVec(const GeometryObject::Predicate &predicate, std::vector< shared_ptr<const GeometryObjectD<dim> > >& dest, const PathHints *path = 0) const;
//
//     virtual GeometryObject::Subtree getPathsTo(const GeometryObject& el, const PathHints* path = 0) const override;
//
//     virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const override;
//
//     virtual bool contains(const DVec& p) const override;
//
//     virtual shared_ptr<Material> getMaterial(const DVec& p) const override;
//
//     virtual std::size_t getChildrenCount() const { return children.size() * repeat_count; }
//
//     virtual shared_ptr<GeometryObject> getChildNo(std::size_t child_no) const;
//
//     virtual std::size_t getRealChildrenCount() const override;
//
//     virtual shared_ptr<GeometryObject> getRealChildNo(std::size_t child_no) const override;
//
//     unsigned getRepeatCount() const { return repeat_count; }
//
//     void setRepeatCount(unsigned new_repeat_count) {
//         if (repeat_count == new_repeat_count) return;
//         repeat_count = new_repeat_count;
//         this->fireChildrenChanged();    //TODO should this be called? or simple change?
//
// };
//
// /// Lattice container that arranges its children in two-dimensional lattice
// struct PLASK_API Lattice: public GeometryObjectTransform<3> {
//
//     /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
//     typedef typename GeometryObjectTransform<3>::DVec DVec;
//
//     /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
//     typedef typename GeometryObjectTransform<3>::Box Box;
//
//     /// Type of this child.
//     typedef typename GeometryObjectTransform<3>::ChildType ChildType;
//
//     /// Type of translation geometry element in space of this.
//     typedef typename GeometryObjectTransform<3>::TranslationT TranslationT;
//
//     /// Type of the vector holding container children
//     typedef typename GeometryObjectTransform<3>::TranslationVector TranslationVector;
//
//     using GeometryObjectTransform<dim>::getChild;
//
//     /// Lattice vectors
//     DVec vec0, vec1;
//
//     /// Create a lattice
//
// };
//
// } // namespace plask
//
// #endif // PLASK__GEOMETRY_LATTICE_H
