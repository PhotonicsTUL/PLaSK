#ifndef PLASK__GEOMETRY_CONTAINER_TRANS_H
#define PLASK__GEOMETRY_CONTAINER_TRANS_H

#include "container.h"
#include "spatial_index.h"

#include <boost/thread.hpp>
#include <atomic>

namespace plask {

/**
 * Geometry objects container in which every child has an associated aligner.
 * @ingroup GEOMETRY_OBJ
 */
template < int dim >
struct PLASK_API TranslationContainer: public WithAligners<GeometryObjectContainer<dim>, align::AlignerD<dim>> {

    friend struct Lattice;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectContainer<dim>::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectContainer<dim>::Box Box;

    /// Type of this child.
    typedef typename GeometryObjectContainer<dim>::ChildType ChildType;

    /// Type of child aligner.
    typedef align::AlignerD<dim> ChildAligner;

    /// Type of translation geometry object in space of this.
    typedef typename GeometryObjectContainer<dim>::TranslationT TranslationT;

    using GeometryObjectContainer<dim>::children;
    using GeometryObjectContainer<dim>::shared_from_this;

    static const char* NAME;

    TranslationContainer(): cache(nullptr) {}

    TranslationContainer(const TranslationContainer& to_copy)
        : WithAligners<GeometryObjectContainer<dim>, align::AlignerD<dim>>(to_copy) {}

    ~TranslationContainer();

    std::string getTypeName() const override { return NAME; }

    /**
     * Add new child (translated) to end of children vector.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el new child
     * @param aligner
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(shared_ptr<ChildType> el, ChildAligner aligner);

    /**
     * Add new child (translated) to end of children vector.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el new child
     * @param translation trasnalation of child
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(shared_ptr<ChildType> el, const DVec& translation = Primitive<dim>::ZERO_VEC);

    /**
     * Add new child (translated) to end of children vector.
     * @param el new child
     * @param aligner
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(shared_ptr<ChildType> el, ChildAligner aligner) {
        if (el) this->ensureCanHaveAsChild(*el);
        return addUnsafe(el, aligner);
    }

    /**
     * Add new child (translated) to end of children vector.
     * @param el new child
     * @param translation trasnalation of child
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(shared_ptr<ChildType> el, const DVec& translation = Primitive<dim>::ZERO_VEC) {
        if (el) this->ensureCanHaveAsChild(*el);
        return addUnsafe(el, translation);
    }

    /**
     * Insert new child (translated) to end of children vector.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after inserting the new child.
     * @param el new child
     * @param pos insert position
     * @param aligner
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint insertUnsafe(const std::size_t pos, shared_ptr<ChildType> el, ChildAligner aligner);

    /**
     * Insert new child (translated) to end of children vector.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after inserting the new child.
     * @param el new child
     * @param pos insert position
     * @param translation trasnalation of child
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint insertUnsafe(const std::size_t pos, shared_ptr<ChildType> el, const DVec& translation = Primitive<dim>::ZERO_VEC);

    /**
     * Insert new child (translated) to end of children vector.
     * @param el new child
     * @param pos insert position
     * @param aligner
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if inserting the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint insert(const std::size_t pos, shared_ptr<ChildType> el, ChildAligner aligner) {
        if (el) this->ensureCanHaveAsChild(*el);
        return insertUnsafe(pos, el, aligner);
    }

    /**
     * Insert new child (translated) to end of children vector.
     * @param el new child
     * @param pos insert position
     * @param translation trasnalation of child
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if inserting the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint insert(const std::size_t pos, shared_ptr<ChildType> el, const DVec& translation = Primitive<dim>::ZERO_VEC) {
        if (el) this->ensureCanHaveAsChild(*el);
        return insertUnsafe(pos, el, translation);
    }

    //methods overwrite to use cache:
    shared_ptr<Material> getMaterial(const DVec& p) const override {
        return ensureHasCache()->getMaterial(p);
    }

    bool contains(const DVec& p) const override {
        return ensureHasCache()->contains(p);
    }

    GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const override {
        return ensureHasCache()->getPathsAt(this->shared_from_this(), point, all);
    }

    //some methods must be overwrite to invalidate cache:
    void onChildChanged(const GeometryObject::Event& evt) override {
        if (evt.isResize()) invalidateCache();
        WithAligners<GeometryObjectContainer<dim>, align::AlignerD<dim>>::onChildChanged(evt);
    }

    bool removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) override {
        if (WithAligners<GeometryObjectContainer<dim>, align::AlignerD<dim>>::removeIfTUnsafe(predicate)) {
            invalidateCache();
            return true;
        } else
            return false;
    }

    void removeAtUnsafe(std::size_t index) override {
        invalidateCache();
        WithAligners<GeometryObjectContainer<dim>, align::AlignerD<dim>>::removeAtUnsafe(index);
    }

    //virtual void writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const;

    shared_ptr<GeometryObject> shallowCopy() const override;

    shared_ptr<GeometryObject> deepCopy(std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const override;

protected:
    shared_ptr<GeometryObject> changedVersionForChildren(std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const override;

    /// Destroy geometry cache, this should be called when children was changed (it will be rebuild by first operation which use it).
    void invalidateCache();

    /**
     * Construct cache if it not exists and return it.
     * @return non-null pointer to cache
     */
    SpatialIndexNode<dim>* ensureHasCache();

    /**
     * Construct cache if it not exists and return it.
     *
     * This method is thread-safty (can be call by many threads at once, and only first call will build the cache).
     * @return non-null pointer to cache
     */
    SpatialIndexNode<dim>* ensureHasCache() const;

private:
    shared_ptr<TranslationT> newTranslation(const shared_ptr<ChildType>& el, ChildAligner aligner);

    /**
     * Cache which allow to do some geometry operation (like getMaterial) much faster if this container has many children.
     * It is create by first operation which use it (see ensureHasCache method), and destroy on each change of children (see invalidateCache method).
     */
    std::atomic<SpatialIndexNode<dim>*> cache;

    /// Mutex used by const version of ensureHasCache.
    boost::mutex cache_mutex;

};

PLASK_API_EXTERN_TEMPLATE_STRUCT(TranslationContainer<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(TranslationContainer<3>)

} // namespace plask

#endif // PLASK__GEOMETRY_CONTAINER_TRANS_H
