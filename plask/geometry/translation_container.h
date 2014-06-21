#ifndef PLASK__GEOMETRY_CONTAINER_TRANS_H
#define PLASK__GEOMETRY_CONTAINER_TRANS_H

#include "container.h"

#include <boost/thread.hpp>
#include <atomic>

namespace plask {

/**
 * Base class for cache and nodes of cache.
 *
 * It has some methods similar to this in GeometryObjectContainer API and is used by TranslationContainer.
 */
template <int DIMS>
struct CacheNode {

    virtual shared_ptr<Material> getMaterial(const Vec<DIMS>& p) const = 0;

    virtual bool contains(const Vec<DIMS>& p) const = 0;

    virtual GeometryObject::Subtree getPathsAt(shared_ptr<const GeometryObject> caller, const Vec<DIMS> &point, bool all) const = 0;

    virtual ~CacheNode() {}
};

template <int dim>
using TranslationContainerChildAligner = typename chooseType<dim-2,
        align::Aligner<Primitive<3>::DIRECTION_TRAN, Primitive<3>::DIRECTION_VERT>,
        align::Aligner<>
    >::type;

/**
 * Geometry objects container in which every child has an associated translation vector.
 * @ingroup GEOMETRY_OBJ
 */
template < int dim >
struct PLASK_API TranslationContainer: public WithAligners<GeometryObjectContainer<dim>, TranslationContainerChildAligner<dim>> {

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectContainer<dim>::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectContainer<dim>::Box Box;

    /// Type of this child.
    typedef typename GeometryObjectContainer<dim>::ChildType ChildType;

    /// Type of child aligner.
    typedef TranslationContainerChildAligner<dim> ChildAligner;

    /// Type of translation geometry object in space of this.
    typedef typename GeometryObjectContainer<dim>::TranslationT TranslationT;

    using GeometryObjectContainer<dim>::children;
    using GeometryObjectContainer<dim>::shared_from_this;

    static constexpr const char* NAME = dim == 2 ?
                ("container" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D) :
                ("container" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);

    TranslationContainer(): cache(nullptr) {}

    ~TranslationContainer();

    virtual std::string getTypeName() const { return NAME; }

    /**
     * Add new child (translated) to end of children vector.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el new child
     * @param aligner
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(shared_ptr<ChildType> el, ChildAligner aligner);

    /**
     * Add new child (trasnlated) to end of children vector.
     * @param el new child
     * @param aligner
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(shared_ptr<ChildType> el, ChildAligner aligner) {
        this->ensureCanHaveAsChild(*el);
        return addUnsafe(el, aligner);
    }
    /**
     * Add new child (translated) to end of children vector.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el new child
     * @param translation trasnalation of child
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(shared_ptr<ChildType> el, const DVec& translation = Primitive<dim>::ZERO_VEC);

    /**
     * Add new child (trasnlated) to end of children vector.
     * @param el new child
     * @param translation trasnalation of child
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(shared_ptr<ChildType> el, const DVec& translation = Primitive<dim>::ZERO_VEC) {
        this->ensureCanHaveAsChild(*el);
        return addUnsafe(el, translation);
    }

    //methods overwrite to use cache:
    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        return ensureHasCache()->getMaterial(p);
    }

    virtual bool contains(const DVec& p) const {
        return ensureHasCache()->contains(p);
    }

    virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const {
        return ensureHasCache()->getPathsAt(this->shared_from_this(), point, all);
    }

    //some methods must be overwrite to invalidate cache:
    virtual void onChildChanged(const GeometryObject::Event& evt) {
        if (evt.isResize()) invalidateCache();
        WithAligners<GeometryObjectContainer<dim>, TranslationContainerChildAligner<dim>>::onChildChanged(evt);
    }

    virtual bool removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
        if (WithAligners<GeometryObjectContainer<dim>, TranslationContainerChildAligner<dim>>::removeIfTUnsafe(predicate)) {
            invalidateCache();
            return true;
        } else
            return false;
    }

    virtual void removeAtUnsafe(std::size_t index) {
        invalidateCache();
        WithAligners<GeometryObjectContainer<dim>, TranslationContainerChildAligner<dim>>::removeAtUnsafe(index);
    }

    //virtual void writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const;

protected:
    virtual shared_ptr<GeometryObject> changedVersionForChildren(std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const;

    /// Destroy geometry cache, this should be called when children was changed (it will be rebuild by first operation which use it).
    void invalidateCache();

    /**
     * Construct cache if it not exists and return it.
     * @return non-null pointer to cache
     */
    CacheNode<dim>* ensureHasCache();

    /**
     * Construct cache if it not exists and return it.
     *
     * This method is thread-safty (can be call by many threads at once, and only first call will build the cache).
     * @return non-null pointer to cache
     */
    CacheNode<dim>* ensureHasCache() const;

private:
    shared_ptr<TranslationT> newTranslation(const shared_ptr<ChildType>& el, ChildAligner aligner);

    /**
     * Cache which allow to do some geometry operation (like getMaterial) much faster if this container has many children.
     * It is create by first operation which use it (see ensureHasCache method), and destroy on each change of children (see invalidateCache method).
     */
    std::atomic<CacheNode<dim>*> cache;

    /// Mutex used by const version of ensureHasCache.
    boost::mutex cache_mutex;

};

PLASK_API_EXTERN_TEMPLATE_STRUCT(TranslationContainer<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(TranslationContainer<3>)

} // namespace plask

#endif // PLASK__GEOMETRY_CONTAINER_TRANS_H
