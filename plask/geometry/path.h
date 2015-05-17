#ifndef PLASK__GEOMETRY_PATH_H
#define PLASK__GEOMETRY_PATH_H

#include <map>
#include <set>
#include <plask/config.h>
#include "transform.h"

namespace plask {

struct GeometryObject;

//TODO redefine to structure which alow to cast to container and translation
typedef std::pair< shared_ptr<GeometryObject>, shared_ptr<GeometryObject> > Edge;

struct Path;

/**
Represent hints for path finder.

Hints are used to find unique path for all GeometryObject pairs,
even if one of the pair object is inserted to the geometry graph in more than one place.

Each hint allow to choose one child for geometry object container and it is a pair:
geometry object container -> object in container.

Typically, hints are returned by methods which adds new objects to containers.

@see @ref geometry_paths
*/
struct PLASK_API PathHints {

    /// Type for map: geometry object container -> object in container
#ifdef PLASK_SHARED_PTR_STD
    typedef std::map<weak_ptr<GeometryObject>, std::set<weak_ptr<GeometryObject>, std::owner_less<weak_ptr<GeometryObject>>>,
                     std::owner_less<weak_ptr<GeometryObject>>> HintMap;
#else
    typedef std::map<weak_ptr<GeometryObject>, std::set<weak_ptr<GeometryObject>>> HintMap;
#endif

    /**
     * Type for arc in graph. Pair: container of geometry objects -> object in container.
     * @see @ref geometry_paths
     */
    typedef Edge Hint;

    /// Hints map.
    HintMap hintFor;

    /**
     * \param hint initial hint in the path hints
     */
    explicit PathHints(const Hint& hint) {
        addHint(hint);
    }

    /**
     * \param path path which is split to the path hints
     */
    explicit PathHints(const Path& path) {
        addAllHintsFromPath(path);
    }

    /**
     * Construct path hints with all hinst included in @p path.
     * @param path path
     */
    explicit PathHints(const std::vector< shared_ptr<const GeometryObject> >& path) {
        addAllHintsFromPath(path);
    }

    /**
     * Construct path hints with all hinst included in @p subtree.
     * @param subtree subtree
     */
    explicit PathHints(const GeometryObject::Subtree& subtree) {
        addAllHintsFromSubtree(subtree);
    }

    /**
     * Construct empty set of path hints.
     */
    PathHints() = default;

    /**
     * Add hint to hints map. Overwrite if hint for given container already exists.
     * @param hint hint to add
     */
    void addHint(const Hint& hint);

    /**
     * Check if hint is included in this.
     * @param container, child_tran content of hint (container and child - typically translation)
     * @return *c true only if hint is included in @c this
     */
    bool include(shared_ptr<const GeometryObject> container, shared_ptr<const GeometryObject> child_tran) const;

    /**
     * Check if @p hint is included in @c this.
     * @param hint hint to check
     * @return *c true only if @p hint is included in @c this
     */
    bool include(const Hint& hint) const {
        return include(hint.first, hint.second);
    }

    /**
     * Add hint to hints map. Overwrite if hint for given container already exists.
     * @param hint hint to add
     */
    PathHints& operator+=(const Hint& hint) {
        addHint(hint);
        return *this;
    }

    /// Comparison operator
    inline bool operator==(const PathHints& comp) const {
        return !(hintFor < comp.hintFor || comp.hintFor < hintFor);
    }

    /// Comparison operator for using PathHints as map keys
    inline bool operator<(const PathHints& comp) const {
        return hintFor < comp.hintFor;
    }

    /**
     * Add hint to hints map. Overwrite if hint for given container already exists.
     * @param container, child hint to add
     */
    void addHint(weak_ptr<GeometryObject> container, weak_ptr<GeometryObject> child);

    /**
     * Add all hinst included in path objects.
     * @param pathObjects geometry objects which are on path
     */
    void addAllHintsFromPath(const std::vector< shared_ptr<const GeometryObject> >& pathObjects);

    /**
     * Add all hinst included in @p path.
     * @param path path
     */
    void addAllHintsFromPath(const Path& path);

    /**
     * Add all hinst included in @p subtree.
     * @param subtree subtree
     */
    void addAllHintsFromSubtree(const GeometryObject::Subtree& subtree);

    /**
     * Get children for given container.
     * @return children for given container or empty set if there is no hints for given container
     */
    std::set<shared_ptr<GeometryObject>> getChildren(shared_ptr<const GeometryObject> container);

    /**
     * Get children for given container.
     * @return children for given container or empty set if there is no hints for given container
     */
    std::set<shared_ptr<GeometryObject>> getChildren(const GeometryObject& container) {
        return getChildren(container.shared_from_this());
    }

    /**
     * Get child for given container.
     * @return child for given container or @c nullptr if there is no hint for given container
     */
    std::set<shared_ptr<GeometryObject>> getChildren(shared_ptr<const GeometryObject> container) const;

    /**
     * Get child for given container.
     * @return child for given container or @c nullptr if there is no hint for given container
     */
    std::set<shared_ptr<GeometryObject>> getChildren(const GeometryObject& container) const {
        return getChildren(container.shared_from_this());
    }

    template <int dim> static
    std::set<shared_ptr<Translation<dim>>> castToTranslation(std::set<shared_ptr<GeometryObject>> src) {
        std::set<shared_ptr<Translation<dim>>> result;
        for (auto& e: src) result.insert(dynamic_pointer_cast<Translation<dim>>(e));
        return result;
    }

    /**
     * Get child for given container casted to Translation object.
     * @param container container
     * @return casted child for given container or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> std::set<shared_ptr<Translation<dim>>> getTranslationChildren(shared_ptr<const GeometryObject> container) {
        return castToTranslation<dim>(getChildren(container));
    }

    /**
     * Get child for given container casted to Translation object.
     * @param container container
     * @return casted child for given container or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> std::set<shared_ptr<Translation<dim>>> getTranslationChildren(const GeometryObject& container) {
        return getTranslationChildren<dim>(container.shared_from_this());
    }

    /**
     * Get child for given container casted to Translation object.
     * @param container container
     * @return casted child for given container or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> std::set<shared_ptr<Translation<dim>>> getTranslationChildren(shared_ptr<const GeometryObject> container) const {
        return castToTranslation<dim>(getChildren(container));
    }

    /**
     * Get child for given container casted to Translation object.
     * @param container container
     * @return casted child for given container or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> std::set<shared_ptr<Translation<dim>>> getTranslationChildren(const GeometryObject& container) const {
        return getTranslationChildren<dim>(container.shared_from_this());
    }

    /**
     * Remove all hints which refer to deleted objects.
     */
    void cleanDeleted();

};

/**
 * Path in geometry graph.
 */
struct PLASK_API Path {

  private:

    bool completeToFirst(const GeometryObject& newFirst, const PathHints* hints = nullptr);

    bool completeFromLast(const GeometryObject& newLast, const PathHints* hints = nullptr);

  public:

    Path(const std::vector< shared_ptr<const GeometryObject> >& path)
        : objects(path) {}

    Path(std::vector< shared_ptr<const GeometryObject> >&& path)
        : objects(std::move(path)) {}

    Path(const GeometryObject::Subtree& paths)
        : objects(paths.toLinearPath().objects) {}

    Path(GeometryObject::Subtree&& paths)
        : objects(std::move(paths.toLinearPath().objects)) {}

    // These are the same as default constructors, so can be skiped:
    // Path(const Path& path): objects(path.objects) {}
    // Path(Path&& path): objects(path.objects) {}

    Path(const PathHints::Hint& hint) { append(hint); }

    Path(const GeometryObject& object) { append(object); }

    Path(shared_ptr<const GeometryObject> object) { append(*object); }

    /// Path content
    std::vector< shared_ptr<const GeometryObject> > objects;

    /**
     * Push front content of @a toAdd vector to objects.
     *
     * Skip last object from @p toAdd if it is first in objects, but neither check path integrity nor complete path.
     * @param toAdd objects to push on front of objects
     * @see operator+=(const std::vector< shared_ptr<const GeometryObject> >& path)
     */
    void push_front(const std::vector< shared_ptr<const GeometryObject> >& toAdd);

    /**
     * Push back content of @a toAdd vector to objects.
     *
     * Skip first object from @p toAdd if it is last in objects, but neither check path integrity nor complete path.
     * @param toAdd objects to push on back of objects
     * @see operator+=(const std::vector< shared_ptr<const GeometryObject> >& path)
     */
    void push_back(const std::vector< shared_ptr<const GeometryObject> >& toAdd);

    /**
     * Append @p path content to this path.
     *
     * Try complete missing path fragment if necessary, and throw exception it is impossible or ambiguous.
     * @param path objects to add
     * @param hints optional path hints which are use to non-ambiguous completion of paths
     */
    Path& append(const std::vector< shared_ptr<const GeometryObject> >& path, const PathHints* hints = nullptr);

    /**
     * Append @p paths content to this path.
     *
     * Try complete missing path fragment if necessary, and throw exception it is impossible or ambiguous.
     * @param path objects to add, exception will be throwed if it have branches
     * @param hints optional path hints which are used to non-ambiguous completion of paths
     */
    Path& append(const GeometryObject::Subtree& path, const PathHints* hints = nullptr);

    /**
     * Append @p path content to this path.
     *
     * Try complete missing path fragment if necessary, and throw exception it is impossible or ambiguous.
     * @param path objects to add
     * @param hints optional path hints which are use to non-ambiguous completion of paths
     */
    Path& append(const Path& path, const PathHints* hints = nullptr);

    /**
     * Append @p hint to this path.
     *
     * Try complete missing path fragment if necessary, and throw exception it is impossible or ambiguous.
     * @param hint objects to add
     * @param hints optional path hints which are use to non-ambiguous completion of paths
     */
    Path& append(const PathHints::Hint& hint, const PathHints* hints = nullptr);

    /**
     * Append @p object to this path.
     *
     * Try complete missing path fragment if necessary, and throw exception it is impossible or ambiguous.
     * @param object objects to add
     * @param hints optional path hints which are use to non-ambiguous completion of paths
     */
    Path& append(const GeometryObject& object, const PathHints* hints = nullptr);

    /**
     * Append @p object to this path.
     *
     * Try complete missing path fragment if necessary, and throw exception it is impossible or ambiguous.
     * @param object objects to add
     * @param hints optional path hints which are use to non-ambiguous completion of paths
     */
    Path& append(shared_ptr<const GeometryObject> object, const PathHints* hints = nullptr);

    Path& operator+=(const std::vector< shared_ptr<const GeometryObject> >& path) { return append(path); }

    Path& operator+=(const GeometryObject::Subtree& paths) { return append(paths); }

    Path& operator+=(const Path& path) { return append(path); }

    Path& operator+=(const PathHints::Hint& hint) { return append(hint); }

    Path& operator+=(const GeometryObject& object) { return append(object); }

    Path& operator+=(shared_ptr<const GeometryObject> object) { return append(object); }

    /**
     * Get path hints implicted by this.
     * @return path hints which include all hints implicted by this path
     */
    PathHints getPathHints() const;

    /// \return first object of the path
    shared_ptr<const GeometryObject> front() const { return objects.front(); }

    /// \return last object of the path
    shared_ptr<const GeometryObject> back() const { return objects.back(); }

    /**
     * Get path hinst implicted by this.
     * @return path hints which include all hints implicted by this path
     */
    operator PathHints() const { return getPathHints(); }
};

}

#endif // PLASK__GEOMETRY_PATH_H

