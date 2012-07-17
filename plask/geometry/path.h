#ifndef PLASK__GEOMETRY_PATH_H
#define PLASK__GEOMETRY_PATH_H

#include <map>
#include <set>
#include <plask/config.h>
#include "element.h"
#include "transform.h"

namespace plask {

//TODO redefine to structure which alow to cast to container and translation
typedef std::pair< shared_ptr<GeometryElement>, shared_ptr<GeometryElement> > Edge;

struct Path;

/**
Represent hints for path finder.

Hints are used to find unique path for all GeometryElement pairs,
even if one of the pair element is inserted to the geometry graph in more than one place.

Each hint allow to choose one child for geometry element container and it is a pair:
geometry element container -> element in container.

Typically, hints are returned by methods which adds new elements to containers.

@see @ref geometry_paths
*/
struct PathHints {

    ///Type for map: geometry element container -> element in container
#ifdef PLASK_SHARED_PTR_STD
    typedef std::map<weak_ptr<GeometryElement>, std::set<weak_ptr<GeometryElement>, std::owner_less<weak_ptr<GeometryElement>>>,
                     std::owner_less<weak_ptr<GeometryElement>>> HintMap;
#else
    typedef std::map<weak_ptr<GeometryElement>, std::set<weak_ptr<GeometryElement>>> HintMap;
#endif

    /**
     * Type for arc in graph. Pair: container of geometry elements -> element in container.
     * @see @ref geometry_paths
     */
    typedef Edge Hint;

    ///Hints map.
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
     * Construct path hints with all hinst included in @p subtree.
     * @param subtree subtree
     */
    explicit PathHints(const GeometryElement::Subtree& subtree) {
        addAllHintsFromSubtree(subtree);
    }

    PathHints() = default;

    /**
     * Add hint to hints map. Overwrite if hint for given container already exists.
     * @param hint hint to add
     */
    void addHint(const Hint& hint);

    /**
     * Add hint to hints map. Overwrite if hint for given container already exists.
     * @param hint hint to add
     */
    PathHints& operator+=(const Hint& hint) {
        addHint(hint);
        return *this;
    }

    /// Comparison operator for using PathHints as map keys
    inline bool operator<(const PathHints& comp) const {
        return hintFor < comp.hintFor;
    }

    /**
     * Add hint to hints map. Overwrite if hint for given container already exists.
     * @param container, child hint to add
     */
    void addHint(weak_ptr<GeometryElement> container, weak_ptr<GeometryElement> child);

    /**
     * Add all hinst included in path elements.
     * @param pathElements geometry elements which are on path
     */
    void addAllHintsFromPath(std::vector< shared_ptr<const GeometryElement> > pathElements);

    /**
     * Add all hinst included in @p path.
     * @param path path
     */
    void addAllHintsFromPath(const Path& path);

    /**
     * Add all hinst included in @p subtree.
     * @param subtree subtree
     */
    void addAllHintsFromSubtree(const GeometryElement::Subtree& subtree);

    /**
     * Get children for given container.
     * @return children for given container or empty set if there is no hints for given container
     */
    std::set<shared_ptr<GeometryElement>> getChildren(shared_ptr<const GeometryElement> container);

    /**
     * Get children for given container.
     * @return children for given container or empty set if there is no hints for given container
     */
    std::set<shared_ptr<GeometryElement>> getChildren(const GeometryElement& container) {
        return getChildren(container.shared_from_this());
    }

    /**
     * Get child for given container.
     * @return child for given container or @c nullptr if there is no hint for given container
     */
    std::set<shared_ptr<GeometryElement>> getChildren(shared_ptr<const GeometryElement> container) const;

    /**
     * Get child for given container.
     * @return child for given container or @c nullptr if there is no hint for given container
     */
    std::set<shared_ptr<GeometryElement>> getChildren(const GeometryElement& container) const {
        return getChildren(container.shared_from_this());
    }

    template <int dim> static
    std::set<shared_ptr<Translation<dim>>> castToTranslation(std::set<shared_ptr<GeometryElement>> src) {
        std::set<shared_ptr<Translation<dim>>> result;
        for (auto& e: src) result.insert(dynamic_pointer_cast<Translation<dim>>(e));
        return result;
    }

    /**
     * Get child for given container casted to Translation object.
     * @param container container
     * @return casted child for given container or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> std::set<shared_ptr<Translation<dim>>> getTranslationChildren(shared_ptr<const GeometryElement> container) {
        return castToTranslation<dim>(getChildren(container));
    }

    /**
     * Get child for given container casted to Translation object.
     * @param container container
     * @return casted child for given container or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> std::set<shared_ptr<Translation<dim>>> getTranslationChildren(const GeometryElement& container) {
        return getTranslationChildren<dim>(container.shared_from_this());
    }

    /**
     * Get child for given container casted to Translation object.
     * @param container container
     * @return casted child for given container or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> std::set<shared_ptr<Translation<dim>>> getTranslationChildren(shared_ptr<const GeometryElement> container) const {
        return castToTranslation<dim>(getChildren(container));
    }

    /**
     * Get child for given container casted to Translation object.
     * @param container container
     * @return casted child for given container or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> std::set<shared_ptr<Translation<dim>>> getTranslationChildren(const GeometryElement& container) const {
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
struct Path {

  private:

    bool completeToFirst(const GeometryElement& newFirst, const PathHints* hints = nullptr);

    bool completeFromLast(const GeometryElement& newLast, const PathHints* hints = nullptr);

  public:

    Path(const std::vector< shared_ptr<const GeometryElement> >& path)
        : elements(path) {}

    Path(std::vector< shared_ptr<const GeometryElement> >&& path)
        : elements(path) {}

    Path(const GeometryElement::Subtree& paths)
        : elements(paths.toLinearPath()) {}

    //This are the same as default constructors, so can be skiped:
    //Path(const Path& path): elements(path.elements) {}
    //Path(Path&& path): elements(path.elements) {}

    Path(const PathHints::Hint& hint) { append(hint); }

    Path(const GeometryElement& element) { append(element); }

    Path(shared_ptr<const GeometryElement> element) { append(*element); }

    ///Path content.
    std::vector< shared_ptr<const GeometryElement> > elements;

    /**
     * Push front content of @a toAdd vector to elements.
     *
     * Skip last element from @p toAdd if it is first in elements, but neither check path integrity nor complete path.
     * @param toAdd elements to push on front of elements
     * @see operator+=(const std::vector< shared_ptr<const GeometryElement> >& path)
     */
    void push_front(const std::vector< shared_ptr<const GeometryElement> >& toAdd);

    /**
     * Push back content of @a toAdd vector to elements.
     *
     * Skip first element from @p toAdd if it is last in elements, but neither check path integrity nor complete path.
     * @param toAdd elements to push on back of elements
     * @see operator+=(const std::vector< shared_ptr<const GeometryElement> >& path)
     */
    void push_back(const std::vector< shared_ptr<const GeometryElement> >& toAdd);

    /**
     * Append @p path content to this path.
     *
     * Try complete missing path fragment if necessary, and throw exception it is impossible or ambiguous.
     * @param path elements to add
     * @param hints optional path hints which are use to non-ambiguous completion of paths
     */
    Path& append(const std::vector< shared_ptr<const GeometryElement> >& path, const PathHints* hints = nullptr);

    /**
     * Append @p paths content to this path.
     *
     * Try complete missing path fragment if necessary, and throw exception it is impossible or ambiguous.
     * @param path elements to add, exception will be throwed if it have branches
     * @param hints optional path hints which are used to non-ambiguous completion of paths
     */
    Path& append(const GeometryElement::Subtree& path, const PathHints* hints = nullptr);

    /**
     * Append @p path content to this path.
     *
     * Try complete missing path fragment if necessary, and throw exception it is impossible or ambiguous.
     * @param path elements to add
     * @param hints optional path hints which are use to non-ambiguous completion of paths
     */
    Path& append(const Path& path, const PathHints* hints = nullptr);

    /**
     * Append @p hint to this path.
     *
     * Try complete missing path fragment if necessary, and throw exception it is impossible or ambiguous.
     * @param hint elements to add
     * @param hints optional path hints which are use to non-ambiguous completion of paths
     */
    Path& append(const PathHints::Hint& hint, const PathHints* hints = nullptr);

    /**
     * Append @p element to this path.
     *
     * Try complete missing path fragment if necessary, and throw exception it is impossible or ambiguous.
     * @param element elements to add
     * @param hints optional path hints which are use to non-ambiguous completion of paths
     */
    Path& append(const GeometryElement& element, const PathHints* hints = nullptr);

    /**
     * Append @p element to this path.
     *
     * Try complete missing path fragment if necessary, and throw exception it is impossible or ambiguous.
     * @param element elements to add
     * @param hints optional path hints which are use to non-ambiguous completion of paths
     */
    Path& append(shared_ptr<const GeometryElement> element, const PathHints* hints = nullptr);

    Path& operator+=(const std::vector< shared_ptr<const GeometryElement> >& path) { return append(path); }

    Path& operator+=(const GeometryElement::Subtree& paths) { return append(paths); }

    Path& operator+=(const Path& path) { return append(path); }

    Path& operator+=(const PathHints::Hint& hint) { return append(hint); }

    Path& operator+=(const GeometryElement& element) { return append(element); }

    Path& operator+=(shared_ptr<const GeometryElement> element) { return append(element); }

    /**
     * Get path hints implicted by this.
     * @return path hints which include all hints implicted by this path
     */
    PathHints getPathHints() const;

    /**
     * Get path hinst implicted by this.
     * @return path hints which include all hints implicted by this path
     */
    operator PathHints() const { return getPathHints(); }
};

}

#endif // PLASK__GEOMETRY_PATH_H

