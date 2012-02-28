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
    typedef std::map< weak_ptr<GeometryElement>, std::set< weak_ptr<GeometryElement> > > HintMap;

    /**
     * Type for arc in graph. Pair: container of geometry elements -> element in container.
     * @see @ref geometry_paths
     */
    typedef Edge Hint;

    ///Hints map.
    HintMap hintFor;

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

    /**
     * Add hint to hints map. Overwrite if hint for given container already exists.
     * @param container, child hint to add
     */
    void addHint(weak_ptr<GeometryElement> container, weak_ptr<GeometryElement> child);

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
class Path {

    void addElements(const GeometryElement::Subtree* path_nodes);

    void addElements(const GeometryElement::Subtree& paths);

public:

    std::vector< shared_ptr<const GeometryElement> > elements;

    Path& operator+=(const GeometryElement::Subtree& paths);

    Path& operator+=(const Path& path);

    Path& operator+=(const PathHints::Hint& hint);

    Path& operator+=(const GeometryElement& last);

    Path& append(const GeometryElement& last, const PathHints& hints);

};

}

#endif // PLASK__GEOMETRY_PATH_H
