#ifndef PLASK__GEOMETRY_PATH_H
#define PLASK__GEOMETRY_PATH_H

#include <map>
#include <plask/config.h>
#include "element.h"
#include "transform.h"

namespace plask {

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
    typedef std::map< weak_ptr<GeometryElement>, weak_ptr<GeometryElement> > HintMap;

    /**
     * Type for arc in graph. Pair: container of geometry elements -> element in container.
     * @see @ref geometry_paths
     */
    typedef HintMap::value_type Hint;

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
     * Get child for given container.
     * @return child for given container or @c nullptr if there is no hint for given container
     */
    shared_ptr<GeometryElement> getChild(shared_ptr<const GeometryElement> container);

    /**
     * Get child for given container.
     * @return child for given container or @c nullptr if there is no hint for given container
     */
    shared_ptr<GeometryElement> getChild(const GeometryElement& container) {
        return getChild(container.shared_from_this());
    }

    /**
     * Get child for given container.
     * @return child for given container or @c nullptr if there is no hint for given container
     */
    shared_ptr<GeometryElement> getChild(shared_ptr<const GeometryElement> container) const;

    /**
     * Get child for given container.
     * @return child for given container or @c nullptr if there is no hint for given container
     */
    shared_ptr<GeometryElement> getChild(const GeometryElement& container) const {
        return getChild(container.shared_from_this());
    }

    /**
     * Get child for given hint.
     * @return child for given hint or @c nullptr if there is no hint for given container
     */
    static shared_ptr<GeometryElement> getChild(const Hint& hint);

    /**
     * Get container for given hint.
     * @param hint hint
     * @return container for given hint or @c nullptr if there is no hint for given container
     */
    static shared_ptr<GeometryElement> getContainer(const Hint& hint);

    /**
     * Get child for given container casted to Translation object.
     * @param container container
     * @return casted child for given container or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> shared_ptr<Translation<dim>> getTranslationChild(shared_ptr<const GeometryElement> container) {
        return dynamic_pointer_cast<Translation<dim>>(getChild(container));
    }

    /**
     * Get child for given container casted to Translation object.
     * @param container container
     * @return casted child for given container or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> shared_ptr<Translation<dim>> getTranslationChild(const GeometryElement& container) {
        return getTranslationChild<dim>(container.shared_from_this());
    }

    /**
     * Get child for given container casted to Translation object.
     * @param container container
     * @return casted child for given container or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> shared_ptr<Translation<dim>> getTranslationChild(shared_ptr<const GeometryElement> container) const {
        return dynamic_pointer_cast<Translation<dim>>(getChild(container));
    }

    /**
     * Get child for given container casted to Translation object.
     * @param container container
     * @return casted child for given container or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> shared_ptr<Translation<dim>> getTranslationChild(const GeometryElement& container) const {
        return getTranslationChild<dim>(container.shared_from_this());
    }

    /**
     * Get child for given hint casted to Translation object.
     * @param container container
     * @return casted child for given hint or @c nullptr if there is no hint or it cannot be casted
     */
    template <int dim> static shared_ptr<Translation<dim>> getTranslationChild(const Hint& hint) {
        return dynamic_pointer_cast<Translation<dim>>(getChild(hint));
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

    std::vector< shared_ptr<GeometryElement> > elements;

    Path& operator+=(const PathHints::Hint& hint);

    Path& operator+=(const GeometryElement& last);

    Path& append(const GeometryElement& last, const PathHints& hints);

};

}

#endif // PLASK__GEOMETRY_PATH_H
