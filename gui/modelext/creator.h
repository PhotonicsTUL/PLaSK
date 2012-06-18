#ifndef PLASK_GUI_MODEL_EXT_CREATOR_H
#define PLASK_GUI_MODEL_EXT_CREATOR_H

#include <plask/geometry/element.h>

/**
 * Create geometry element.
 */
struct GeometryElementCreator {

    /**
     * Construct element.
     * @param dim number of dimentions of constructed element
     * @return constructed element or nullptr if this doesn't support given @p dim
     */
    virtual plask::shared_ptr<plask::GeometryElement> getElement(int dim) const = 0;

    template <int dim>
    plask::shared_ptr< plask::GeometryElementD<dim> > getElement() const {
        return plask::static_pointer_cast< plask::GeometryElementD<dim> >(getElement(dim));
    }

    /**
     * Get displayable name of created element.
     * @return name of created element
     */
    virtual std::string getName() const = 0;

    /**
     * Check if this can create element with given number of dimentions @p dim.
     *
     * Default implementation returns @c true only for @p dim equal to 2 or 3.
     * @param dim number of dimentions
     * @return @c true only if this can create element with given number of dimentions @p dim
     */
    virtual bool supportDimensionsCount(int dim) const {
        return dim == 2 || dim == 3;
    }

    //TODO icons? Qt actions?
};

const std::vector<const GeometryElementCreator*>& getCreators();
const std::vector<const GeometryElementCreator*>& getCreators(int dim);

//std::vector<GeometryElementCreator&> get3dCreators();

#endif // PLASK_GUI_MODEL_EXT_CREATOR_H
