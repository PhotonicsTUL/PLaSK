#ifndef PLASK_GUI_MODEL_EXT_CREATOR_H
#define PLASK_GUI_MODEL_EXT_CREATOR_H

#include <plask/geometry/element.h>

/**
 * Create geometry element.
 */
struct GeometryElementCreator {

    /**
     * Construct element.
     * @return constructed element
     */
    virtual plask::shared_ptr<plask::GeometryElement> getElement() const = 0;

    template <int dim>
    plask::shared_ptr< plask::GeometryElementD<dim> > getElement() const {
        return plask::static_pointer_cast< plask::GeometryElementD<dim> >(getElement());
    }

    /**
     * Get displayable name of created element.
     * @return name of created element
     */
    virtual std::string getName() const = 0;

    /**
     * Get number of dimentions of created element.
     * @return number of dimentions
     */
    virtual int getDimensionsCount() const = 0;

    //TODO icons? Qt actions?
};

template <int dim>
std::vector<const GeometryElementCreator*> getCreators();

//std::vector<GeometryElementCreator&> get3dCreators();

#endif // PLASK_GUI_MODEL_EXT_CREATOR_H
