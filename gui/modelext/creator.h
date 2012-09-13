#ifndef PLASK_GUI_MODEL_EXT_CREATOR_H
#define PLASK_GUI_MODEL_EXT_CREATOR_H

#include <plask/geometry/object.h>

#include <QMimeData>

/**
 * Create geometry object.
 */
struct GeometryObjectCreator {

    /**
     * Construct object.
     * @param dim number of dimentions of constructed object
     * @return constructed object or nullptr if this doesn't support given @p dim
     */
    virtual plask::shared_ptr<plask::GeometryObject> getObject(int dim) const = 0;

    template <int dim>
    plask::shared_ptr< plask::GeometryObjectD<dim> > getObject() const {
        return plask::static_pointer_cast< plask::GeometryObjectD<dim> >(getObject(dim));
    }

    /**
     * Get displayable name of created object.
     * @return name of created object
     */
    virtual std::string getName() const = 0;

    /**
     * Check if this can create object with given number of dimentions @p dim.
     *
     * Default implementation returns @c true only for @p dim equal to 2 or 3.
     * @param dim number of dimentions
     * @return @c true only if this can create object with given number of dimentions @p dim
     */
    virtual bool supportDimensionsCount(int dim) const {
        return dim == 2 || dim == 3;
    }

    //TODO icons? Qt actions?

    static GeometryObjectCreator* fromMimeData(const QMimeData * data);
};

const std::vector<const GeometryObjectCreator*>& getCreators();
const std::vector<const GeometryObjectCreator*>& getCreators(int dim);

//std::vector<GeometryObjectCreator&> get3DCreators();

#define MIME_PTR_TO_CREATOR "data/pointer-to-object-creator"



#endif // PLASK_GUI_MODEL_EXT_CREATOR_H
