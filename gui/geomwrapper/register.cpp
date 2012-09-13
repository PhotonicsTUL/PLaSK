#include "register.h"

#include <typeinfo>
#include <typeindex>
#include <unordered_map>
#include <map>

#include "container.h"
#include "leaf.h"
#include "transform.h"
#include "geometry.h"
#include <plask/utils/cache.h>

typedef ObjectWrapper* construct_object_wrapper_t(plask::shared_ptr<plask::GeometryObject> to_wrap);

struct Register {

    //TODO possible destructor which disconnect from all changed signals is required

    template <typename WrapperType>
    static ObjectWrapper* constructWrapper(plask::shared_ptr<plask::GeometryObject> to_wrap) {
        auto res = new WrapperType();
        res->setWrappedObject(to_wrap);
        return res;
    }

    /// Constructors of wrappers for geometry objects, map: type id index of plask::GeometryObjects -> wrapper constructor.
    std::unordered_map<std::type_index, construct_object_wrapper_t*> wrappersConstructors;

    /// Constructed wrappers for geometry objects, map: geometry object -> wrapper for key object.
    //TODO GeometryObject -> shared_ptr<ObjectWrapper>, ObjectWrapper should hold weak_ptr or raw ptr.
    plask::StrongCache<plask::GeometryObject, ObjectWrapper> constructed;

    /// Construct geometry object wrapper using wrappersConstructors. Doesn't change constructed map.
    ObjectWrapper* construct(plask::shared_ptr<plask::GeometryObject> el) {
        auto i = wrappersConstructors.find(std::type_index(typeid(*el)));
        return i == wrappersConstructors.end() ? constructWrapper<ObjectWrapper>(el) : i->second(el);
    }

    /**
     * Get wrapper for given object. Try get it from constructed map first.
     */
    plask::shared_ptr<ObjectWrapper> get(plask::shared_ptr<plask::GeometryObject> el) {
        if (auto res = constructed.get(el))
            return res;
        else
            return constructed(el, construct(el));
    }

    template <typename WrapperType, typename PlaskType = typename WrapperType::WrappedType>
    void appendConstructor() {
        wrappersConstructors[std::type_index(typeid(*plask::make_shared<PlaskType>()))] =
                &constructWrapper<WrapperType>;
    }

    Register() {
        appendConstructor< TranslationWrapper<2> >();
        appendConstructor< TranslationWrapper<3> >();
        appendConstructor< StackWrapper<2> >();
        appendConstructor< StackWrapper<3> >();
        appendConstructor< MultiStackWrapper<2> >();
        appendConstructor< MultiStackWrapper<3> >();
        appendConstructor< ShelfWrapper >();
        appendConstructor< BlockWrapper<2> >();
        appendConstructor< BlockWrapper<3> >();
        appendConstructor< ExtrusionWrapper >();
        appendConstructor< Geometry2DCartesianWrapper >();
        appendConstructor< Geometry2DCylindricalWrapper >();
    }

};

Register geom_register;

plask::shared_ptr<ObjectWrapper> ext(plask::shared_ptr<plask::GeometryObject> el) {
    return geom_register.get(el);
}

plask::shared_ptr<ObjectWrapper> ext(const plask::GeometryObject& el) {
    return ext(const_cast<plask::GeometryObject&>(el).shared_from_this());
}


std::string NamesFromExtensions::getName(const plask::GeometryObject &object, plask::AxisNames &axesNames) const {
    return ext(object)->getName();
}

std::vector<std::string> NamesFromExtensions::getPathNames(const plask::GeometryObject &parent, const plask::GeometryObject &child, std::size_t index_of_child_in_parent) const {
    //TODO
    return std::vector<std::string>();
}
