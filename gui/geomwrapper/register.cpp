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

typedef ElementWrapper* construct_element_wrapper_t(plask::shared_ptr<plask::GeometryElement> to_wrap);

struct Register {

    //TODO possible destructor which disconnect from all changed signals is required

    template <typename WrapperType>
    static ElementWrapper* constructWrapper(plask::shared_ptr<plask::GeometryElement> to_wrap) {
        auto res = new WrapperType();
        res->setWrappedElement(to_wrap);
        return res;
    }

    /// Constructors of wrappers for geometry elements, map: type id index of plask::GeometryElements -> wrapper constructor.
    std::unordered_map<std::type_index, construct_element_wrapper_t*> wrappersConstructors;

    /// Constructed wrappers for geometry elements, map: geometry element -> wrapper for key element.
    plask::Cache<plask::GeometryElement, ElementWrapper> constructed;

    /// Construct geometry element wrapper using wrappersConstructors. Doesn't change constructed map.
    ElementWrapper* construct(plask::shared_ptr<plask::GeometryElement> el) {
        auto i = wrappersConstructors.find(std::type_index(typeid(*el)));
        return i == wrappersConstructors.end() ? constructWrapper<ElementWrapper>(el) : i->second(el);
    }

    /**
     * Get wrapper for given element. Try get it from constructed map first.
     */
    plask::shared_ptr<ElementWrapper> get(plask::shared_ptr<plask::GeometryElement> el) {
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
        appendConstructor< Geometry2DCartesianWrapper >();
        appendConstructor< Geometry2DCylindricalWrapper >();
    }

};

Register geom_register;

plask::shared_ptr<ElementWrapper> ext(plask::shared_ptr<plask::GeometryElement> el) {
    return geom_register.get(el);
}
