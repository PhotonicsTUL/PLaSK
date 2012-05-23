#include "register.h"

#include <typeinfo>
#include <typeindex>
#include <unordered_map>
#include <map>

typedef Element* construct_element_wrapper_t(plask::shared_ptr<plask::GeometryElement> to_wrap);

struct Register {

    /// Constructors of wrappers for geometry elements, map: type id index of plask::GeometryElements -> wrapper constructor.
    std::unordered_map<std::type_index, construct_element_wrapper_t*> wrappersConstructors;

    /// Constructed wrappers for geometry elements, map: geometry element -> wrapper for key element.
    std::map< const plask::GeometryElement*, plask::weak_ptr<Element> > constructed;

    /// If evt is delete event, remove source of event from constructed map.
    void removeOnDelete(const plask::GeometryElement::Event& evt) {
        if (evt.isDelete()) constructed.erase(&evt.source());
    }

    /// Construct geometry element wrapper using wrappersConstructors. Doesn't change constructed map.
    Element* construct(plask::shared_ptr<plask::GeometryElement> el) {
        auto i = wrappersConstructors.find(std::type_index(typeid(*el)));
        return i == wrappersConstructors.end() ? new Element(el) : i->second(el);
    }

    /**
     * Get wrapper for given element. Try get it from constructed map first.
     */
    plask::shared_ptr<Element> get(plask::shared_ptr<plask::GeometryElement> el) {
        auto constr_iter = constructed.find(el.get());
        if (constr_iter != constructed.end()) {
            if (auto res = constr_iter->second.lock())
                return res;
            else
                constructed.erase(constr_iter);
        }
        auto res = plask::shared_ptr<Element>(construct(el));
        constructed[el.get()] = res;
        el->changedConnectMethod(this, &Register::removeOnDelete);
        return res;
    }

};

Register geom_register;

plask::shared_ptr<Element> geomExt(plask::shared_ptr<plask::GeometryElement> el) {
    return geom_register.get(el);
}
