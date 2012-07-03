#ifndef PLASK__PYTHON_MODULES_H
#define PLASK__PYTHON_MODULES_H

#include <type_traits>

#include <plask/module.h>
#include "python_globals.h"
#include "python_provider.h"

#include <plask/provider/optical.h>

namespace plask { namespace python {

// template <typename ModuleT> struct ExportModule;

namespace detail {

    template <typename ModuleT, typename EnableSpace=void, typename EnableMesh=void>
    struct ExportedModuleDefaultDefs
    {
        template <typename PyModule>
        static auto init(PyModule& module) -> PyModule& { return module; }
    };

    template <typename ModuleT, typename EnableMesh>
    struct ExportedModuleDefaultDefs<ModuleT,
        typename std::enable_if<std::is_base_of<ModuleOver<typename ModuleT::SpaceType>, ModuleT>::value>::type,
        EnableMesh>
    {
        template <typename PyModule>
        static auto init(PyModule& module) -> PyModule& {
            module.add_property("geometry", &ModuleT::getGeometry, &ModuleT::setGeometry, "Geometry provided to the module");
            return module;
        }
    };

    template <typename ModuleT>
    struct ExportedModuleDefaultDefs<ModuleT,
        typename std::enable_if<std::is_base_of<ModuleOver<typename ModuleT::SpaceType>, ModuleT>::value>::type,
        typename std::enable_if<std::is_base_of<ModuleWithMesh<typename ModuleT::SpaceType, typename ModuleT::MeshType>, ModuleT>::value>::type>
    {
        template <typename PyModule>
        static auto init(PyModule& module) -> PyModule& {
            module.add_property("geometry", &ModuleT::getGeometry, &ModuleT::setGeometry, "Geometry provided to the module");
            module.add_property("mesh", &ModuleT::getMesh, &ModuleT::setMesh, "Mesh provided to the module");
            return module;
        }
    };

    template <typename Class, typename ReceiverT>
    struct ReceiverSetter
    {
        ReceiverSetter(ReceiverT Class::* field) : field(field) {}
        void operator()(Class& self, typename ReceiverT::PropertyTag::ValueType const& value) {
            self.*field = value;
        }

      private:
        ReceiverT Class::* field;
    };

} // namespace detail

/**
 * This class should be instantiated to export a module to Python.
 *
 * It automatically determines one of the standard bases for the module and exports its methods
 * and fields. Furthermore it provides convenient methods to export providers and receivers.
 */
template <typename ModuleT>
struct ExportModule : public py::class_<ModuleT, shared_ptr<ModuleT>, py::bases<Module>, boost::noncopyable> {

    typedef ModuleT Class;
    typedef py::class_<ModuleT, shared_ptr<ModuleT>, py::bases<Module>, boost::noncopyable> PyClass;

    template <typename... Args>
    ExportModule(Args&&... args) : PyClass(std::forward<Args>(args)...) {
        detail::ExportedModuleDefaultDefs<ModuleT>::init(*this);
    }

    template <typename ProviderT>
    ExportModule& add_provider(const char* name, ProviderT Class::* field, const char* help) {
        RegisterProvider<ProviderT>();
        this->def_readonly(name, field, help);
        return *this;
    }

    template <typename ReceiverT>
    ExportModule& add_receiver(const char* name, ReceiverT Class::* field, const char* help) {
        RegisterReceiver<ReceiverT>();
        this->add_property(name, py::make_getter(field),
                           py::make_function(detail::ReceiverSetter<Class,ReceiverT>(field),
                                             py::default_call_policies(),
                                             boost::mpl::vector3<void, Class&, typename ReceiverT::PropertyTag::ValueType const&>()
                                            ),
                           help
                          );
        return *this;
    }


};

// Here are some useful defines.
// Note that if you use them to define methods, properties etc, you should also use MODULE
#define CLASS(cls, name, help) typedef cls __Class__; ExportModule<cls> __module__(name, help);
#define METHOD(method, help, ...) __module__.def(BOOST_PP_STRINGIZE(method), &__Class__::method, help, (py::arg("arg1") , ## __VA_ARGS__))
#define RO_PROPERTY(name, get, help) __module__.add_property(BOOST_PP_STRINGIZE(name), &__Class__::get, help)
#define RW_PROPERTY(name, get, set, help) __module__.add_property(BOOST_PP_STRINGIZE(name), &__Class__::get, &__Class__::set, help)
#define RO_FIELD(name, help) __module__.def_readonly(BOOST_PP_STRINGIZE(name), &__Class__::name, help)
#define RW_FIELD(name, help) __module__.def_readwrite(BOOST_PP_STRINGIZE(name), &__Class__::name, help)
#define PROVIDER(name, help) __module__.add_provider(BOOST_PP_STRINGIZE(name), &__Class__::name, help)
#define RECEIVER(name, help) __module__.add_receiver(BOOST_PP_STRINGIZE(name), &__Class__::name, help)


using py::arg; // for more convenient specification of default arguments

}} // namespace plask::python

#endif // PLASK__PYTHON_MODULES_H
