#ifndef PLASK__PYTHON_MODULES_H
#define PLASK__PYTHON_MODULES_H

#include <plask/module.h>
#include "python_globals.h"
#include "python_provider.h"

namespace plask { namespace python {

// template <typename ModuleT> struct ExportModule;

namespace detail {

    template <typename ModuleT, typename BaseT>
    struct ExportedModuleDefaultDefs {
        template <typename PyModule>
        static auto init(PyModule& module) -> PyModule& {}
    };

    template <typename ModuleT>
    struct ExportedModuleDefaultDefs<ModuleT, ModuleOver<CalculationSpace>> {
        template <typename PyModule>
        static auto init(PyModule& module) -> PyModule& {
            module.add_property("geometry", &ModuleT::getGeometry, &ModuleT::setGeometry, "Geometry provided to the module");
        }
    };

    template <typename ModuleT>
    struct ExportedModuleDefaultDefs<ModuleT, ModuleWithMesh<CalculationSpace, Mesh<2>>> {
        template <typename PyModule>
        static auto init(PyModule& module) -> PyModule& {
            module.add_property("geometry", &ModuleT::getGeometry, &ModuleT::setGeometry, "Geometry provided to the module");
            module.add_property("mesh", &ModuleT::getMesh, &ModuleT::setMesh, "Mesh provided to the module");
        }
    };

    template <typename ModuleT>
    struct ExportedModuleDefaultDefs<ModuleT, ModuleWithMesh<CalculationSpace, Mesh<3>>> {
        template <typename PyModule>
        static auto init(PyModule& module) -> PyModule& {
            module.add_property("geometry", &ModuleT::getGeometry, &ModuleT::setGeometry, "Geometry provided to the module");
            module.add_property("mesh", &ModuleT::getMesh, &ModuleT::setMesh, "Mesh provided to the module");
        }
    };


    template <typename Class, typename PropertyTag, typename SpaceType>
    struct ReceiverSetter
    {
        typedef ReceiverFor<PropertyTag,SpaceType> Data;

        ReceiverSetter(Data Class::* field) : field(field) {}
        void operator()(Class& self, typename PropertyTag::ValueType const& value) {
            self.*field = value;
        }

      private:
        Data Class::* field;
    };



} // namespace detail

/**
 * This class should be instantiated to export a module to Python.
 *
 * It automatically determines one of the standard bases for the module and exports its methods
 * and fields. Furthermode it provides convenient methods to export providers and receivers (TODO).
 */
template <typename ModuleT>
struct ExportModule : public py::class_<ModuleT, shared_ptr<ModuleT>, py::bases<Module>, boost::noncopyable> {

    typedef ModuleT Class;
    typedef py::class_<ModuleT, shared_ptr<ModuleT>, py::bases<Module>, boost::noncopyable> PyClass;

    template <typename... Args>
    ExportModule(Args&&... args) : PyClass(std::forward<Args>(args)...) {
        detail::ExportedModuleDefaultDefs<ModuleT, typename ModuleT::BASE_MODULE_TYPE>::init(*this);
    }

    template <typename ProviderT>
    ExportModule& add_provider(const char* name, const ProviderT& field, const char* help) {
        this->def_readonly(name, field, help);
        return *this;
    }

    template <typename PropertyTag, typename SpaceType>
    ExportModule& add_receiver(const char* name, ReceiverFor<PropertyTag,SpaceType> Class::* field, const char* help) {
        this->add_property(name, py::make_getter(field),
                           py::make_function(detail::ReceiverSetter<Class,PropertyTag,SpaceType>(field),
                                             py::default_call_policies(),
                                             boost::mpl::vector3<void, Class&, typename PropertyTag::ValueType const&>()
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

#define USING_PROPERTY(name) RegisterProvider<name>()


using py::arg; // for more convenient specification of default arguments

}} // namespace plask::python

#endif // PLASK__PYTHON_MODULES_H
