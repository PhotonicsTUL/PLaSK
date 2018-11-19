#ifndef PLASK__PYTHON_SOLVERS_H
#define PLASK__PYTHON_SOLVERS_H

#include <type_traits>

#include <plask/solver.h>
#include "python_globals.h"
#include "python_provider.h"
#include "python_boundaries.h"

namespace plask { namespace python {

// template <typename SolverT> struct ExportSolver;

namespace detail {

    template <typename SolverT, typename EnableSpace=void, typename EnableMesh=void>
    struct ExportedSolverDefaultDefs
    {
        template <typename PySolver>
        static auto init(PySolver& solver) -> PySolver& { return solver; }
    };

    template <typename SolverT, typename EnableMesh>
    struct ExportedSolverDefaultDefs<SolverT,
        typename std::enable_if<std::is_base_of<SolverOver<typename SolverT::SpaceType>, SolverT>::value>::type,
        EnableMesh>
    {
        template <typename PySolver>
        static auto init(PySolver& solver) -> PySolver& {
            solver.add_property("geometry", &SolverT::getGeometry, &SolverT::setGeometry, "Geometry provided to the solver");
            return solver;
        }
    };

    template <typename SolverT>
    struct ExportedSolverDefaultDefs<SolverT,
        typename std::enable_if<std::is_base_of<SolverOver<typename SolverT::SpaceType>, SolverT>::value>::type,
        typename std::enable_if<std::is_base_of<SolverWithMesh<typename SolverT::SpaceType, typename SolverT::MeshType>, SolverT>::value>::type>
    {
        static void Solver_setMesh(SolverT& self, py::object omesh) {
            try {
                shared_ptr<typename SolverT::MeshType> mesh = py::extract<shared_ptr<typename SolverT::MeshType>>(omesh);
                self.setMesh(mesh);
            } catch (py::error_already_set&) {
                PyErr_Clear();
                try {
                    shared_ptr<MeshGeneratorD<SolverT::MeshType::DIM>> meshg = py::extract<shared_ptr<MeshGeneratorD<SolverT::MeshType::DIM>>>(omesh);
                    self.setMesh(meshg);
                } catch (py::error_already_set&) {
                    throw TypeError(u8"Cannot convert argument to proper mesh type.");
                }
            }
        }

        template <typename PySolver>
        static auto init(PySolver& solver) -> PySolver& {
            solver.add_property("geometry", &SolverT::getGeometry, &SolverT::setGeometry, u8"Geometry provided to the solver");
            solver.add_property("mesh", &SolverT::getMesh, &Solver_setMesh, u8"Mesh provided to the solver");
            return solver;
        }
    };

} // namespace detail

constexpr const char* docstring_attr_receiver() { return
    u8"Receiver of the {2} required for computations [{3}].\n"
    u8"{4}\n\n"

    u8"You will find usage details in the documentation of the receiver class\n"
    u8":class:`~plask.flow.{0}Receiver{1}`.\n\n"

    u8"Example:\n"
    u8"   Connect the reveiver to a provider from some other solver:\n\n"

    u8"   >>> solver.{5} = other_solver.out{0}\n\n"

    u8"See also:\n\n"
    u8"   Receciver class: :class:`plask.flow.{0}Receiver{1}`\n\n"
    u8"   Provider class: :class:`plask.flow.{0}Provider{1}`\n\n"
    u8"   Data filter: :class:`plask.filter.{0}Filter{1}`\n";
}

template <PropertyType propertyType> constexpr const char* docstring_attr_provider_impl();

template <> constexpr const char* docstring_attr_provider_impl<SINGLE_VALUE_PROPERTY>() { return
    u8"Provider of the computed {2} [{3}].\n"
    u8"{4}\n\n"

    u8"{7}({5})\n\n"

    u8"{6}\n"

    u8":return: Value of the {2} **[{3}]**.\n\n"

    u8"Example:\n"
    u8"   Connect the provider to a receiver in some other solver:\n\n"

    u8"   >>> other_solver.in{0} = solver.{7}%\n\n"

    u8"   Obtain the provided value:\n\n"

    u8"   >>> solver.{7}({5})\n"
    u8"   1000\n\n"

    u8"See also:\n\n"
    u8"   Provider class: :class:`plask.flow.{0}Provider{1}`\n\n"
    u8"   Receciver class: :class:`plask.flow.{0}Receiver{1}`\n";
}

template <> constexpr const char* docstring_attr_provider_impl<MULTI_VALUE_PROPERTY>() { return
    u8"Provider of the computed {2} [{3}].\n"
    u8"{4}\n\n"

    u8"{7}(n=0{5})\n\n"

    u8"{9}"
    u8"{6}\n"

    u8":return: Value of the {2} **[{3}]**.\n\n"

    u8"You may obtain the number of different values this provider can return by\n"
    u8"testing its length.\n\n"

    u8"Example:\n"
    u8"   Connect the provider to a receiver in some other solver:\n\n"

    u8"   >>> other_solver.in{0} = solver.{7}\n\n"

    u8"   Obtain the provided value:\n\n"

    u8"   >>> solver.{7}(n=0{5})\n"
    u8"   1000\n\n"

    u8"   Test the number of provided values:\n\n"

    u8"   >>> len(solver.{7})\n"
    u8"   3\n\n"

    u8"See also:\n\n"
    u8"   Provider class: :class:`plask.flow.{0}Provider{1}`\n\n"
    u8"   Receciver class: :class:`plask.flow.{0}Receiver{1}`\n";
}

template <> constexpr const char* docstring_attr_provider_impl<FIELD_PROPERTY>() { return
    u8"Provider of the computed {2} [{3}].\n"
    u8"{4}\n\n"

    u8"{7}(mesh{5}, interpolation='default')\n\n"

    u8":param mesh mesh: Target mesh to get the field at.\n"
    u8":param str interpolation: Requested interpolation method.\n"
    u8"{6}\n"

    u8":return: Data with the {2} on the specified mesh **[{3}]**.\n\n"

    u8"Example:\n"
    u8"   Connect the provider to a receiver in some other solver:\n\n"

    u8"   >>> other_solver.in{0} = solver.{7}\n\n"

    u8"   Obtain the provided field:\n\n"

    u8"   >>> solver.{7}(mesh{5})\n"
    u8"   <plask.Data at 0x1234567>\n\n"

    u8"See also:\n\n"
    u8"   Provider class: :class:`plask.flow.{0}Provider{1}`\n\n"
    u8"   Receciver class: :class:`plask.flow.{0}Receiver{1}`\n";
}

template <> constexpr const char* docstring_attr_provider_impl<MULTI_FIELD_PROPERTY>() { return
    u8"Provider of the computed {2} [{3}].\n"
    u8"{4}\n\n"

    u8"{7}(n=0, mesh{5}, interpolation='default')\n\n"

    u8"{9}"
    u8":param mesh mesh: Target mesh to get the field at.\n"
    u8":param str interpolation: Requested interpolation method.\n"
    u8"{6}\n"

    u8":return: Data with the {2} on the specified mesh **[{3}]**.\n\n"

    u8"You may obtain the number of different values this provider can return by\n"
    u8"testing its length.\n\n"

    u8"Example:\n"
    u8"   Connect the provider to a receiver in some other solver:\n\n"

    u8"   >>> other_solver.in{0} = solver.{7}\n\n"

    u8"   Obtain the provided field:\n\n"

    u8"   >>> solver.{7}(0, mesh{5})\n"
    u8"   <plask.Data at 0x1234567>\n\n"

    u8"   Test the number of provided values:\n\n"

    u8"   >>> len(solver.{7})\n"
    u8"   3\n\n"

    u8"See also:\n\n"
    u8"   Provider class: :class:`plask.flow.{0}Provider{1}`\n\n"
    u8"   Receciver class: :class:`plask.flow.{0}Receiver{1}`\n";
}


template <typename PropertyTag>
static constexpr const char* docstring_attr_provider() {
    return docstring_attr_provider_impl<PropertyTag::propertyType>();
}

}} // namespace plask::python

#include "python_property_desc.h"

namespace plask { namespace python {

/**
 * This class should be instantiated to export a solver to Python.
 *
 * It automatically determines one of the standard bases for the solver and exports its methods
 * and fields. Furthermore it provides convenient methods to export providers and receivers.
 */
template <typename SolverT>
struct ExportSolver : public py::class_<SolverT, shared_ptr<SolverT>, py::bases<Solver>, boost::noncopyable> {

    typedef SolverT Class;
    typedef py::class_<SolverT, shared_ptr<SolverT>, py::bases<Solver>, boost::noncopyable> PyClass;

    template <typename... Args>
    ExportSolver(Args&&... args) : PyClass(std::forward<Args>(args)...) {
        detail::ExportedSolverDefaultDefs<SolverT>::init(*this);
    }

    template <typename ProviderT, typename ClassT>
    typename std::enable_if<std::is_base_of<ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType>, ProviderT>::value, ExportSolver>::type&
    add_provider(const char* name, ProviderT ClassT::* field, const char* addhelp) {

        typedef ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType> ClassT::* BaseTypePtr;

        this->def_readonly(name, reinterpret_cast<BaseTypePtr>(field),
            format(docstring_attr_provider<typename ProviderT::PropertyTag>(),
                   type_name<typename ProviderT::PropertyTag>(),                                // {0} Gain
                   spaceSuffix<typename ProviderT::SpaceType>(),                                // {1} Cartesian2D
                   ProviderT::PropertyTag::NAME,                                                // {2} material gain
                   ProviderT::PropertyTag::UNIT,                                                // {3} 1/cm
                   addhelp,                                                                     // {4} Gain in the active region.
                   docstrig_property_optional_args<typename ProviderT::PropertyTag>(),          // {5} wavelength
                   docstrig_property_optional_args_desc<typename ProviderT::PropertyTag>(),     // {6} :param: wavelength
                   name,                                                                        // {7} inGain
                   docstring_provider_multi_param<typename ProviderT::PropertyTag>(),           // {8} deriv=''
                   docstring_provider_multi_param_desc<typename ProviderT::PropertyTag>()       // {9} :param str deriv
                  ).c_str()
        );
        return *this;
    }

    template <typename ProviderT, typename ClassT>
    typename std::enable_if<!std::is_base_of<ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType>, ProviderT>::value, ExportSolver>::type&
    add_provider(const char* name, ProviderT ClassT::* field, const char* addhelp) {

        static_assert(std::is_base_of<Provider, ProviderT>::value, "add_provider used for non-provider type");

        this->def_readonly(name, field, addhelp);
        return *this;
    }

    template <typename ReceiverT, typename ClassT>
    ExportSolver& add_receiver(const char* name, ReceiverT ClassT::* field, const char* addhelp) {

        static_assert(std::is_base_of<ReceiverBase, ReceiverT>::value, "add_receiver used for non-receiver type");

        this->add_property(name, py::make_getter<ReceiverT ClassT::*>(field),
                           py::make_function(detail::ReceiverSetter<Class,ClassT,ReceiverT>(field),
                                             py::default_call_policies(),
                                             boost::mpl::vector3<void, Class&, py::object>()
                                            ),
                           format(docstring_attr_receiver(), type_name<typename ReceiverT::ProviderType::PropertyTag>(),
                                  spaceSuffix<typename ReceiverT::SpaceType>(), ReceiverT::ProviderType::PropertyTag::NAME,
                                  ReceiverT::ProviderType::PropertyTag::UNIT, addhelp, name).c_str()
                          );
        return *this;
    }

    template <typename Boundary, typename ValueT>
    ExportSolver& add_boundary_conditions(const char* name, BoundaryConditions<Boundary,ValueT> Class::* field, const char* help) {

        std::string boundary_class;
        if (PyTypeObject* mesh = py::converter::registry::lookup(py::type_id<typename Boundary::MeshType>()).m_class_object) {
            std::string nam = py::extract<std::string>(PyObject_GetAttrString((PyObject*)mesh, "__name__"));
            std::string mod = py::extract<std::string>(PyObject_GetAttrString((PyObject*)mesh, "__module__"));
            boundary_class = " (:class:`" + mod + "." + nam + ".Boundary`)";
        } else
            boundary_class = "";

        std::string value_class, value_class_desc;
        if (PyTypeObject* mesh = py::converter::registry::lookup(py::type_id<ValueT>()).m_class_object) {
            std::string nam = py::extract<std::string>(PyObject_GetAttrString((PyObject*)mesh, "__name__"));
            std::string mod = py::extract<std::string>(PyObject_GetAttrString((PyObject*)mesh, "__module__"));
            value_class = " (:class:`" + mod + "." + nam + "`)";
            value_class_desc = "\n.. autoclass:: " + mod + "." + nam + "\n";
        } else {
            value_class = "";
            value_class_desc = "";
        }

        detail::RegisterBoundaryConditions<Boundary, ValueT>();

        this->def_readonly(name, field, format(
            u8"{0} \n\n"

            u8"This field holds a list of boundary conditions for the solver. You may access\n"
            u8"and alter is elements a normal Python list. Each element is a special class\n"
            u8"that has two attributes:\n\n"

            u8"============= ==================================================================\n"
            u8":attr:`place` Boundary condition location{2}.\n"
            u8":attr:`value` Boundary condition value{3}.\n"
            u8"============= ==================================================================\n\n"

            u8"When you add new boundary condition, you may use two-argument ``append``, or\n"
            u8"``prepend`` methods, or three-argument ``insert`` method, where you separately\n"
            u8"specify the place and the value. See the below example for clarification.\n\n"

            u8"Example:\n"
            u8"    >>> solver.{1}.clear()\n"
            u8"    >>> solver.{1}.append(solver.mesh.Bottom(), some_value)\n"
            u8"    >>> solver.{1}[0].value = different_value\n"
            u8"    >>> solver.{1}.insert(0, solver.mesh.Top(), new_value)\n"
            u8"    >>> solver.{1}[1].value == different_value\n"
            u8"    True\n"
            u8"{4}",
            help, name, boundary_class, value_class, value_class_desc).c_str()
        );
        return *this;
    }

};

// Here are some useful defines.
// Note that if you use them to define methods, properties etc, you should also use MODULE
#define CLASS(cls, name, help) typedef cls __Class__; \
        ExportSolver<cls> solver(name, \
        name "(name=\"\")\n\n" help, \
        py::init<std::string>(py::arg("name")=""));
#define METHOD(name, method, help, ...) solver.def(BOOST_PP_STRINGIZE(name), &__Class__::method, help, (py::arg("arg1") , ## __VA_ARGS__))
#define RO_PROPERTY(name, get, help) solver.add_property(BOOST_PP_STRINGIZE(name), &__Class__::get, help)
#define RW_PROPERTY(name, get, set, help) solver.add_property(BOOST_PP_STRINGIZE(name), &__Class__::get, &__Class__::set, help)
#define RO_FIELD(name, help) solver.def_readonly(BOOST_PP_STRINGIZE(name), &__Class__::name, help)
#define RW_FIELD(name, help) solver.def_readwrite(BOOST_PP_STRINGIZE(name), &__Class__::name, help)
#define PROVIDER(name, addhelp) solver.add_provider(BOOST_PP_STRINGIZE(name), &__Class__::name, addhelp)
#define RECEIVER(name, addhelp) solver.add_receiver(BOOST_PP_STRINGIZE(name), &__Class__::name, addhelp)
#define BOUNDARY_CONDITIONS(name, help) solver.add_boundary_conditions(BOOST_PP_STRINGIZE(name), &__Class__::name, help)


using py::arg; // for more convenient specification of default arguments

}} // namespace plask::python

#endif // PLASK__PYTHON_SOLVERS_H
