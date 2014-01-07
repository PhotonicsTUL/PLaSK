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
            solver.def("setGeometry", &SolverT::setGeometry, "Set geometry for the solver", py::arg("geometry"));
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
            } catch (py::error_already_set) {
                PyErr_Clear();
                try {
                    shared_ptr<MeshGeneratorOf<typename SolverT::MeshType>> meshg = py::extract<shared_ptr<MeshGeneratorOf<typename SolverT::MeshType>>>(omesh);
                    self.setMesh(meshg);
                } catch (py::error_already_set) {
                    throw TypeError("Cannot convert argument to proper mesh type.");
                }
            }
        }

        template <typename PySolver>
        static auto init(PySolver& solver) -> PySolver& {
            solver.add_property("geometry", &SolverT::getGeometry, &SolverT::setGeometry, "Geometry provided to the solver");
            solver.add_property("mesh", &SolverT::getMesh, &Solver_setMesh, "Mesh provided to the solver");
            return solver;
        }
    };

} // namespace detail

extern const char* docstring_attr_receiver;
template <PropertyType propertyType> const char* docstring_attr_provider();

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
            format(docstring_attr_provider<ProviderT::PropertyTag::propertyType>(),
                   type_name<typename ProviderT::PropertyTag>(),                                // %1% Gain
                   spaceSuffix<typename ProviderT::SpaceType>(),                                // %2% Cartesian2D
                   ProviderT::PropertyTag::NAME,                                                // %3% material gain
                   ProviderT::PropertyTag::UNIT,                                                // %4% 1/cm
                   addhelp,                                                                     // %5% Gain in the active region.
                   docstrig_property_optional_args<typename ProviderT::PropertyTag>(),          // %6% wavelength
                   docstrig_property_optional_args_desc<typename ProviderT::PropertyTag>(),     // %7% :param: wavelength
                   name                                                                         // %8% inGain
                  ).c_str()
        );
        return *this;
    }

    template <typename ProviderT, typename ClassT>
    typename std::enable_if<!std::is_base_of<ProviderFor<typename ProviderT::PropertyTag, typename ProviderT::SpaceType>, ProviderT>::value, ExportSolver>::type&
    add_provider(const char* name, ProviderT ClassT::* field, const char* addhelp) {

        static_assert(std::is_base_of<Provider, ProviderT>::value, "add_provider used for non-provider type");

        this->def_readonly(name, field,
            format(docstring_attr_provider<ProviderT::PropertyTag::propertyType>(),
                   type_name<typename ProviderT::PropertyTag>(),                                // %1% Gain
                   spaceSuffix<typename ProviderT::SpaceType>(),                                // %2% Cartesian2D
                   ProviderT::PropertyTag::NAME,                                                // %3% material gain
                   ProviderT::PropertyTag::UNIT,                                                // %4% 1/cm
                   addhelp,                                                                     // %5% Gain in the active region.
                   docstrig_property_optional_args<typename ProviderT::PropertyTag>(),          // %6% wavelength
                   docstrig_property_optional_args_desc<typename ProviderT::PropertyTag>(),     // %7% :param: wavelength
                   name                                                                         // %8% inGain
                  ).c_str()
        );
        return *this;
    }

    template <typename ReceiverT, typename ClassT>
    ExportSolver& add_receiver(const char* name, ReceiverT ClassT::* field, const char* addhelp) {

        static_assert(std::is_base_of<ReceiverBase, ReceiverT>::value, "add_receiver used for non-receiver type");

        this->add_property(name, py::make_getter(field),
                           py::make_function(detail::ReceiverSetter<Class,ClassT,ReceiverT>(field),
                                             py::default_call_policies(),
                                             boost::mpl::vector3<void, Class&, py::object>()
                                            ),
                           format(docstring_attr_receiver, type_name<typename ReceiverT::ProviderType::PropertyTag>(),
                                  spaceSuffix<typename ReceiverT::SpaceType>(), ReceiverT::ProviderType::PropertyTag::NAME,
                                  ReceiverT::ProviderType::PropertyTag::UNIT, addhelp, name).c_str()
                          );
        return *this;
    }

    template <typename MeshT, typename ValueT>
    ExportSolver& add_boundary_conditions(const char* name, BoundaryConditions<MeshT,ValueT> Class::* field, const char* help) {

        std::string boundary_class;
        if (PyTypeObject* mesh = py::converter::registry::lookup(py::type_id<MeshT>()).m_class_object) {
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

        detail::RegisterBoundaryConditions<MeshT, ValueT>();

        this->def_readonly(name, field, format(
            "%1% \n\n"

            "This field holds a list of boundary conditions for the solver. You may access\n"
            "and alter is elements a normal Python list. Each element is a special class\n"
            "that has two attributes:\n\n"

            "============= ==================================================================\n"
            ":attr:`place` Boundary condition location%3%.\n"
            ":attr:`value` Boundary condition value%4%.\n"
            "============= ==================================================================\n\n"

            "When you add new boundary condition, you may use two-argument ``append``, or\n"
            "``prepend`` methods, or three-argument ``insert`` method, where you separately\n"
            "specify the place and the value. See the below example for clarification.\n\n"

            "Example:\n"
            "    >>> solver.%2%.clear()\n"
            "    >>> solver.%2%.append(solver.mesh.Bottom(), some_value)\n"
            "    >>> solver.%2%[0].value = different_value\n"
            "    >>> solver.%2%.insert(0, solver.mesh.Top(), new_value)\n"
            "    >>> solver.%2%[1].value == different_value\n"
            "    True\n"
            "%5%",
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
