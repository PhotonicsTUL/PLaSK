#ifndef PLASK__PYTHON_MODULES_H
#define PLASK__PYTHON_MODULES_H

#include <type_traits>

#include <plask/solver.h>
#include "python_globals.h"
#include "python_provider.h"

#include <plask/provider/optical.h>

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
            } catch (py::error_already_set) {
                PyErr_Clear();
                try {
                    MeshGeneratorOf<typename SolverT::MeshType>* mesh = py::extract<MeshGeneratorOf<typename SolverT::MeshType>*>(omesh);
                    self.setMesh(*mesh);
                } catch (py::error_already_set) {
                    throw TypeError("Cannot convert argument to proper mesh type.");
                }
            }
        }

        template <typename PySolver>
        static auto init(PySolver& solver) -> PySolver& {
            solver.add_property("geometry", &SolverT::getGeometry, &SolverT::setGeometry, "Geometry provided to the solver");
            solver.add_property("mesh", &SolverT::getMesh, Solver_setMesh, "Mesh provided to the solver");
            return solver;
        }
    };

    template <typename Class, typename ReceiverT>
    struct ReceiverSetter
    {
        typedef typename ReceiverT::PropertyTag::ValueType ValueT;

        ReceiverSetter(ReceiverT Class::* field) : field(field) {}

        void operator()(Class& self, py::object obj) {
            try {
                ValueT value = py::extract<ValueT>(obj);
                self.*field = value;
            } catch (py::error_already_set) {
                PyErr_Clear();
                detail::RegisterReceiverImpl<ReceiverT, ReceiverT::PropertyTag::propertyType>::setValue(self.*field, obj);
            }
        }

      private:
        ReceiverT Class::* field;
    };

} // namespace detail

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

    template <typename ProviderT>
    ExportSolver& add_provider(const char* name, ProviderT Class::* field, const char* help) {
        RegisterProvider<ProviderT>();
        this->def_readonly(name, field, help);
        return *this;
    }

    template <typename ReceiverT>
    ExportSolver& add_receiver(const char* name, ReceiverT Class::* field, const char* help) {
        RegisterReceiver<ReceiverT>();
        this->add_property(name, py::make_getter(field),
                           py::make_function(detail::ReceiverSetter<Class,ReceiverT>(field),
                                             py::default_call_policies(),
//                                              boost::mpl::vector3<void, Class&, typename ReceiverT::PropertyTag::ValueType const&>()
                                             boost::mpl::vector3<void, Class&, py::object>()
                                            ),
                           help
                          );
        return *this;
    }


};

// Here are some useful defines.
// Note that if you use them to define methods, properties etc, you should also use MODULE
#define CLASS(cls, name, help) typedef cls __Class__; ExportSolver<cls> __solver__(name, help);
#define METHOD(method, help, ...) __solver__.def(BOOST_PP_STRINGIZE(method), &__Class__::method, help, (py::arg("arg1") , ## __VA_ARGS__))
#define RO_PROPERTY(name, get, help) __solver__.add_property(BOOST_PP_STRINGIZE(name), &__Class__::get, help)
#define RW_PROPERTY(name, get, set, help) __solver__.add_property(BOOST_PP_STRINGIZE(name), &__Class__::get, &__Class__::set, help)
#define RO_FIELD(name, help) __solver__.def_readonly(BOOST_PP_STRINGIZE(name), &__Class__::name, help)
#define RW_FIELD(name, help) __solver__.def_readwrite(BOOST_PP_STRINGIZE(name), &__Class__::name, help)
#define PROVIDER(name, help) __solver__.add_provider(BOOST_PP_STRINGIZE(name), &__Class__::name, help)
#define RECEIVER(name, help) __solver__.add_receiver(BOOST_PP_STRINGIZE(name), &__Class__::name, help)


using py::arg; // for more convenient specification of default arguments

}} // namespace plask::python

#endif // PLASK__PYTHON_MODULES_H
