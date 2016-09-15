// (c) 2006 Dr. Andreas Beyer <mail@a-beyer.de>

#include <algorithm>
#include <exception>
#include <set>
#include <typeinfo>
#include <boost/python.hpp>

namespace plask { namespace python {

namespace py = boost::python;


template<typename KeyT>
struct set_to_python_list_conventer {

    static PyObject* convert(const std::set<KeyT>& set)  {
        boost::python::list list;
        for (auto item: set) {
            list.append(boost::python::object(item));
        }
        return boost::python::incref(list.ptr());
    }

    set_to_python_list_conventer() {
        boost::python::to_python_converter<std::set<KeyT>, set_to_python_list_conventer<KeyT>>();
    }
};


namespace detail {

    template<class KeyT> class py_set  {

    public:
        // some typedefs for convinience
        typedef std::set<KeyT> setT;

        // object access
        static bool contains(const setT& self, const KeyT key) { return self.count(key)>0; }
        // we must define add() for it gets explicit argument types
        static void add(setT& self, const KeyT key) { self.insert(key); }
        static void remove(setT& self, const KeyT key) { // improve error handling here
            if (!contains(self, key)) {
                PyErr_SetObject(PyExc_KeyError, py::str(py::object(key)).ptr());
                throw py::error_already_set();
            }
            self.erase(key);
        }

        static std::string repr(const setT& self) {
            std::stringstream out;
            out << "set([";
            int i = self.size()-1;
            if (i == -1) out << "])";
            else
                for(auto item = self.begin(); item != self.end(); ++item, --i) {
                    out << std::string(py::extract<std::string>(py::object(*item).attr("__repr__")())) << (i?", ":"])");
                }
            return out.str();
        }

        static std::string str(const setT& self) {
            std::stringstream out;
            out << "{ ";
            int i = self.size()-1;
            if (i == -1) out << "}";
            else
                for(auto item = self.begin(); item != self.end(); ++item, --i) {
                    out << std::string(py::extract<std::string>(py::object(*item).attr("__repr__")())) << (i?", ":" }");
                }
            return out.str();
        }

        // set operations
        static setT set_union(const setT& self, const setT &other) {
                setT result;
                std::set_union(self.begin(), self.end(), other.begin(), other.end(), std::inserter(result, result.begin()));
                return result;
        }

        static setT set_intersection(const setT& self, const setT &other) {
                setT result;
                std::set_intersection(self.begin(), self.end(), other.begin(), other.end(), std::inserter(result, result.begin()));
                return result;
        }

        static setT set_difference(const setT& self, const setT &other) {
                setT result;
                std::set_difference(self.begin(), self.end(), other.begin(), other.end(), std::inserter(result, result.begin()));
                return result;
        }

        static setT set_symmetric_difference(const setT& self, const setT &other) {
                setT result;
                std::set_symmetric_difference(self.begin(), self.end(), other.begin(), other.end(), std::inserter(result, result.begin()));
                return result;
        }

    };
}

inline void block_hashing(boost::python::object) {
    // do something more intelligent here
    throw "Objects of this type are unhashable";
}

// export mutable set
template<class KeyT> void
export_set(const char* py_name) {
   typedef detail::py_set<KeyT> PySetT;
   typedef std::set<KeyT> SetT;

   boost::python::class_<SetT> (py_name, "Mutable set.")
       .def("__len__", &SetT::size)
       .def("__contains__",&PySetT::contains)
       .def("add", &PySetT::add, "Add object to set.")
       .def("__delitem__", &PySetT::remove)
       .def("remove", &PySetT::remove, "Remove object from set.")
       .def("__iter__", boost::python::iterator<SetT>())

       .def("__str__", &PySetT::str)
       .def("__repr__", &PySetT::repr)
       .def("__hash__", &block_hashing)

       .def("union", &PySetT::set_union, "Return the union of sets as a new set.")
       .def("__add__", &PySetT::set_union)
       .def("intersection", &PySetT::set_intersection, "Return the union of sets as a new set.")
       .def("__mul__", &PySetT::set_intersection)
       .def("difference", &PySetT::set_difference, "Return the difference of sets as a new set.")
       .def("__sub__", &PySetT::set_difference, "set difference")
       .def("symmetric_difference", &PySetT::set_symmetric_difference, "Return objects unique to either set.")
   ;
}

// export immutable set
template<class KeyT> void
export_frozenset(const char* py_name) {
   typedef detail::py_set<KeyT> PySetT;
   typedef std::set<KeyT> SetT;

   boost::python::class_<SetT> (py_name, "Immutable set")
       .def("__len__", &SetT::size)
       .def("__contains__", &PySetT::contains)
       .def("__iter__", boost::python::iterator<SetT>())

       .def(boost::python::self < boost::python::self)
       .def(boost::python::self == boost::python::self)

       .def("__str__", &PySetT::str)
       .def("__repr__", &PySetT::repr)

       .def("union", &PySetT::set_union, "Return the union of sets as a new set.")
       .def("__add__", &PySetT::set_union)
       .def("intersection", &PySetT::set_intersection, "Return the union of sets as a new set.")
       .def("__mul__", &PySetT::set_intersection)
       .def("difference", &PySetT::set_difference, "Return the difference of sets as a new set.")
       .def("__sub__", &PySetT::set_difference, "set difference")
       .def("symmetric_difference", &PySetT::set_symmetric_difference, "Return objects unique to either set.")
   ;
}

}} // namespace plask::python