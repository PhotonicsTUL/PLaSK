// (c) 2006 Dr. Andreas Beyer <mail@a-beyer.de>

#include <algorithm>
#include <exception>
#include <set>
#include <typeinfo>
#include <boost/python.hpp>

namespace plask { namespace python {

template<class KeyType> class py_set : public std::set<KeyType>,
 public boost::python::wrapper< std::set<KeyType> >  {

   public:
      // some typedefs for convinience
      typedef std::set<KeyType> source_type;
      typedef py_set<KeyType> wrap_type;

      // constructors
      py_set<KeyType> () {};
      py_set<KeyType> (const source_type& s )
          { insert(s.begin(), s.end()); }

      // element access
      bool contains(const KeyType key)
          { return count(key)>0; }
      // we must define add() for it gets explicit argument types
      void add(const KeyType key)
          { insert(key); }
      void remove(const KeyType key)
          // improve error handling here
          { if (!contains(key)) throw "element not in set"; erase(key);  }

      // set operations
      source_type set_union(wrap_type &other) {
              source_type result;
              std::set_union(this->begin(), this->end(), other.begin(), other.end(),
                    inserter(result, result.begin()));
              return result;
      }

      source_type set_intersection(wrap_type &other) {
              source_type result;
              std::set_intersection(this->begin(), this->end(), other.begin(), other.end(),
                    inserter(result, result.begin()));
              return result;
      }

      source_type set_difference(wrap_type &other) {
              source_type result;
              std::set_difference(this->begin(), this->end(), other.begin(), other.end(),
                    inserter(result, result.begin()));
              return result;
      }

      source_type set_symmetric_difference(wrap_type &other) {
              source_type result;
              std::set_symmetric_difference(this->begin(), this->end(), other.begin(), other.end(),
                        inserter(result, result.begin()));
              return result;
      }

};

inline void block_hashing(boost::python::object) {
    // do something more intelligent here
    throw "Objects of this type are unhashable";
}

// export mutable set
template<class KeyType> void
export_set(const char* py_name) {
   typedef py_set<KeyType> set_T;

   boost::python::class_<set_T > (py_name, "mutable set")
       .def("__len__",        &set_T::size)
       .def("__contains__",    &set_T::contains)
       .def("add",        &set_T::add, "add element")
       .def("__delitem__",    &set_T::remove)
       .def("remove",        &set_T::remove, "remove element")
       .def("__iter__",        boost::python::iterator<set_T> ())

       .def("__hash__",        &block_hashing)

       .def("union",        &set_T::set_union, "set union")
       .def("__or__",        &set_T::set_union, "set union")
       .def("intersection",    &set_T::set_intersection, "set intersection")
       .def("__and__",        &set_T::set_intersection, "set intersection")
       .def("difference",    &set_T::set_difference, "elements not in second set")
       .def("__sub__",        &set_T::set_difference, "set difference")
       .def("symmetric_difference",    &set_T::set_symmetric_difference, "elements unique to either set")
       .def("__xor__",        &set_T::set_symmetric_difference, "symmetric set difference")
   ;

   boost::python::implicitly_convertible<py_set<KeyType>, std::set<KeyType> >();
   boost::python::implicitly_convertible<std::set<KeyType>, py_set<KeyType> >();
}

// export immutable set
template<class KeyType> void
export_frozenset(const char* py_name) {
   typedef py_set<KeyType> set_T;

   boost::python::class_<set_T > (py_name, "immutable set")
       .def("__len__",        &set_T::size)
       .def("__contains__",    &set_T::contains)
       .def("__iter__",        boost::python::iterator<set_T> ())

       .def(boost::python::self < boost::python::self)
       .def(boost::python::self == boost::python::self)

       .def("union",        &set_T::set_union, "set union")
       .def("__or__",        &set_T::set_union, "set union")
       .def("intersection",    &set_T::set_intersection, "set intersection")
       .def("__and__",        &set_T::set_intersection, "set intersection")
       .def("difference",    &set_T::set_difference, "elements not in second set")
       .def("__sub__",        &set_T::set_difference, "set different")
       .def("symmetric_difference",    &set_T::set_symmetric_difference, "elements unique to either set")
       .def("__xor__",        &set_T::set_symmetric_difference, "symmetric set different")
   ;

   boost::python::implicitly_convertible<py_set<KeyType>, std::set<KeyType> >();
   boost::python::implicitly_convertible<std::set<KeyType>, py_set<KeyType> >();

}

}} // namespace plask::python