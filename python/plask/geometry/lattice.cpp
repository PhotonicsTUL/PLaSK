/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "geometry.hpp"

#include "plask/geometry/lattice.hpp"

namespace plask { namespace python {

template <int dim> inline const char* ArangeName();
template <> inline const char* ArangeName<2>() { return "Arange2D"; }
template <> inline const char* ArangeName<3>() { return "Arange3D"; }

template <int dim> inline const char* ArangeDoc();
template <> inline const char* ArangeDoc<2>() {
    return u8"Arange2D(item, step, count)\n"
           u8"Container that repeats its item, shifting each repetition by the specified step.\n\n"
           u8"Args:\n"
           u8"    item (GeometryObject2D): Object to repeat.\n"
           u8"    step (vec): 2D vector, by which each repetition is shifted from the previous\n"
           u8"                one.\n"
           u8"    count (int): Number of item repetitions.\n"
           u8"    warning (bool): Boolean value indicating is overlapping warnings\n"
           u8"                    are displayed.\n";
}
template <> inline const char* ArangeDoc<3>() {
    return u8"Arange3D(item, step, count)\n"
           u8"Container that repeats its item, shifting each repetition by the specified step.\n\n"
           u8"Args:\n"
           u8"    item (GeometryObject3D): Object to repeat.\n"
           u8"    step (vec): 3D vector, by which each repetition is shifted from the previous\n"
           u8"                one.\n"
           u8"    count (int): Number of item repetitions.\n"
           u8"    warning (bool): Boolean value indicating is overlapping warnings\n"
           u8"                    are displayed.\n";
}

template <int dim> inline static void init_Arange() {
    py::class_<ArrangeContainer<dim>, shared_ptr<ArrangeContainer<dim>>, py::bases<GeometryObjectTransform<dim>>,
               boost::noncopyable>(
        ArangeName<dim>(), ArangeDoc<dim>(),
        py::init<const shared_ptr<typename ArrangeContainer<dim>::ChildType>&, const typename ArrangeContainer<dim>::DVec,
                 unsigned>((py::arg("item"), "step", "count", py::arg("warning") = true)))
        .add_property("step", &ArrangeContainer<dim>::getTranslation, &ArrangeContainer<dim>::setTranslation,
                      "Vector, by which each repetition is shifted from the previous one.")
        .add_property("count", &ArrangeContainer<dim>::getRepeatCount, &ArrangeContainer<dim>::setRepeatCount,
                      "Number of item repetitions.")
        .def_readwrite("warning", &ArrangeContainer<dim>::warn_overlapping,
                       "Boolean value indicating is overlapping warnings are displayed.")
        .def("__len__", &ArrangeContainer<dim>::getChildrenCount);
}

// shared_ptr<GeometryObject> GeometryObject__getitem__(py::object oself, int i);

class LatticeVertices {
    shared_ptr<Lattice> lattice;
    size_t segment;

    friend class LatticeSegments;

    std::vector<LateralVec<int>>& getSegment() const {
        if (segment >= lattice->segments.size()) throw IndexError("lattice segment has been removed");
        return lattice->segments[segment];
    }

    size_t index(int i) const {
        size_t n = getSegment().size();
        if (i < 0) i += n;
        if (i < 0 || i >= n) throw IndexError("vertex index out of range");
        return i;
    }

  public:
    static std::map<Lattice*, std::deque<LatticeVertices*>> pool;

    LatticeVertices(shared_ptr<Lattice> lattice, size_t segment) : lattice(lattice), segment(segment) {
        pool[lattice.get()].push_back(this);
    }

    LatticeVertices(const LatticeVertices& src) : lattice(src.lattice), segment(src.segment) {
        pool[lattice.get()].push_back(this);
    }

    ~LatticeVertices() {
        auto& pool = LatticeVertices::pool[lattice.get()];
        auto ptr = std::find(pool.begin(), pool.end(), this);
        if (ptr != pool.end()) pool.erase(ptr);
        if (pool.empty()) LatticeVertices::pool.erase(lattice.get());
    }

    size_t __len__() const { return getSegment().size(); }

    LateralVec<int> __getitem__(int i) const { return LateralVec<int>(getSegment()[index(i)]); }

    void __setitem__(int i, const LateralVec<int>& v) {
        size_t idx = index(i);
        lattice->segments[segment][idx] = v;
        lattice->refillContainer();
    }

    void append(const LateralVec<int>& v) {
        getSegment().push_back(v);
        lattice->refillContainer();
    }

    void insert(int i, const Vec<2, double>& v) {
        std::vector<LateralVec<int>>& segment = getSegment();
        size_t n = segment.size();
        if (i < 0) i += n;
        if (i < 0 || i > n) throw IndexError("vertex index out of range");
        if (i == n)
            segment.push_back(v);
        else
            segment.insert(segment.begin() + i, v);
        lattice->refillContainer();
    }

    void __delitem__(int i) {
        getSegment().erase(getSegment().begin() + index(i));
        lattice->refillContainer();
    }

    std::string __str__() const {
        std::string result = "[";
        for (size_t i = 0; i < getSegment().size(); ++i) {
            result += str(getSegment()[i]);
            result += (i != getSegment().size() - 1) ? ", " : "]";
        }
        return result;
    }

    std::string __repr__() const {
        std::string result = "[";
        for (size_t i = 0; i < getSegment().size(); ++i) {
            result += format("plask.vec({}, {})", getSegment()[i].c0, getSegment()[i].c1);
            result += (i != getSegment().size() - 1) ? ", " : "]";
        }
        return result;
    }

    struct Iterator {
        const LatticeVertices* vertices;
        size_t i;

        Iterator(const LatticeVertices* v) : vertices(v), i(0) {}

        LateralVec<int> __next__() {
            if (i >= vertices->getSegment().size()) throw StopIteration();
            return vertices->__getitem__(i++);
        }

        Iterator* __iter__() { return this; }
    };

    Iterator __iter__() const { return Iterator(this); }
};

class LatticeSegments {
    shared_ptr<Lattice> lattice;

    size_t index(int i) const {
        size_t n = lattice->segments.size();
        if (i < 0) i += n;
        if (i < 0 || i >= n) throw IndexError("vertex index out of range");
        return i;
    }

    inline static std::vector<LateralVec<int>> verticesFromSequence(py::object vertices) {
        std::vector<LateralVec<int>> result;
        for (py::stl_input_iterator<py::object> vertices_it(vertices), vertices_end_it; vertices_it != vertices_end_it;
             ++vertices_it) {
            if (py::len(*vertices_it) != 2)
                throw TypeError("each vertex in lattice segment must have exactly two integer coordinates");
            py::stl_input_iterator<int> coord_it(*vertices_it);
            result.emplace_back(*(coord_it++), *(coord_it++));
        }
        return result;
    }

  public:
    LatticeSegments(shared_ptr<Lattice> lattice) : lattice(lattice) {}

    static LatticeSegments fromLattice(shared_ptr<Lattice> lattice) { return LatticeSegments(lattice); }

    size_t __len__() const { return lattice->segments.size(); }

    LatticeVertices __getitem__(int i) const { return LatticeVertices(lattice, index(i)); }

    void __setitem__(int i, py::object value) {
        size_t idx = index(i);
        lattice->segments[idx] = std::move(verticesFromSequence(value));
        lattice->refillContainer();
    }

    void append(py::object value) {
        lattice->segments.push_back(verticesFromSequence(value));
        lattice->refillContainer();
    }

    void insert(int i, py::object value) {
        size_t n = lattice->segments.size();
        if (i < 0) i += n;
        if (i < 0 || i > n) throw IndexError("vertex index out of range");
        if (i == n) {
            lattice->segments.push_back(verticesFromSequence(value));
        } else {
            lattice->segments.insert(lattice->segments.begin() + i, verticesFromSequence(value));
            for (auto& item : LatticeVertices::pool[lattice.get()]) {
                if (item->segment >= i) ++item->segment;
            }
        }
        lattice->refillContainer();
    }

    void __delitem__(int i) {
        size_t idx = index(i);
        lattice->segments.erase(lattice->segments.begin() + idx);
        for (auto& item : LatticeVertices::pool[lattice.get()]) {
            if (item->segment == idx)
                item->segment = std::size_t(-1);
            else if (item->segment > idx)
                --item->segment;
        };
        lattice->refillContainer();
    }

    std::string __str__() const {
        std::string result = "[";
        for (size_t i = 0; i < lattice->segments.size(); ++i) {
            result += "[";
            for (size_t j = 0; j < lattice->segments[i].size(); ++j) {
                result += format("[{}, {}]", lattice->segments[i][j].c0, lattice->segments[i][j].c1);
                result += (j != lattice->segments[i].size() - 1) ? ", " : "]";
            }
            result += (i != lattice->segments.size() - 1) ? ", " : "]";
        }
        return result;
    }

    std::string __repr__() const {
        std::string result = "[";
        for (size_t i = 0; i < lattice->segments.size(); ++i) {
            result += "[";
            for (size_t j = 0; j < lattice->segments[i].size(); ++j) {
                result += format("vec({}, {})", lattice->segments[i][j].c0, lattice->segments[i][j].c1);
                result += (j != lattice->segments[i].size() - 1) ? ", " : "]";
            }
            result += (i != lattice->segments.size() - 1) ? ", " : "]";
        }
        return result;
    }

    struct Iterator {
        const LatticeSegments* segments;
        size_t i;

        Iterator(const LatticeSegments* ls) : segments(segments), i(0) {}

        LatticeVertices __next__() {
            if (i >= segments->__len__()) throw StopIteration();
            return segments->__getitem__(i++);
        }

        Iterator* __iter__() { return this; }
    };

    Iterator __iter__() const { return Iterator(this); }
};

std::map<Lattice*, std::deque<LatticeVertices*>> LatticeVertices::pool;

static void lattice_set_segments(Lattice& self, const py::object& value) {
    std::vector<std::vector<LateralVec<int>>> segments;
    for (py::stl_input_iterator<py::object> segments_it(value), segments_end_it; segments_it != segments_end_it; ++segments_it) {
        std::vector<LateralVec<int>> segment;
        for (py::stl_input_iterator<py::object> vertices_it(*segments_it), vertices_end_it; vertices_it != vertices_end_it;
             ++vertices_it) {
            if (py::len(*vertices_it) != 2)
                throw TypeError("each vertex in lattice segment must have exactly two integer coordinates");
            py::stl_input_iterator<int> coord_it(*vertices_it);
            segment.emplace_back(*(coord_it++), *(coord_it++));
        }
        segments.push_back(std::move(segment));
    }
    self.setSegments(std::move(segments));
}

static void lattice_set_vec0(Lattice& self, const Vec<3>& vec) {
    self.vec0 = vec;
    self.refillContainer();
}

static void lattice_set_vec1(Lattice& self, const Vec<3>& vec) {
    self.vec1 = vec;
    self.refillContainer();
}

void register_geometry_container_lattice() {
    init_Arange<2>();
    init_Arange<3>();

    py::class_<Lattice, shared_ptr<Lattice>, py::bases<GeometryObjectTransform<3>>, boost::noncopyable> lattice_class(
        "Lattice", "Lattice container that arranges its children in two-dimensional lattice.",
        py::init<const shared_ptr<typename Lattice::ChildType>&, const typename Lattice::DVec, const typename Lattice::DVec>(
            (py::arg("item"), py::arg("vec0") = plask::Primitive<3>::ZERO_VEC, py::arg("vec1") = plask::Primitive<3>::ZERO_VEC)));
    lattice_class.def("__len__", &Lattice::getChildrenCount)
        .add_property("segments", &LatticeSegments::fromLattice, lattice_set_segments,
                      "List of lattices limiting lattice segments.")
        .add_property("vec0", py::make_getter(&Lattice::vec0), lattice_set_vec0, "First lattice vector.")
        .add_property("vec1", py::make_getter(&Lattice::vec1), lattice_set_vec1, "Second lattice vector.");

    py::scope lattice_scope = lattice_class;

    py::class_<LatticeSegments> segments("Segments", py::no_init);
    segments  //
        .def("__len__", &LatticeSegments::__len__)
        .def("__getitem__", &LatticeSegments::__getitem__)
        .def("__setitem__", &LatticeSegments::__setitem__)
        .def("append", &LatticeSegments::append)
        .def("insert", &LatticeSegments::insert)
        .def("__delitem__", &LatticeSegments::__delitem__)
        .def("__str__", &LatticeSegments::__str__)
        .def("__repr__", &LatticeSegments::__repr__)
        .def("__iter__", &LatticeSegments::__iter__);
    {
        py::scope segments_scope = segments;
        py::class_<LatticeSegments::Iterator>("Iterator", py::no_init)  //
            .def("__iter__", &LatticeSegments::Iterator::__iter__, py::return_self<>())
            .def("__next__", &LatticeSegments::Iterator::__next__);
    }

    py::class_<LatticeVertices> vertices("Vertices", py::no_init);
    vertices  //
        .def("__len__", &LatticeVertices::__len__)
        .def("__getitem__", &LatticeVertices::__getitem__)
        .def("__setitem__", &LatticeVertices::__setitem__)
        .def("append", &LatticeVertices::append)
        .def("insert", &LatticeVertices::insert)
        .def("__delitem__", &LatticeVertices::__delitem__)
        .def("__str__", &LatticeVertices::__str__)
        .def("__repr__", &LatticeVertices::__repr__)
        .def("__iter__", &LatticeVertices::__iter__);
    {
        py::scope vertices_scope = vertices;
        py::class_<LatticeVertices::Iterator>("Iterator", py::no_init)  //
            .def("__iter__", &LatticeVertices::Iterator::__iter__, py::return_self<>())
            .def("__next__", &LatticeVertices::Iterator::__next__);
    }
}

}}  // namespace plask::python
