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
#include "triangular2d.hpp"

#include <boost/range/irange.hpp>
#include <boost/icl/interval_map.hpp>
#include <unordered_map>

#include "../utils/interpolation.hpp"
#include "rectangular_common.hpp"

namespace plask {

Vec<3, double> TriangularMesh2D::Element::barycentric(Vec<2, double> p) const {
    // formula comes from https://codeplea.com/triangular-interpolation
    // but it is modified a bit, to reuse more diffs and call cross (which can be optimized)
    Vec<2, double>
            diff_2_3 = getNode(1) - getNode(2),
            diff_p_3 = p - getNode(2),
            diff_1_3 = getNode(0) - getNode(2);
    const double den = cross(diff_1_3, diff_2_3);   // diff_2_3.c1 * diff_1_3.c0 - diff_2_3.c0 * diff_1_3.c1
    Vec<3, double> res;
    res.c0 = cross(diff_p_3, diff_2_3) / den;   // (diff_2_3.c1 * diff_p_3.c0 - diff_2_3.c0 * diff_p_3.c1) / den
    res.c1 = cross(diff_1_3, diff_p_3) / den;   // (- diff_1_3.c1 * diff_p_3.c0 + diff_1_3.c0 * diff_p_3.c1) / den
    res.c2 = 1.0 - res.c0 - res.c1;
    return res;
}

Box2D TriangularMesh2D::Element::getBoundingBox() const {
    LocalCoords lo = getNode(0);
    LocalCoords up = lo;
    const LocalCoords B = getNode(1);
    if (B.c0 < lo.c0) lo.c0 = B.c0; else up.c0 = B.c0;
    if (B.c1 < lo.c1) lo.c1 = B.c1; else up.c1 = B.c1;
    const LocalCoords C = getNode(2);
    if (C.c0 < lo.c0) lo.c0 = C.c0; else if (C.c0 > up.c0) up.c0 = C.c0;
    if (C.c1 < lo.c1) lo.c1 = C.c1; else if (C.c1 > up.c1) up.c1 = C.c1;
    return Box2D(lo, up);
}

TriangularMesh2D::Builder::Builder(TriangularMesh2D &mesh): mesh(mesh) {
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i)
        this->indexOfNode[mesh.nodes[i]] = i;
}

TriangularMesh2D::Builder::Builder(TriangularMesh2D &mesh, std::size_t predicted_number_of_elements, std::size_t predicted_number_of_nodes)
    : Builder(mesh)
{
    mesh.elementNodes.reserve(mesh.elementNodes.size() + predicted_number_of_elements);
    mesh.nodes.reserve(mesh.nodes.size() + predicted_number_of_nodes);
}

TriangularMesh2D::Builder::~Builder() {
    mesh.elementNodes.shrink_to_fit();
    mesh.nodes.shrink_to_fit();
}

TriangularMesh2D::Builder &TriangularMesh2D::Builder::add(TriangularMesh2D::LocalCoords p1, TriangularMesh2D::LocalCoords p2, TriangularMesh2D::LocalCoords p3) {
    mesh.elementNodes.push_back({addNode(p1), addNode(p2), addNode(p3)});
    return *this;
}

std::size_t TriangularMesh2D::Builder::addNode(TriangularMesh2D::LocalCoords node) {
    auto it = this->indexOfNode.emplace(node, mesh.nodes.size());
    if (it.second) // new element has been appended to the map
        this->mesh.nodes.push_back(node);
    return it.first->second;    // an index of the node (inserted or found)
}

struct ElementIndexValueGetter {
    TriangularMesh2D::ElementIndex::Rtree::value_type operator()(std::size_t index) const {
        TriangularMesh2D::Element el = this->src_mesh->getElement(index);
        const auto n0 = el.getNode(0);
        const auto n1 = el.getNode(1);
        const auto n2 = el.getNode(2);
        return std::make_pair(
                    TriangularMesh2D::ElementIndex::Box(
                        vec(std::min(std::min(n0.c0, n1.c0), n2.c0), std::min(std::min(n0.c1, n1.c1), n2.c1)),
                        vec(std::max(std::max(n0.c0, n1.c0), n2.c0), std::max(std::max(n0.c1, n1.c1), n2.c1))
                    ),
                    index);
    }

    const TriangularMesh2D* src_mesh;

    ElementIndexValueGetter(const TriangularMesh2D& src_mesh): src_mesh(&src_mesh) {}
};

TriangularMesh2D::ElementIndex::ElementIndex(const TriangularMesh2D &mesh)
    : mesh(mesh),
      rtree(makeFunctorIndexedIterator(ElementIndexValueGetter(mesh), 0),
            makeFunctorIndexedIterator(ElementIndexValueGetter(mesh), mesh.getElementsCount()))
{}

std::size_t TriangularMesh2D::ElementIndex::getIndex(Vec<2, double> p) const
{
    for (auto v: rtree | boost::geometry::index::adaptors::queried(boost::geometry::index::intersects(p))) {
        const Element el = mesh.getElement(v.second);
        if (el.contains(p)) return v.second;
    }
    return INDEX_NOT_FOUND;
}

optional<TriangularMesh2D::Element> TriangularMesh2D::ElementIndex::getElement(Vec<2, double> p) const {
    for (auto v: rtree | boost::geometry::index::adaptors::queried(boost::geometry::index::intersects(p))) {
        const Element el = mesh.getElement(v.second);
        if (el.contains(p)) return el;
    }
    return optional<TriangularMesh2D::Element>();
}

TriangularMesh2D TriangularMesh2D::masked(const TriangularMesh2D::Predicate &predicate) const {
    TriangularMesh2D result;
    Builder builder(result, elementNodes.size(), nodes.size());
    for (auto el: elements())
        if (predicate(el)) builder.add(el);
    return result;
}

void TriangularMesh2D::writeXML(XMLElement &object) const {
    object.attr("type", "triangular2d");
    for (auto & node: nodes)
        object.addTag("node").attr("tran", node.tran()).attr("vert", node.vert());
    for (auto & el: elementNodes)
        object.addTag("element").attr("a", el[0]).attr("b", el[1]).attr("c", el[2]);
}

inline TriangularMesh2D::Segment makeSegment(std::size_t a, std::size_t b) {
    return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
}

void TriangularMesh2D::countSegmentsOf(TriangularMesh2D::SegmentsCounts& counter, const TriangularMesh2D::Element& el) {
    ++counter[makeSegment(el.getNodeIndex(0), el.getNodeIndex(1))];
    ++counter[makeSegment(el.getNodeIndex(1), el.getNodeIndex(2))];
    ++counter[makeSegment(el.getNodeIndex(2), el.getNodeIndex(0))];
}

TriangularMesh2D::SegmentsCounts TriangularMesh2D::countSegments() const {
    TriangularMesh2D::SegmentsCounts result;
    for (const auto el: elements()) countSegmentsOf(result, el);
    return result;
}

TriangularMesh2D::SegmentsCounts TriangularMesh2D::countSegmentsIn(const Box2D &box) const {
    TriangularMesh2D::SegmentsCounts result;
    for (const auto el: elements())
        if (box.contains(el.getNode(0)) && box.contains(el.getNode(1)) && box.contains(el.getNode(2)))
            countSegmentsOf(result, el);
    return result;
}

TriangularMesh2D::SegmentsCounts TriangularMesh2D::countSegmentsIn(const std::vector<Box2D>& boxes) const {
    TriangularMesh2D::SegmentsCounts result;
    for (const auto el: elements()) {
        bool vertex0_included = false, vertex1_included = false, vertex2_included = false;
        for (auto& box: boxes) {
            if (!vertex0_included) vertex0_included = box.contains(el.getNode(0));
            if (!vertex1_included) vertex1_included = box.contains(el.getNode(1));
            if (!vertex2_included) vertex2_included = box.contains(el.getNode(2));
            if (vertex0_included && vertex1_included && vertex2_included) {
                countSegmentsOf(result, el);
                break;
            }
        }
    }
    return result;
}

TriangularMesh2D::SegmentsCounts TriangularMesh2D::countSegmentsIn(const GeometryD<2> &geometry, const GeometryObject &object, const PathHints *path) const {
    TriangularMesh2D::SegmentsCounts result;
    for (const auto el: elements())
        if (geometry.objectIncludes(object, path, el.getNode(0)) &&
            geometry.objectIncludes(object, path, el.getNode(1)) &&
            geometry.objectIncludes(object, path, el.getNode(2)))
            countSegmentsOf(result, el);
    return result;
}

std::set<std::size_t> TriangularMesh2D::allBoundaryNodes(const TriangularMesh2D::SegmentsCounts& segmentsCount) {
    std::set<std::size_t> result;
    for (const std::pair<TriangularMesh2D::Segment, std::size_t>& s: segmentsCount)
        if (s.second == 1) {
            result.insert(s.first.first);
            result.insert(s.first.second);
        }
    return result;
}

template <int DIR, template<class> class Compare = std::less>  // DIR - oś prostopadła do przedziałów trzymanych w boost::icl::interval_map
struct SegmentSetMember: private Compare<TriangularMesh2D::Segment> {

    constexpr static int SEG_DIR = 1 - DIR; // kierunek rozciągania się przedziałów w interval_map

    TriangularMesh2D::Segment segment;
    double min, max;    // mniejsza i większa współrzędna (w osi DIR) końców segmentu

    SegmentSetMember() = default;

    SegmentSetMember(const TriangularMesh2D& mesh, const TriangularMesh2D::Segment& segment)
        : segment(segment),
          min(mesh.nodes[segment.first][DIR]),
          max(mesh.nodes[segment.second][DIR])
    {
        //if (max < min) std::swap(min, max);
        if (Compare<double>()(max, min))  // max < min ?
            std::swap(min, max);
    }

    bool operator<(const SegmentSetMember& other) const {
        return Compare<TriangularMesh2D::Segment>::operator()(this->segment, other.segment);
                //this->segment < other.segment;
    }

    bool operator==(const SegmentSetMember& other) const {
        return this->segment == other.segment;
    }

    /**
     * Check if this segment dominate given point.
     * @param mesh
     * @param point
     * @return whether this segment dominate given point
     */
    bool dominates(const TriangularMesh2D& mesh, const Vec<2, double>& point) const {
        // this two cases are supported also by interpolation, but they are common and we want to ensure to check them exact:
        const Vec<2, double>& f = mesh.nodes[segment.first];
        if (point[SEG_DIR] == f[SEG_DIR])
            return Compare<double>()(point[DIR], f[DIR]); //point[DIR] < f[DIR];
        const Vec<2, double>& s = mesh.nodes[segment.second];
        if (point[SEG_DIR] == s[SEG_DIR])
            return Compare<double>()(point[DIR], s[DIR]); //point[DIR] < f[DIR];

        if (f[SEG_DIR] < s[SEG_DIR])
            return Compare<double>()(point[DIR], interpolation::linear(f[SEG_DIR], f[DIR], s[SEG_DIR], s[DIR], point[SEG_DIR]));
            //point[DIR] < interpolation::linear(f[SEG_DIR], f[DIR], s[SEG_DIR], s[DIR], point[SEG_DIR]);
        else    // same as above, but f<->s are swapped
            return Compare<double>()(point[DIR], interpolation::linear(s[SEG_DIR], s[DIR], f[SEG_DIR], f[DIR], point[SEG_DIR]));
    }
};

// DIR - oś prostopadła do przedziałów trzymanych w boost::icl::interval_map
template<int DIR, template<class> class Compare = std::less>
struct SegmentSet: Compare<double> {
    SegmentSet() = default;

    SegmentSet(const SegmentSetMember<DIR, Compare>& to_append)
        : maxOfMins(to_append.min) { set.insert(to_append); }

    SegmentSet(const TriangularMesh2D& mesh, const TriangularMesh2D::Segment& segment)
        : SegmentSet(SegmentSetMember<DIR, Compare>(mesh, segment)) {}

    SegmentSet& operator+=(const SegmentSet& right) {
        //if (maxOfMins < right.maxOfMins) {
        if (Compare<double>::operator()(maxOfMins, right.maxOfMins)) {
            SetT this_set_backup = std::move(set);
            set = right.set;
            maxOfMins = right.maxOfMins;
            insert(this_set_backup);
        } else
            insert(right.set);
        return *this;
    }

    bool operator==(const SegmentSet& other) const {
        return this->set == other.set;
    }

    // check if point is dominated by any segment in set represented by this
    bool dominates(const TriangularMesh2D& mesh, const Vec<2, double>& point) const {
        for (const SegmentSetMember<DIR, Compare>& s: set)
            if (s.dominates(mesh, point)) return true;
        return false;
    }

private:
    typedef std::set<SegmentSetMember<DIR, Compare>, Compare<SegmentSetMember<DIR, Compare>>> SetT;

    void insert(const SetT& right) {
        for (auto& s: right)
            if (!Compare<double>::operator()(s.max, maxOfMins))
                set.insert(s);
            //if (s.max >= maxOfMins) set.insert(s);
    }

    SetT set;

    /**
     * Maximum of s.min, where s iterates over set.
     *
     * In set we store only non-dominated segments, which have max >= maxOfMins.
     */
    double maxOfMins;
};

template<int SEG_DIR, template<class> class Compare>
std::set<std::size_t> TriangularMesh2D::dirBoundaryNodes(const TriangularMesh2D::SegmentsCounts &segmentsCount) const {
    typedef SegmentSet<1-SEG_DIR, Compare> SegmentSetT;
    std::set<std::size_t> result;
    boost::icl::interval_map<double, SegmentSetT> non_dominated;
    for (const std::pair<TriangularMesh2D::Segment, std::size_t>& s: segmentsCount)
        if (s.second == 1) {
            const TriangularMesh2D::Segment& seg = s.first;
            non_dominated += std::make_pair(
                boost::icl::interval<double>::closed(this->nodes[seg.first][SEG_DIR], this->nodes[seg.second][SEG_DIR]),
                SegmentSetT(*this, seg)
            );
            result.insert(s.first.first);
            result.insert(s.first.second);
        }
    // remove nodes which are dominated:
    for (auto it = result.begin(); it != result.end(); ) {
        const auto& node_coords = this->nodes[*it];
        if (non_dominated.find(node_coords[SEG_DIR])->second.dominates(*this, node_coords))
            it = result.erase(it);
        else
            ++it;
    }
    return result;
}

TriangularMesh2D::Boundary TriangularMesh2D::getRightBoundary() {
    return Boundary( [](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>&) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::RIGHT>(mesh.countSegments())));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getTopBoundary() {
    return Boundary( [](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>&) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::TOP>(mesh.countSegments())));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getLeftBoundary() {
    return Boundary( [](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>&) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::LEFT>(mesh.countSegments())));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getBottomBoundary() {
    return Boundary( [](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>&) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::BOTTOM>(mesh.countSegments())));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getRightOfBoundary(const Box2D &box) {
    return Boundary( [box](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>&) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::RIGHT>(mesh.countSegmentsIn(box))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getTopOfBoundary(const Box2D &box) {
    return Boundary( [box](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>&) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::TOP>(mesh.countSegmentsIn(box))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getLeftOfBoundary(const Box2D &box) {
    return Boundary( [box](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>&) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::LEFT>(mesh.countSegmentsIn(box))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getBottomOfBoundary(const Box2D &box) {
    return Boundary( [box](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>&) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::BOTTOM>(mesh.countSegmentsIn(box))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getRightOfBoundary(const std::vector<Box2D> &boxes) {
    return Boundary( [boxes](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>&) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::RIGHT>(mesh.countSegmentsIn(boxes))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getTopOfBoundary(const std::vector<Box2D> &boxes) {
    return Boundary( [boxes](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>&) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::TOP>(mesh.countSegmentsIn(boxes))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getLeftOfBoundary(const std::vector<Box2D> &boxes) {
    return Boundary( [boxes](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>&) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::LEFT>(mesh.countSegmentsIn(boxes))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getBottomOfBoundary(const std::vector<Box2D> &boxes) {
    return Boundary( [boxes](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>&) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::BOTTOM>(mesh.countSegmentsIn(boxes))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getRightOfBoundary(shared_ptr<const GeometryObject> object) {
    return Boundary( [=](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>& geom) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::RIGHT>(mesh.countSegmentsIn(*geom, *object))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getRightOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path) {
    return Boundary( [=](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>& geom) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::RIGHT>(mesh.countSegmentsIn(*geom, *object, &path))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getTopOfBoundary(shared_ptr<const GeometryObject> object) {
    return Boundary( [=](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>& geom) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::TOP>(mesh.countSegmentsIn(*geom, *object))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getTopOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path) {
    return Boundary( [=](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>& geom) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::TOP>(mesh.countSegmentsIn(*geom, *object, &path))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getLeftOfBoundary(shared_ptr<const GeometryObject> object) {
    return Boundary( [=](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>& geom) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::LEFT>(mesh.countSegmentsIn(*geom, *object))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getLeftOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path) {
    return Boundary( [=](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>& geom) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::LEFT>(mesh.countSegmentsIn(*geom, *object, &path))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getBottomOfBoundary(shared_ptr<const GeometryObject> object) {
    return Boundary( [=](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>& geom) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::BOTTOM>(mesh.countSegmentsIn(*geom, *object))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getBottomOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path) {
    return Boundary( [=](const TriangularMesh2D& mesh, const shared_ptr<const GeometryD<2>>& geom) {
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<BoundaryDir::BOTTOM>(mesh.countSegmentsIn(*geom, *object, &path))));
    } );
}

TriangularMesh2D::Boundary TriangularMesh2D::getBoundary(const std::string &boundary_desc) {
    if (boundary_desc == "bottom") return getBottomBoundary();
    if (boundary_desc == "left") return getLeftBoundary();
    if (boundary_desc == "right") return getRightBoundary();
    if (boundary_desc == "top") return getTopBoundary();
    if (boundary_desc == "all") return getAllBoundary();
    return Boundary();
}

TriangularMesh2D::Boundary TriangularMesh2D::getBoundary(XMLReader &boundary_desc, Manager &manager) {
    auto side = boundary_desc.getAttribute("side");
    if (side) {
        if (*side == "bottom")
            return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getBottomBoundary, &getBottomOfBoundary);
        if (*side == "left")
            return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getLeftBoundary, &getLeftOfBoundary);
        if (*side == "right")
            return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getRightBoundary, &getRightOfBoundary);
        if (*side == "top")
            return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getTopBoundary, &getTopOfBoundary);
        if (*side == "all")
            return details::parseBoundaryFromXML<Boundary, 2>(boundary_desc, manager, &getAllBoundary, &getAllBoundaryIn);
        throw XMLBadAttrException(boundary_desc, "side", *side);
    }
    return Boundary();
}

std::size_t readTriangularMesh2D_readNodeIndex(XMLReader& reader, const char* attrName, std::size_t nodes_size) {
    std::size_t result = reader.requireAttribute<std::size_t>(attrName);
    if (result >= nodes_size)
        reader.throwException(format("{} in <element> equals {} and is out of range [0, {})", attrName, result, nodes_size));
    return result;
}

TriangularMesh2D TriangularMesh2D::read(XMLReader& reader) {
    TriangularMesh2D result;

    if (reader.requireTagOrEnd()) { // has tag?
        std::string tag_name = reader.getNodeName();
        if (tag_name == "triangle") {   // sequence of triangles
            TriangularMesh2D::Builder builder(result);
            do {
                builder.add(
                    vec(reader.requireAttribute<double>("a0"), reader.requireAttribute<double>("a1")),
                    vec(reader.requireAttribute<double>("b0"), reader.requireAttribute<double>("b1")),
                    vec(reader.requireAttribute<double>("c0"), reader.requireAttribute<double>("c1"))
                );
                reader.requireTagEnd();
            } while (reader.requireTagOrEnd("triangle"));
        } else if (tag_name == "node") {
            result.nodes.emplace_back(reader.requireAttribute<double>("tran"), reader.requireAttribute<double>("vert"));
            reader.requireTagEnd();
            bool accept_nodes = true;   // accept <node> and <element> tags if true, else accept only <element>
            while (reader.requireTagOrEnd()) {
                std::string tag_name = reader.getNodeName();
                if (accept_nodes && tag_name == "node") {
                    result.nodes.emplace_back(reader.requireAttribute<double>("tran"), reader.requireAttribute<double>("vert"));
                    reader.requireTagEnd();
                } else if (tag_name == "element") {
                    result.elementNodes.push_back({
                        readTriangularMesh2D_readNodeIndex(reader, "a", result.nodes.size()),
                        readTriangularMesh2D_readNodeIndex(reader, "b", result.nodes.size()),
                        readTriangularMesh2D_readNodeIndex(reader, "c", result.nodes.size())
                    });
                    reader.requireTagEnd();
                    accept_nodes = false;
                }
            }
        } else
            reader.throwUnexpectedElementException("expected <triangle> or <node> tag, got <" + tag_name + ">");
    }

    return result;
}

static shared_ptr<Mesh> readTriangularMesh2D(XMLReader& reader) {
    return plask::make_shared<TriangularMesh2D>(TriangularMesh2D::read(reader));
}

static RegisterMeshReader rectangular2d_reader("triangular2d", readTriangularMesh2D);

template PLASK_API std::set<std::size_t> TriangularMesh2D::boundaryNodes<TriangularMesh2D::BoundaryDir::TOP>(const TriangularMesh2D::SegmentsCounts& segmentsCount) const;
template PLASK_API std::set<std::size_t> TriangularMesh2D::boundaryNodes<TriangularMesh2D::BoundaryDir::LEFT>(const TriangularMesh2D::SegmentsCounts& segmentsCount) const;
template PLASK_API std::set<std::size_t> TriangularMesh2D::boundaryNodes<TriangularMesh2D::BoundaryDir::RIGHT>(const TriangularMesh2D::SegmentsCounts& segmentsCount) const;
template PLASK_API std::set<std::size_t> TriangularMesh2D::boundaryNodes<TriangularMesh2D::BoundaryDir::BOTTOM>(const TriangularMesh2D::SegmentsCounts& segmentsCount) const;
template PLASK_API std::set<std::size_t> TriangularMesh2D::boundaryNodes<TriangularMesh2D::BoundaryDir::ALL>(const TriangularMesh2D::SegmentsCounts& segmentsCount) const;

// ------------------ Nearest Neighbor interpolation ---------------------

template<typename DstT, typename SrcT>
NearestNeighborTriangularMesh2DLazyDataImpl<DstT, SrcT>::NearestNeighborTriangularMesh2DLazyDataImpl(
        const shared_ptr<const TriangularMesh2D>& src_mesh,
        const DataVector<const SrcT>& src_vec,
        const shared_ptr<const MeshD<2>>& dst_mesh,
        const InterpolationFlags& flags)
    : InterpolatedLazyDataImpl<DstT, TriangularMesh2D, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
      nodesIndex(boost::irange(std::size_t(0), src_mesh->size()),
                 typename RtreeOfTriangularMesh2DNodes::parameters_type(),
                 TriangularMesh2DGetterForRtree(src_mesh.get()))
{
}

template <typename DstT, typename SrcT>
DstT NearestNeighborTriangularMesh2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    auto point = this->dst_mesh->at(index);
    auto wrapped_point = this->flags.wrap(point);
    for (auto v: nodesIndex | boost::geometry::index::adaptors::queried(boost::geometry::index::nearest(wrapped_point, 1)))
        return this->flags.postprocess(point, this->src_vec[v]);
    return DstT();
}

template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<double, double>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<dcomplex, dcomplex>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;


// ------------------ Barycentric / Linear interpolation ---------------------

template<typename DstT, typename SrcT>
BarycentricTriangularMesh2DLazyDataImpl<DstT, SrcT>::BarycentricTriangularMesh2DLazyDataImpl(const shared_ptr<const TriangularMesh2D> &src_mesh, const DataVector<const SrcT> &src_vec, const shared_ptr<const MeshD<2> > &dst_mesh, const InterpolationFlags &flags)
    : InterpolatedLazyDataImpl<DstT, TriangularMesh2D, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
      elementIndex(*src_mesh)
{
}

template <typename DstT, typename SrcT>
DstT BarycentricTriangularMesh2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    auto point = this->dst_mesh->at(index);
    auto wrapped_point = this->flags.wrap(point);
    for (auto v: elementIndex.rtree | boost::geometry::index::adaptors::queried(boost::geometry::index::intersects(wrapped_point))) {
        const auto el = this->src_mesh->getElement(v.second);
        const auto b = el.barycentric(wrapped_point);
        if (b.c0 < 0.0 || b.c1 < 0.0 || b.c2 < 0.0) continue; // wrapped_point is outside of the triangle
        return this->flags.postprocess(point,
                                       b.c0*this->src_vec[el.getNodeIndex(0)] +
                                       b.c1*this->src_vec[el.getNodeIndex(1)] +
                                       b.c2*this->src_vec[el.getNodeIndex(2)]);
    }
    return NaN<decltype(this->src_vec[0])>();
}

template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<double, double>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<dcomplex, dcomplex>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;


// ------------------ Element mesh Nearest Neighbor interpolation ---------------------

template<typename DstT, typename SrcT>
NearestNeighborElementTriangularMesh2DLazyDataImpl<DstT, SrcT>::NearestNeighborElementTriangularMesh2DLazyDataImpl(
        const shared_ptr<const TriangularMesh2D::ElementMesh>& src_mesh,
        const DataVector<const SrcT>& src_vec,
        const shared_ptr<const MeshD<2>>& dst_mesh,
        const InterpolationFlags& flags)
    : InterpolatedLazyDataImpl<DstT, TriangularMesh2D::ElementMesh, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
      elementIndex(src_mesh->getOriginalMesh())
{
}

template<typename DstT, typename SrcT>
DstT NearestNeighborElementTriangularMesh2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const {
    auto point = this->dst_mesh->at(index);
    auto wrapped_point = this->flags.wrap(point);
    std::size_t element_index = elementIndex.getIndex(wrapped_point);
    if (element_index == TriangularMesh2D::ElementIndex::INDEX_NOT_FOUND)
        return NaN<decltype(this->src_vec[0])>();
    return this->flags.postprocess(point, this->src_vec[element_index]);
}

template struct PLASK_API NearestNeighborElementTriangularMesh2DLazyDataImpl<double, double>;
template struct PLASK_API NearestNeighborElementTriangularMesh2DLazyDataImpl<dcomplex, dcomplex>;
template struct PLASK_API NearestNeighborElementTriangularMesh2DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API NearestNeighborElementTriangularMesh2DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;
template struct PLASK_API NearestNeighborElementTriangularMesh2DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API NearestNeighborElementTriangularMesh2DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;
template struct PLASK_API NearestNeighborElementTriangularMesh2DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API NearestNeighborElementTriangularMesh2DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;
template struct PLASK_API NearestNeighborElementTriangularMesh2DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API NearestNeighborElementTriangularMesh2DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;

bool TriangularMesh2D::ElementMesh::hasSameNodes(const MeshD<2> &to_compare) const {
    if (const ElementMesh* c = dynamic_cast<const ElementMesh*>(&to_compare))
        if (this->originalMesh == c->originalMesh) return true;
    return MeshD<2>::hasSameNodes(to_compare);
}

}   // namespace plask
