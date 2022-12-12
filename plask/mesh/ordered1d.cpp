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
#include "ordered1d.hpp"

#include "../utils/stl.hpp"
#include "../log/log.hpp"

namespace plask {

void OrderedAxis::sortPointsAndRemoveNonUnique(double min_dist)
{
    std::sort(this->points.begin(), this->points.end());
    auto almost_equal = [min_dist](const double& x, const double& y) -> bool { return std::abs(x-y) < min_dist; };
    this->points.erase(std::unique(this->points.begin(), this->points.end(), almost_equal), this->points.end());
}

OrderedAxis::OrderedAxis(std::initializer_list<PointType> points, double min_dist): points(points), warn_too_close(true) {
    sortPointsAndRemoveNonUnique(min_dist);
}

OrderedAxis::OrderedAxis(const std::vector<PointType>& points, double min_dist): points(points), warn_too_close(true) {
    sortPointsAndRemoveNonUnique(min_dist);
}

OrderedAxis::OrderedAxis(std::vector<PointType>&& points, double min_dist): points(std::move(points)), warn_too_close(true) {
    sortPointsAndRemoveNonUnique(min_dist);
}

bool OrderedAxis::operator==(const plask::OrderedAxis& to_compare) const {
    return points == to_compare.points;
}

void OrderedAxis::writeXML(XMLElement &object) const {
    object.attr("type", "ordered");
    //object.indent();
    for (auto x: this->points) {
        object.writeText(x);
        object.writeText(" ");
    }
    //object.writeText("\n");
}



OrderedAxis::native_const_iterator OrderedAxis::find(double to_find) const {
    return std::lower_bound(points.begin(), points.end(), to_find);
}

OrderedAxis::native_const_iterator OrderedAxis::findUp(double to_find) const {
    return std::upper_bound(points.begin(), points.end(), to_find);
}

OrderedAxis::native_const_iterator OrderedAxis::findNearest(double to_find) const {
    return find_nearest_binary(points.begin(), points.end(), to_find);
}

OrderedAxis &OrderedAxis::operator=(const MeshAxis &src) {
    bool resized = size() != src.size();
    points.clear();
    points.reserve(src.size());
    for (auto i: src) points.push_back(i);
    std::sort(points.begin(), points.end());
    if (resized) fireResized(); else fireChanged();
    return *this;
}

OrderedAxis &OrderedAxis::operator=(OrderedAxis &&src) {
    bool resized = size() != src.size();
    this->points = std::move(src.points);
    if (resized) fireResized(); else fireChanged();
    return *this;
}

OrderedAxis &OrderedAxis::operator=(const OrderedAxis &src) {
    bool resized = size() != src.size();
    points = src.points;
    if (resized) fireResized(); else fireChanged();
    return *this;
}

bool OrderedAxis::addPoint(double new_node_cord, double min_dist) {
    auto where = std::lower_bound(points.begin(), points.end(), new_node_cord);
    if (where == points.end()) {
        if (points.size() == 0 || new_node_cord - points.back() > min_dist) {
            points.push_back(new_node_cord);
            fireResized();
            return true;
        }
    } else {
        if (*where - new_node_cord > min_dist && (where == points.begin() || new_node_cord - *(where-1) > min_dist)) {
            points.insert(where, new_node_cord);
            fireResized();
            return true;
        }
    }
    if (warn_too_close) writelog(LOG_WARNING, "Points in ordered mesh too close, skipping point at {0}", new_node_cord);
    return false;
}

void OrderedAxis::addPointsLinear(double first, double last, std::size_t points_count) {
    if (points_count == 0) return;
    --points_count;
    const double len = last - first;
    auto get_el = [&](std::size_t i) { return first + double(i) * len / double(points_count); };
    addOrderedPoints(makeFunctorIndexedIterator(get_el, 0), makeFunctorIndexedIterator(get_el, points_count+1), points_count+1);
    fireResized();
}

void OrderedAxis::removePoint(std::size_t index) {
    points.erase(points.begin() + index);
    fireResized();
}

void OrderedAxis::removePoints(std::size_t start, std::size_t stop) {
    points.erase(points.begin() + start, points.begin() + stop);
    fireResized();
}

void OrderedAxis::removePoints(std::size_t start, std::size_t stop, std::ptrdiff_t step) {
    if (step > 0) {
        if (stop < start) return;
        if (step == 1)
            points.erase(points.begin() + start, points.begin() + stop);
        else
            for (std::size_t i = start; i < stop; i += step) {
                points.erase(points.begin() + (i--));
                stop--;
            }
    } else
        if (stop > start) return;
        if (step == -1)
            points.erase(points.begin() + stop, points.begin() + start);
        else if (step == 0)
            throw Exception("OrderedAxis: step cannot be zero");
        else
            for (std::size_t i = start; i > stop; i += step)
                points.erase(points.begin() + i);
    fireResized();
}


void OrderedAxis::clear() {
    points.clear();
    fireResized();
}

shared_ptr<MeshAxis> OrderedAxis::clone() const {
    return plask::make_shared<OrderedAxis>(*this);
}

shared_ptr<OrderedMesh1D> readRectilinearMeshAxis(XMLReader& reader) {
    auto result = plask::make_shared<OrderedMesh1D>();
    if (reader.hasAttribute("start")) {
         double start = reader.requireAttribute<double>("start");
         double stop = reader.requireAttribute<double>("stop");
         size_t count = reader.requireAttribute<size_t>("num");
         result->addPointsLinear(start, stop, count);
         reader.requireTagEnd();
    } else {
         std::string data = reader.requireTextInCurrentTag();
         for (auto point: boost::tokenizer<boost::char_separator<char>>(data, boost::char_separator<char>(" ,;\t\n"))) {
             try {
                 double val = boost::lexical_cast<double>(point);
                 result->addPoint(val);
             } catch (boost::bad_lexical_cast&) {
                 throw XMLException(reader, format("Value '{0}' cannot be converted to float", point));
             }
         }
    }
    return result;
}

shared_ptr<OrderedMesh1D> readOrderedMesh1D(XMLReader& reader) {
    reader.requireTag("axis");
    auto result = readRectilinearMeshAxis(reader);
    reader.requireTagEnd();
    return result;
}

static RegisterMeshReader rectilinearmesh_reader("ordered", readOrderedMesh1D);


// obsolete:

shared_ptr<OrderedMesh1D> readOrderedMesh1D_obsolete(XMLReader& reader) {
    writelog(LOG_WARNING, "Mesh type \"{0}\" is obsolete, use \"ordered\" instead.", reader.requireAttribute("type"));
    return readOrderedMesh1D(reader);
}
RegisterMeshReader rectilinearmesh1d_reader("rectilinear1d", readOrderedMesh1D_obsolete);




}   // namespace plask
