#include <plask/log/log.h>
#include <plask/manager.h>

#include "generator_rectangular.h"

namespace plask {

inline static void addPoints(OrderedAxis& dst, double lo, double up, bool singleMaterial, double min_ply, unsigned max_points) {
    dst.addPoint(lo);
    dst.addPoint(up);
    if (!singleMaterial) {
        const double ply = abs(up - lo);
        const unsigned points = (min_ply != 0.)? std::min(unsigned(std::ceil(ply / abs(min_ply))), max_points) : max_points;
        for (long i = long(points) - 1; i > 0; --i) {
            dst.addPoint(lo + i * ply / points, 0.5*ply/points);
        }
    }
}

shared_ptr<OrderedAxis> makeGeometryGrid1D(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    auto mesh = plask::make_shared<OrderedAxis>();
    OrderedAxis::WarningOff warning_off(mesh);

    std::vector<Box2D> boxes = geometry->getLeafsBoundingBoxes();
    std::vector<shared_ptr<const GeometryObject>> leafs = geometry->getLeafs();

    for (std::size_t i = 0; i < boxes.size(); ++i)
        if (boxes[i].isValid()) {
            addPoints(*mesh, boxes[i].lower.c0, boxes[i].upper.c0, leafs[i]->isUniform(Primitive<3>::DIRECTION_TRAN), leafs[i]->min_ply, leafs[i]->max_points);
        }

    return mesh;
}

shared_ptr<MeshD<1>> OrderedMesh1DSimpleGenerator::generate(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    auto mesh = makeGeometryGrid1D(geometry);
    writelog(LOG_DETAIL, "mesh.Rectangular1D.SimpleGenerator: Generating new mesh ({0})", mesh->size());
    return mesh;
}


shared_ptr<RectangularMesh<2>> makeGeometryGrid(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    shared_ptr<OrderedAxis> axis0(new OrderedAxis), axis1(new OrderedAxis);
    OrderedAxis::WarningOff warning_off0(axis0);
    OrderedAxis::WarningOff warning_off1(axis1);

    std::vector<Box2D> boxes = geometry->getLeafsBoundingBoxes();
    std::vector<shared_ptr<const GeometryObject>> leafs = geometry->getLeafs();

    for (std::size_t i = 0; i < boxes.size(); ++i)
        if (boxes[i].isValid()) {
            addPoints(*axis0, boxes[i].lower.c0, boxes[i].upper.c0, leafs[i]->isUniform(Primitive<3>::DIRECTION_TRAN), leafs[i]->min_ply, leafs[i]->max_points);
            addPoints(*axis1, boxes[i].lower.c1, boxes[i].upper.c1, leafs[i]->isUniform(Primitive<3>::DIRECTION_VERT), leafs[i]->min_ply, leafs[i]->max_points);
        }

    shared_ptr<RectangularMesh<2>> mesh = plask::make_shared<RectangularMesh<2>>(std::move(axis0), std::move(axis1));
    mesh->setOptimalIterationOrder();
    return mesh;
}

shared_ptr<MeshD<2>> RectangularMesh2DSimpleGenerator::generate(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    shared_ptr<RectangularMesh<2>> mesh = makeGeometryGrid(geometry);
    writelog(LOG_DETAIL, "mesh.Rectangular2D.SimpleGenerator: Generating new mesh ({0}x{1})", mesh->axis0->size(), mesh->axis1->size());
    return mesh;
}


shared_ptr<MeshD<2>> RectangularMesh2DFrom1DGenerator::generate(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    return plask::make_shared<RectangularMesh<2>>(horizontal_generator->get<MeshAxis>(geometry), makeGeometryGrid(geometry)->axis1);
}


shared_ptr<RectangularMesh<3>> makeGeometryGrid(const shared_ptr<GeometryObjectD<3>>& geometry)
{
    shared_ptr<OrderedAxis> axis0(new OrderedAxis), axis1(new OrderedAxis), axis2(new OrderedAxis);
    OrderedAxis::WarningOff warning_off0(axis0);
    OrderedAxis::WarningOff warning_off1(axis1);
    OrderedAxis::WarningOff warning_off2(axis2);

    std::vector<Box3D> boxes = geometry->getLeafsBoundingBoxes();
    std::vector<shared_ptr<const GeometryObject>> leafs = geometry->getLeafs();

    for (std::size_t i = 0; i < boxes.size(); ++i)
        if (boxes[i].isValid()) {
            addPoints(*axis0, boxes[i].lower.c0, boxes[i].upper.c0, leafs[i]->isUniform(Primitive<3>::DIRECTION_LONG), leafs[i]->min_ply, leafs[i]->max_points);
            addPoints(*axis1, boxes[i].lower.c1, boxes[i].upper.c1, leafs[i]->isUniform(Primitive<3>::DIRECTION_TRAN), leafs[i]->min_ply, leafs[i]->max_points);
            addPoints(*axis2, boxes[i].lower.c2, boxes[i].upper.c2, leafs[i]->isUniform(Primitive<3>::DIRECTION_VERT), leafs[i]->min_ply, leafs[i]->max_points);
        }

    shared_ptr<RectangularMesh<3>> mesh = plask::make_shared<RectangularMesh<3>>(std::move(axis0), std::move(axis1), std::move(axis2));
    mesh->setOptimalIterationOrder();
    return mesh;
}

shared_ptr<MeshD<3>> RectangularMesh3DSimpleGenerator::generate(const shared_ptr<GeometryObjectD<3>>& geometry)
{
    auto mesh = makeGeometryGrid(geometry);
    writelog(LOG_DETAIL, "mesh.Rectangular3D.SimpleGenerator: Generating new mesh ({0}x{1}x{2})", mesh->axis0->size(), mesh->axis1->size(), mesh->axis2->size());
    return mesh;
}



shared_ptr<OrderedAxis> refineAxis(const shared_ptr<MeshAxis>& axis, double spacing) {
    if (spacing == 0. || isinf(spacing) || isnan(spacing)) return make_shared<OrderedAxis>(*axis);
    size_t total = 1;
    for (size_t i = 1; i < axis->size(); ++i) {
        total += size_t(max(round((axis->at(i) - axis->at(i-1)) / spacing), 1.));
    }
    std::vector<double> points;
    points.reserve(total);
    for (size_t i = 1; i < axis->size(); ++i) {
        double offset = axis->at(i-1);
        double range = axis->at(i) - offset;
        double steps = max(round(range / spacing), 1.);
        double step = range / steps;
        for (size_t j = 0, n = size_t(steps); j < n; ++j) {
            points.push_back(offset + j * step);
        }
    }
    points.push_back(axis->at(axis->size()-1));
    assert(points.size() == total);
    return shared_ptr<OrderedAxis>(new OrderedAxis(std::move(points)));
}

shared_ptr<MeshD<1>> OrderedMesh1DRegularGenerator::generate(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    auto mesh = refineAxis(makeGeometryGrid1D(geometry), spacing);
    writelog(LOG_DETAIL, "mesh.Rectangular1D.RegularGenerator: Generating new mesh ({0})", mesh->size());
    return mesh;
}

shared_ptr<MeshD<2>> RectangularMesh2DRegularGenerator::generate(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    auto mesh1 = makeGeometryGrid(geometry);
    auto mesh = make_shared<RectangularMesh<2>>(refineAxis(mesh1->axis0, spacing0), refineAxis(mesh1->axis1, spacing1));
    writelog(LOG_DETAIL, "mesh.Rectangular2D.RegularGenerator: Generating new mesh ({0}x{1})", mesh->axis0->size(), mesh->axis1->size());
    return mesh;
}

shared_ptr<MeshD<3>> RectangularMesh3DRegularGenerator::generate(const shared_ptr<GeometryObjectD<3>>& geometry)
{
    auto mesh1 = makeGeometryGrid(geometry);
    auto mesh = make_shared<RectangularMesh<3>>(refineAxis(mesh1->axis0, spacing0), refineAxis(mesh1->axis1, spacing1), refineAxis(mesh1->axis2, spacing2));
    writelog(LOG_DETAIL, "mesh.Rectangular3D.RegularGenerator: Generating new mesh ({0}x{1}x{2})", mesh->axis0->size(), mesh->axis1->size(), mesh->axis2->size());
    return mesh;
}


template <int dim>
std::pair<double, double> RectangularMeshRefinedGenerator<dim>::getMinMax(const shared_ptr<OrderedAxis> &axis)
{
    double min = INFINITY, max = 0;
    for (size_t i = 1; i != axis->size(); ++i) {
        double L = axis->at(i) - axis->at(i-1);
        if (L < min) min = L;
        if (L > max) max = L;
    }
    return std::pair<double, double>(min, max);
}

template <int dim>
void RectangularMeshRefinedGenerator<dim>::divideLargestSegment(shared_ptr<OrderedAxis> axis)
{
    double max = 0;
    double newpoint;
    for (size_t i = 1; i != axis->size(); ++i) {
        double L = axis->at(i) - axis->at(i-1);
        if (L > max) { max = L; newpoint = 0.5 * (axis->at(i-1) + axis->at(i)); }
    }
    OrderedAxis::WarningOff warning_off(axis);
    axis->addPoint(newpoint);
}

template <int dim>
shared_ptr<OrderedAxis> RectangularMeshRefinedGenerator<dim>::getAxis(shared_ptr<OrderedAxis> axis, const shared_ptr<GeometryObjectD<DIM>>& geometry, size_t dir)
{
    assert(bool(axis));
    OrderedAxis::WarningOff warning_off(axis);

    // Add refinement points
    for (auto ref: this->refinements[dir]) {
        auto object = ref.first.first.lock();
        if (!object) {
             if (this->warn_missing) writelog(LOG_WARNING, "{}: Refinement defined for object not existing any more", name());
        } else {
            auto path = ref.first.second;
            auto boxes = geometry->getObjectBoundingBoxes(*object, path);
            auto origins = geometry->getObjectPositions(*object, path);
            if (this->warn_missing && boxes.size() == 0) writelog(LOG_WARNING, "DivideGenerator: Refinement defined for object absent from the geometry");
            else if (this->warn_multiple && boxes.size() > 1) writelog(LOG_WARNING, "DivideGenerator: Single refinement defined for more than one object");
            auto box = boxes.begin();
            auto origin = origins.begin();
            for (; box != boxes.end(); ++box, ++origin) {
                for (auto x: ref.second) {
                    double zero = (*origin)[dir];
                    double lower = box->lower[dir] - zero;
                    double upper = box->upper[dir] - zero;
                    if (this->warn_outside && (x < lower || x > upper))
                        writelog(LOG_WARNING, "{4}: Refinement at specified at {0} lying at {1} in global coords. is outside of the object ({2} to {3})",
                                            x, x+zero, lower+zero, upper+zero, name());
                    axis->addPoint(zero + x);
                }
            }
        }
    }

    // Have specialization make further axis processing
    return processAxis(axis, geometry, dir);
}

template <> shared_ptr<MeshD<1>>
RectangularMeshRefinedGenerator<1>::generate(const boost::shared_ptr<plask::GeometryObjectD<2>>& geometry)
{
    shared_ptr<OrderedAxis> mesh = makeGeometryGrid1D(geometry);
    getAxis(mesh, geometry, 0);
    writelog(LOG_DETAIL, "mesh.Rectilinear1D::{}: Generating new mesh ({:d})", name(), mesh->size());
    return mesh;
}

template <> shared_ptr<MeshD<2>>
RectangularMeshRefinedGenerator<2>::generate(const boost::shared_ptr<plask::GeometryObjectD<2>>& geometry)
{
    auto mesh = makeGeometryGrid(geometry);
    auto axis0 = dynamic_pointer_cast<OrderedAxis>(mesh->axis0),
         axis1 = dynamic_pointer_cast<OrderedAxis>(mesh->axis1);
    getAxis(axis0, geometry, 0);
    getAxis(axis1, geometry, 1);

    auto minmax0 = getMinMax(axis0), minmax1 = getMinMax(axis1);
    double asp0 = minmax0.second / minmax1.first, asp1 = minmax1.second / minmax0.first;
    double limit = (1+SMALL) * aspect;
    while (aspect != 0. && (asp0 > limit || asp1 > limit)) {
        if (asp0 > aspect) divideLargestSegment(axis0);
        if (asp1 > aspect) divideLargestSegment(axis1);
        minmax0 = getMinMax(axis0); minmax1 = getMinMax(axis1);
        asp0 = minmax0.second / minmax1.first; asp1 = minmax1.second / minmax0.first;
    }

    mesh->setOptimalIterationOrder();
    writelog(LOG_DETAIL, "mesh.Rectangular2D::{}: Generating new mesh ({:d}x{:d}, max. aspect {:.0f}:1)", name(),
             mesh->axis0->size(), mesh->axis1->size(), max(asp0, asp1));
    return mesh;
}

template <> shared_ptr<MeshD<3>>
RectangularMeshRefinedGenerator<3>::generate(const boost::shared_ptr<plask::GeometryObjectD<3>>& geometry)
{
    auto mesh = makeGeometryGrid(geometry);
    auto axis0 = dynamic_pointer_cast<OrderedAxis>(mesh->axis0),
         axis1 = dynamic_pointer_cast<OrderedAxis>(mesh->axis1),
         axis2 = dynamic_pointer_cast<OrderedAxis>(mesh->axis2);
    getAxis(axis0, geometry, 0);
    getAxis(axis1, geometry, 1);
    getAxis(axis2, geometry, 2);

    auto minmax0 = getMinMax(axis0), minmax1 = getMinMax(axis1), minmax2 = getMinMax(axis2);
    double asp0 = minmax0.second / min(minmax1.first, minmax2.first),
           asp1 = minmax1.second / min(minmax0.first, minmax2.first),
           asp2 = minmax2.second / min(minmax0.first, minmax1.first);
    double limit = (1+SMALL) * aspect;
    while (aspect != 0. && (asp0 > limit || asp1 > limit || asp2 > limit)) {
        if (asp0 > aspect) divideLargestSegment(axis0);
        if (asp1 > aspect) divideLargestSegment(axis1);
        if (asp2 > aspect) divideLargestSegment(axis2);
        minmax0 = getMinMax(axis0); minmax1 = getMinMax(axis1); minmax2 = getMinMax(axis2);
        asp0 = minmax0.second / min(minmax1.first, minmax2.first);
        asp1 = minmax1.second / min(minmax0.first, minmax2.first);
        asp2 = minmax2.second / min(minmax0.first, minmax1.first);
    }

    mesh->setOptimalIterationOrder();
    writelog(LOG_DETAIL, "mesh.Rectangular3D::{}: Generating new mesh ({:d}x{:d}x{:d}, max. aspect {:.0f}:1)", name(),
                          mesh->axis0->size(), mesh->axis1->size(), mesh->axis2->size(), max(asp0, max(asp1, asp2)));
    return mesh;
}


template <int dim>
shared_ptr<OrderedAxis> RectangularMeshDivideGenerator<dim>::processAxis(shared_ptr<OrderedAxis> axis, const shared_ptr<GeometryObjectD<DIM>>& geometry, size_t dir)
{
    assert(bool(axis));
    OrderedAxis::WarningOff warning_off(axis);

    if (pre_divisions[dir] == 0) pre_divisions[dir] = 1;
    if (post_divisions[dir] == 0) post_divisions[dir] = 1;

    OrderedAxis& result = *axis.get();

    // Pre-divide each object
    double x = *result.begin();
    std::vector<double> points; points.reserve((pre_divisions[dir]-1)*(result.size()-1));
    for (auto i = result.begin()+1; i!= result.end(); ++i) {
        double w = *i - x;
        for (size_t j = 1; j != pre_divisions[dir]; ++j) points.push_back(x + w*j/pre_divisions[dir]);
        x = *i;
    }
    result.addOrderedPoints(points.begin(), points.end());

    // Now ensure, that the grids do not change to quickly
    if (result.size() > 2 && gradual) {
        size_t end = result.size()-2;
        double w_prev = INFINITY, w = result[1]-result[0], w_next = result[2]-result[1];
        for (size_t i = 0; i <= end;) {
            bool goon = true;
            if (w > 2.001*w_prev) { // .0001 is for border case w == 2*w_prev, to avoid division even in presence of numerical error
                if (result.addPoint(0.5 * (result[i] + result[i+1]))) {
                    ++end;
                    w = w_next = result[i+1] - result[i];
                    goon = false;
                }
            } else if (w > 2.001*w_next) {
                if (result.addPoint(0.5 * (result[i] + result[i+1]))) {
                    ++end;
                    w_next = result[i+1] - result[i];
                    if (i) {
                        --i;
                        w = w_prev;
                        w_prev = (i == 0)? INFINITY : result[i] - result[i-1];
                    } else
                        w = w_next;
                    goon = false;
                }
            }
            if (goon) {
                ++i;
                w_prev = w;
                w = w_next;
                w_next = (i < end)? result[i+2] - result[i+1] : INFINITY;
            }
        }
    }

    // Finally divide each object in post- division
    x = *result.begin();
    points.clear(); points.reserve((post_divisions[dir]-1)*(result.size()-1));
    for (auto i = result.begin()+1; i!= result.end(); ++i) {
        double w = *i - x;
        for (size_t j = 1; j != post_divisions[dir]; ++j) points.push_back(x + w*j/post_divisions[dir]);
        x = *i;
    }

    result.addOrderedPoints(points.begin(), points.end());

    return axis;
}


template<>
RectangularMeshSmoothGenerator<1>::RectangularMeshSmoothGenerator():
    finestep {0.005}, maxstep {INFINITY}, factor {1.2} {}

template<>
RectangularMeshSmoothGenerator<2>::RectangularMeshSmoothGenerator():
    finestep {0.005, 0.005}, maxstep {INFINITY, INFINITY}, factor {1.2, 1.2} {}

template<>
RectangularMeshSmoothGenerator<3>::RectangularMeshSmoothGenerator():
    finestep {0.005, 0.005, 0.005}, maxstep {INFINITY, INFINITY, INFINITY}, factor {1.2, 1.2, 1.2} {}


template <int dim>
shared_ptr<OrderedAxis> RectangularMeshSmoothGenerator<dim>::processAxis(shared_ptr<OrderedAxis> axis, const shared_ptr<GeometryObjectD<DIM>>& geometry, size_t dir)
{
    OrderedAxis::WarningOff warning_off(axis);

    // Next divide each object
    double x = *axis->begin();
    std::vector<double> points; //points.reserve(...);
    for (auto i = axis->begin()+1; i!= axis->end(); ++i) {
        double width = *i - x;
        if (width+OrderedAxis::MIN_DISTANCE <= finestep[dir])
            continue;
        if (factor[dir] == 1.) {
            double m = ceil(width / finestep[dir]);
            double d = width / m;
            for (size_t i = 1, n = size_t(m); i < n; ++i) points.push_back(x + i*d);
            continue;
        }
        double logf = log(factor[dir]);
        double maxm = floor(log(maxstep[dir]/finestep[dir]) / logf + OrderedAxis::MIN_DISTANCE);
        double m = ceil(log(0.5*(width-OrderedAxis::MIN_DISTANCE)/finestep[dir]*(factor[dir]-1.)+1.) / logf) - 1.; // number of points in one half
        size_t lin = 0;
        if (m > maxm) { m = maxm; lin = 1; }
        size_t n = size_t(m);
        double end = finestep[dir] * (pow(factor[dir],m) - 1.) / (factor[dir] - 1.);
        double last = finestep[dir] * pow(factor[dir],m);
        if (lin) {
            lin = size_t(ceil((width-2.*end) / last));
        } else if (width - 2.*end <= last) {
            lin = 1;
        } else {
            lin = 2;
        }
        double s = finestep[dir] * 0.5*width / (end+0.5*lin*last);
        double dx = 0.;
        for (size_t i = 0; i < n; ++i) {
            dx += s; s *= factor[dir];
            points.push_back(x + dx);
        }
        for (size_t i = 0; i < lin; ++i) {
            dx += s; points.push_back(x + dx);
        }
        for (size_t i = 1; i < n; ++i) {
            s /= factor[dir]; dx += s;
            points.push_back(x + dx);
        }
        x = *i;
    }
    axis->addOrderedPoints(points.begin(), points.end());

    return axis;
}


template <int dim>
void RectangularMeshRefinedGenerator<dim>::fromXML(XMLReader& reader, const Manager& manager)
{
    if (reader.getNodeName() == "warnings") {
        warn_missing = reader.getAttribute<bool>("missing", true);
        warn_multiple = reader.getAttribute<bool>("multiple", true);
        warn_outside = reader.getAttribute<bool>("outside", true);
        reader.requireTagEnd();
    } else if (reader.getNodeName() == "refinements") {
        while (reader.requireTagOrEnd()) {
            if (reader.getNodeName() != "axis0" && (dim == 1 || (reader.getNodeName() != "axis1" && (dim == 2 || reader.getNodeName() != "axis2")))) {
                if (dim == 1) throw XMLUnexpectedElementException(reader, "<axis0>");
                if (dim == 2) throw XMLUnexpectedElementException(reader, "<axis0> or <axis1>");
                if (dim == 3) throw XMLUnexpectedElementException(reader, "<axis0>, <axis1>, or <axis2>");
            }
            auto direction = (reader.getNodeName() == "axis0")? typename Primitive<RectangularMeshRefinedGenerator<dim>::DIM>::Direction(0) :
                             (reader.getNodeName() == "axis1")? typename Primitive<RectangularMeshRefinedGenerator<dim>::DIM>::Direction(1) :
                                                                typename Primitive<RectangularMeshRefinedGenerator<dim>::DIM>::Direction(2);
            weak_ptr<GeometryObjectD<RectangularMeshRefinedGenerator<dim>::DIM>> object
                = manager.requireGeometryObject<GeometryObjectD<RectangularMeshRefinedGenerator<dim>::DIM>>(reader.requireAttribute("object"));
            PathHints path; if (auto pathattr = reader.getAttribute("path")) path = manager.requirePathHints(*pathattr);
            if (auto by = reader.getAttribute<unsigned>("by")) {
                double objsize = object.lock()->getBoundingBox().size()[unsigned(direction)];
                for (unsigned i = 1; i < *by; ++i) {
                    double pos = objsize * i / *by;
                    addRefinement(direction, object, path, pos);
                }
            } else if (auto every = reader.getAttribute<double>("every")) {
                double objsize = object.lock()->getBoundingBox().size()[unsigned(direction)];
                for (double pos = *every; pos < objsize; pos += *every)
                    addRefinement(direction, object, path, pos);
            } else if (auto pos = reader.getAttribute<double>("at")) {
                addRefinement(direction, object, path, *pos);
            } else
                throw XMLNoAttrException(reader, "at', 'every', or 'by");
            reader.requireTagEnd();
        }
    } else throw XMLUnexpectedElementException(reader, "proper generator configuration tag");
}

template struct PLASK_API RectangularMeshRefinedGenerator<1>;
template struct PLASK_API RectangularMeshRefinedGenerator<2>;
template struct PLASK_API RectangularMeshRefinedGenerator<3>;


template <typename GeneratorT>
static shared_ptr<MeshGenerator> readTrivialGenerator(XMLReader& reader, const Manager&)
{
    reader.requireTagEnd();
    return plask::make_shared<GeneratorT>();
}

static RegisterMeshGeneratorReader ordered_simplegenerator_reader  ("ordered.simple",   readTrivialGenerator<OrderedMesh1DSimpleGenerator>);
static RegisterMeshGeneratorReader rectangular2d_simplegenerator_reader("rectangular2d.simple", readTrivialGenerator<RectangularMesh2DSimpleGenerator>);
static RegisterMeshGeneratorReader rectangular3d_simplegenerator_reader("rectangular3d.simple", readTrivialGenerator<RectangularMesh3DSimpleGenerator>);


static shared_ptr<MeshGenerator> readRegularGenerator1(XMLReader& reader, const Manager&)
{
    double spacing = INFINITY;
    while (reader.requireTagOrEnd()) {
        if (reader.getNodeName() == "spacing") {
            spacing = reader.getAttribute<double>("every", spacing);
            reader.requireTagEnd();
        } else
            throw XMLUnexpectedElementException(reader, "<spacing>");
    }
    return plask::make_shared<OrderedMesh1DRegularGenerator>(spacing);
}

static shared_ptr<MeshGenerator> readRegularGenerator2(XMLReader& reader, const Manager&)
{
    double spacing0 = INFINITY,
           spacing1 = INFINITY;
    while (reader.requireTagOrEnd()) {
        if (reader.getNodeName() == "spacing") {
            if (reader.hasAttribute("every")) {
                if (reader.hasAttribute("every0")) throw XMLConflictingAttributesException(reader, "every", "every0");
                if (reader.hasAttribute("every1")) throw XMLConflictingAttributesException(reader, "every", "every1");
                spacing0 = spacing1 = reader.requireAttribute<double>("every");
            } else {
                spacing0 = reader.getAttribute<double>("every0", spacing0);
                spacing1 = reader.getAttribute<double>("every1", spacing1);
            }
            reader.requireTagEnd();
        } else
            throw XMLUnexpectedElementException(reader, "<spacing>");
    }
    return plask::make_shared<RectangularMesh2DRegularGenerator>(spacing0, spacing1);
}

static shared_ptr<MeshGenerator> readRegularGenerator3(XMLReader& reader, const Manager&)
{
    double spacing0 = INFINITY,
           spacing1 = INFINITY,
           spacing2 = INFINITY;
    while (reader.requireTagOrEnd()) {
        if (reader.getNodeName() == "spacing") {
            if (reader.hasAttribute("every")) {
                if (reader.hasAttribute("every0")) throw XMLConflictingAttributesException(reader, "every", "every0");
                if (reader.hasAttribute("every1")) throw XMLConflictingAttributesException(reader, "every", "every1");
                if (reader.hasAttribute("every2")) throw XMLConflictingAttributesException(reader, "every", "every2");
                spacing0 = spacing1 = reader.requireAttribute<double>("every");
            } else {
                spacing0 = reader.getAttribute<double>("every0", spacing0);
                spacing1 = reader.getAttribute<double>("every1", spacing1);
                spacing2 = reader.getAttribute<double>("every2", spacing2);
            }
            reader.requireTagEnd();
        } else
            throw XMLUnexpectedElementException(reader, "<spacing>");
    }
    return plask::make_shared<RectangularMesh3DRegularGenerator>(spacing0, spacing1, spacing2);
}

static RegisterMeshGeneratorReader ordered_regulargenerator_reader("ordered.regular", readRegularGenerator1);
static RegisterMeshGeneratorReader rectangular2d_regulargenerator_reader("rectangular2d.regular", readRegularGenerator2);
static RegisterMeshGeneratorReader rectangular3d_regulargenerator_reader("rectangular3d.regular", readRegularGenerator3);


template <int dim>
shared_ptr<MeshGenerator> readRectangularDivideGenerator(XMLReader& reader, const Manager& manager)
{
    auto result = plask::make_shared<RectangularMeshDivideGenerator<dim>>();

    std::set<std::string> read;
    while (reader.requireTagOrEnd()) {
        if (read.find(reader.getNodeName()) != read.end())
            throw XMLDuplicatedElementException(std::string("<generator>"), reader.getNodeName());
        read.insert(reader.getNodeName());
        if (reader.getNodeName() == "prediv") {
            plask::optional<size_t> into = reader.getAttribute<size_t>("by");
            if (into) {
                if (reader.hasAttribute("by0")) throw XMLConflictingAttributesException(reader, "by", "by0");
                if (reader.hasAttribute("by1")) throw XMLConflictingAttributesException(reader, "by", "by1");
                if (reader.hasAttribute("by2")) throw XMLConflictingAttributesException(reader, "by", "by2");
                for (int i = 0; i < dim; ++i) result->pre_divisions[i] = *into;
            } else
                for (int i = 0; i < dim; ++i) result->pre_divisions[i] = reader.getAttribute<size_t>(format("by{0}", i), 1);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "postdiv") {
            plask::optional<size_t> into = reader.getAttribute<size_t>("by");
            if (into) {
                if (reader.hasAttribute("by0")) throw XMLConflictingAttributesException(reader, "by", "by0");
                if (reader.hasAttribute("by1")) throw XMLConflictingAttributesException(reader, "by", "by1");
                if (reader.hasAttribute("by2")) throw XMLConflictingAttributesException(reader, "by", "by2");
                for (int i = 0; i < dim; ++i) result->post_divisions[i] = *into;
            } else
                for (int i = 0; i < dim; ++i) result->post_divisions[i] = reader.getAttribute<size_t>(format("by{0}", i), 1);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "options") {
            result->setGradual(reader.getAttribute<bool>("gradual", result->getGradual()));
            result->setAspect(reader.getAttribute<double>("aspect", result->getAspect()));
            reader.requireTagEnd();
        } else
            result->fromXML(reader, manager);
    }
    return result;
}

template struct PLASK_API RectangularMeshDivideGenerator<1>;
template struct PLASK_API RectangularMeshDivideGenerator<2>;
template struct PLASK_API RectangularMeshDivideGenerator<3>;

static RegisterMeshGeneratorReader ordered_dividinggenerator_reader("ordered.divide", readRectangularDivideGenerator<1>);
static RegisterMeshGeneratorReader rectangular2d_dividinggenerator_reader("rectangular2d.divide", readRectangularDivideGenerator<2>);
static RegisterMeshGeneratorReader rectangular3d_dividinggenerator_reader("rectangular3d.divide", readRectangularDivideGenerator<3>);


template <int dim>
shared_ptr<MeshGenerator> readRectangularSmoothGenerator(XMLReader& reader, const Manager& manager)
{
    auto result = plask::make_shared<RectangularMeshSmoothGenerator<dim>>();

    std::set<std::string> read;
    while (reader.requireTagOrEnd()) {
        if (read.find(reader.getNodeName()) != read.end())
            throw XMLDuplicatedElementException(std::string("<generator>"), reader.getNodeName());
        read.insert(reader.getNodeName());
        if (reader.getNodeName() == "steps") {
            plask::optional<double> small_op = reader.getAttribute<double>("small");	//dons't use small since some windows haders: #define small char
            if (small_op) {
                if (reader.hasAttribute("small0")) throw XMLConflictingAttributesException(reader, "small", "small0");
                if (reader.hasAttribute("small1")) throw XMLConflictingAttributesException(reader, "small", "small1");
                if (reader.hasAttribute("small2")) throw XMLConflictingAttributesException(reader, "small", "small2");
                for (int i = 0; i < dim; ++i) result->finestep[i] = *small_op;
            } else
                for (int i = 0; i < dim; ++i) result->finestep[i] = reader.getAttribute<double>(format("small{:d}", i), result->finestep[i]);
            plask::optional<double> large = reader.getAttribute<double>("large");
            if (large) {
                if (reader.hasAttribute("large0")) throw XMLConflictingAttributesException(reader, "large", "large0");
                if (reader.hasAttribute("large1")) throw XMLConflictingAttributesException(reader, "large", "large1");
                if (reader.hasAttribute("large2")) throw XMLConflictingAttributesException(reader, "large", "large2");
                for (int i = 0; i < dim; ++i) result->maxstep[i] = *large;
            } else
                for (int i = 0; i < dim; ++i) result->maxstep[i] = reader.getAttribute<double>(format("large{:d}", i), result->maxstep[i]);
            plask::optional<double> factor = reader.getAttribute<double>("factor");
            if (factor) {
                if (reader.hasAttribute("factor0")) throw XMLConflictingAttributesException(reader, "factor", "factor0");
                if (reader.hasAttribute("factor1")) throw XMLConflictingAttributesException(reader, "factor", "factor1");
                if (reader.hasAttribute("factor2")) throw XMLConflictingAttributesException(reader, "factor", "factor2");
                for (int i = 0; i < dim; ++i) result->factor[i] = *factor;
            } else
                for (int i = 0; i < dim; ++i) result->factor[i] = reader.getAttribute<double>(format("factor{:d}", i), result->factor[i]);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "options") {
            result->setAspect(reader.getAttribute<double>("aspect", result->getAspect()));
            reader.requireTagEnd();
        } else
            result->fromXML(reader, manager);
    }
    return result;
}

static RegisterMeshGeneratorReader ordered_smoothgenerator_reader("ordered.smooth", readRectangularSmoothGenerator<1>);
static RegisterMeshGeneratorReader rectangular2d_smoothgenerator_reader("rectangular2d.smooth", readRectangularSmoothGenerator<2>);
static RegisterMeshGeneratorReader rectangular3d_smoothgenerator_reader("rectangular3d.smooth", readRectangularSmoothGenerator<3>);

template struct PLASK_API RectangularMeshSmoothGenerator<1>;
template struct PLASK_API RectangularMeshSmoothGenerator<2>;
template struct PLASK_API RectangularMeshSmoothGenerator<3>;


} // namespace plask
