#include <plask/log/log.h>
#include <plask/manager.h>

#include "generator_rectilinear.h"


namespace plask {

shared_ptr<RectilinearMesh2D> RectilinearMesh2DSimpleGenerator::generate(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    auto mesh = make_shared<RectilinearMesh2D>();

    std::vector<Box2D> boxes = geometry->getLeafsBoundingBoxes();

    for (auto& box: boxes) {
        mesh->axis0.addPoint(box.lower.c0);
        mesh->axis0.addPoint(box.upper.c0);
        mesh->axis1.addPoint(box.lower.c1);
        mesh->axis1.addPoint(box.upper.c1);
    }

    if (extend_to_zero) mesh->axis0.addPoint(0.);

    mesh->setOptimalIterationOrder();
    return mesh;
}

shared_ptr<RectilinearMesh3D> RectilinearMesh3DSimpleGenerator::generate(const shared_ptr<GeometryObjectD<3>>& geometry)
{
    auto mesh = make_shared<RectilinearMesh3D>();


    std::vector<Box3D> boxes = geometry->getLeafsBoundingBoxes();

    for (auto& box: boxes) {
        mesh->axis0.addPoint(box.lower.c0);
        mesh->axis0.addPoint(box.upper.c0);
        mesh->axis1.addPoint(box.lower.c1);
        mesh->axis1.addPoint(box.upper.c1);
        mesh->axis2.addPoint(box.lower.c2);
        mesh->axis2.addPoint(box.upper.c2);
    }

    mesh->setOptimalIterationOrder();
    return mesh;
}


RectilinearMesh1D RectilinearMesh2DDivideGenerator::get1DMesh(const RectilinearMesh1D& initial, const shared_ptr<GeometryObjectD<2>>& geometry, size_t dir)
{
    RectilinearMesh1D result = initial;

    // First add refinement points
    for (auto ref: refinements[dir]) {
        auto object = ref.first.first.lock();
        if (!object) {
             if (warn_missing) writelog(LOG_WARNING, "RectilinearMesh2DDivideGenerator: Refinement defined for object not existing any more.");
        } else {
            auto path = ref.first.second;
            auto boxes = geometry->getObjectBoundingBoxes(*object, path);
            auto origins = geometry->getObjectPositions(*object, path);
            if (warn_missing && boxes.size() == 0) writelog(LOG_WARNING, "RectilinearMesh2DDivideGenerator: Refinement defined for object absent from the geometry.");
            else if (warn_multiple && boxes.size() > 1) writelog(LOG_WARNING, "RectilinearMesh2DDivideGenerator: Single refinement defined for more than one object.");
            auto box = boxes.begin();
            auto origin = origins.begin();
            for (; box != boxes.end(); ++box, ++origin) {
                for (auto x: ref.second) {
                    double zero = (*origin)[dir];
                    double lower = box->lower[dir] - zero;
                    double upper = box->upper[dir] - zero;
                    if (warn_outside && (x < lower || x > upper))
                        writelog(LOG_WARNING, "RectilinearMesh2DDivideGenerator: Refinement at %1% outside of the object (%2% to %3%).",
                                            x, lower, upper);
                    result.addPoint(zero + x);
                }
            }
        }
    }

    // Next divide each object
    double x = *result.begin();
    std::vector<double> points; points.reserve((pre_divisions[dir]-1)*(result.size()-1));
    for (auto i = result.begin()+1; i!= result.end(); ++i) {
        double w = *i - x;
        for (size_t j = 1; j != pre_divisions[dir]; ++j) points.push_back(x + w*j/pre_divisions[dir]);
        x = *i;
    }
    result.addOrderedPoints(points.begin(), points.end());

    if (result.size() <= 2) return result;

    // Now ensure, that the grids do not change to quickly
    if (limit_change) {
        size_t end = result.size()-2;
        double w_prev = INFINITY, w = result[1]-result[0], w_next = result[2]-result[1];
        for (size_t i = 0; i <= end;) {
            if (w > 2.*w_prev) {
                result.addPoint(0.5 * (result[i] + result[i+1])); ++end;
                w = w_next = result[i+1] - result[i];
            } else if (w > 2.*w_next) {
                result.addPoint(0.5 * (result[i] + result[i+1])); ++end;
                w_next = result[i+1] - result[i];
                if (i) {
                    --i;
                    w = w_prev;
                    w_prev = (i == 0)? INFINITY : result[i] - result[i-1];
                } else
                    w = w_next;
            } else {
                ++i;
                w_prev = w;
                w = w_next;
                w_next = (i == end)? INFINITY : result[i+2] - result[i+1];
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

    return result;
}

shared_ptr<RectilinearMesh2D> RectilinearMesh2DDivideGenerator::generate(const shared_ptr<GeometryObjectD<2>>& geometry)
{
    RectilinearMesh2D initial;
    std::vector<Box2D> boxes = geometry->getLeafsBoundingBoxes();
    for (auto& box: boxes) {
        initial.axis0.addPoint(box.lower.c0);
        initial.axis0.addPoint(box.upper.c0);
        initial.axis1.addPoint(box.lower.c1);
        initial.axis1.addPoint(box.upper.c1);
    }

    auto mesh = make_shared<RectilinearMesh2D>(get1DMesh(initial.axis0, geometry, 0), get1DMesh(initial.axis1, geometry, 1));

    mesh->setOptimalIterationOrder();
    return mesh;
}






template <typename GeneratorT>
static shared_ptr<MeshGenerator> readTrivialGenerator(XMLReader& reader, const Manager&)
{
    reader.requireTagEnd();
    return make_shared<GeneratorT>();
}


static shared_ptr<MeshGenerator> readRectilinearMesh2DDivideGenerator(XMLReader& reader, const Manager& manager)
{
    auto result = make_shared<RectilinearMesh2DDivideGenerator>();

    std::set<std::string> read;
    while (reader.requireTagOrEnd()) {
        if (read.find(reader.getNodeName()) != read.end())
            throw XMLDuplicatedElementException(std::string("<generator>"), reader.getNodeName());
        read.insert(reader.getNodeName());
        if (reader.getNodeName() == "prediv") {
            boost::optional<size_t> into = reader.getAttribute<size_t>("by");
            if (into) {
                if (reader.hasAttribute("hor_by")) throw XMLConflictingAttributesException(reader, "by", "hor_by");
                if (reader.hasAttribute("vert_by")) throw XMLConflictingAttributesException(reader, "by", "vert_by");
                result->setPreDivision(*into);
            } else
                result->setPreDivision(reader.getAttribute<size_t>("hor_by", 1), reader.getAttribute<size_t>("vert_by", 1));
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "postdiv") {
            boost::optional<size_t> into = reader.getAttribute<size_t>("by");
            if (into) {
                if (reader.hasAttribute("hor_by")) throw XMLConflictingAttributesException(reader, "by", "hor_by");
                if (reader.hasAttribute("vert_by")) throw XMLConflictingAttributesException(reader, "by", "vert_by");
                result->setPostDivision(*into);
            } else
                result->setPostDivision(reader.getAttribute<size_t>("hor_by", 1), reader.getAttribute<size_t>("vert_by", 1));
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "dont_limit_change") {
            result->limit_change = false;
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "warnings") {
            result->warn_missing = reader.getAttribute<bool>("missing", true);
            result->warn_multiple = reader.getAttribute<bool>("multiple", true);
            result->warn_outside = reader.getAttribute<bool>("outside", true);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "refinements") {
            while (reader.requireTagOrEnd()) {
                if (reader.getNodeName() != "horizontal" && reader.getNodeName() != "vertical")
                    throw XMLUnexpectedElementException(reader, "<horizontal ...> of <vertical ...>");
                auto direction = (reader.getNodeName()=="horizontal")? Primitive<2>::DIRECTION_TRAN : Primitive<2>::DIRECTION_UP;
                weak_ptr<GeometryObjectD<2>> object =
                    manager.requireGeometryObject<GeometryObjectD<2>>(reader.requireAttribute("object"));
                double pos = reader.requireAttribute<double>("pos");
                auto path = reader.getAttribute("path");
                if (path) result->addRefinement(direction, object, manager.requirePathHints(*path), pos);
                else result->addRefinement(direction, object, pos);
                reader.requireTagEnd();
            }
        } else throw XMLUnexpectedElementException(reader, "proper 'divide' generator configuration tag");
    }
    return result;
}

static RegisterMeshGeneratorReader rectilinearmesh2d_simplegenerator_reader("rectilinear2d.simple", readTrivialGenerator<RectilinearMesh2DSimpleGenerator>);
static RegisterMeshGeneratorReader rectilinearmesh3d_simplegenerator_reader("rectilinear3d.simple", readTrivialGenerator<RectilinearMesh3DSimpleGenerator>);

static RegisterMeshGeneratorReader rectilinearmesh2d_dividinggenerator_reader("rectilinear2d.divide", readRectilinearMesh2DDivideGenerator);


} // namespace plask