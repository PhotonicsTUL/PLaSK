#include <plask/log/log.h>
#include "generator_rectilinear.h"

namespace plask {

shared_ptr<RectilinearMesh2D> RectilinearMesh2DSimpleGenerator::generate(const shared_ptr<GeometryElementD<2>>& geometry)
{
    auto mesh = make_shared<RectilinearMesh2D>();

    std::vector<Box2D> boxes = geometry->getLeafsBoundingBoxes();

    for (auto& box: boxes) {
        mesh->c0.addPoint(box.lower.c0);
        mesh->c0.addPoint(box.upper.c0);
        mesh->c1.addPoint(box.lower.c1);
        mesh->c1.addPoint(box.upper.c1);
    }

    mesh->setOptimalIterationOrder();
    return mesh;
}

shared_ptr<RectilinearMesh3D> RectilinearMesh3DSimpleGenerator::generate(const shared_ptr<GeometryElementD<3>>& geometry)
{
    auto mesh = make_shared<RectilinearMesh3D>();


    std::vector<Box3D> boxes = geometry->getLeafsBoundingBoxes();

    for (auto& box: boxes) {
        mesh->c0.addPoint(box.lower.c0);
        mesh->c0.addPoint(box.upper.c0);
        mesh->c1.addPoint(box.lower.c1);
        mesh->c1.addPoint(box.upper.c1);
        mesh->c2.addPoint(box.lower.c2);
        mesh->c2.addPoint(box.upper.c2);
    }

    mesh->setOptimalIterationOrder();
    return mesh;
}


RectilinearMesh1D RectilinearMesh2DDivideGenerator::get1DMesh(const RectilinearMesh1D& initial, const shared_ptr<GeometryElementD<2>>& geometry, size_t dir)
{
    RectilinearMesh1D result = initial;

    // First add refinement points
    for (auto ref: refinements[dir]) {
        auto element = ref.first.first.lock();
        if (!element) {
             if (warn_multiple) writelog(LOG_WARNING, "RectilinearMesh2DDivideGenerator: Refinement defined for object not existing any more.");
        } else {
            auto path = ref.first.second;
            auto boxes = geometry->getElementBoundingBoxes(*element, path);
            auto origins = geometry->getElementPositions(*element, path);
            if (warn_multiple) {
                if (boxes.size() == 0) writelog(LOG_WARNING, "RectilinearMesh2DDivideGenerator: Refinement defined for object absent from the geometry.");
                else if (boxes.size() > 1) writelog(LOG_WARNING, "RectilinearMesh2DDivideGenerator: Single refinement defined for more than one object.");
            }
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

    // First divide each element
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

    // Finally divide each element in post- division
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

shared_ptr<RectilinearMesh2D> RectilinearMesh2DDivideGenerator::generate(const shared_ptr<GeometryElementD<2>>& geometry)
{
    RectilinearMesh2D initial;
    std::vector<Box2D> boxes = geometry->getLeafsBoundingBoxes();
    for (auto& box: boxes) {
        initial.c0.addPoint(box.lower.c0);
        initial.c0.addPoint(box.upper.c0);
        initial.c1.addPoint(box.lower.c1);
        initial.c1.addPoint(box.upper.c1);
    }

    auto mesh = make_shared<RectilinearMesh2D>();
    mesh->c0 = get1DMesh(initial.c0, geometry, 0);
    mesh->c1 = get1DMesh(initial.c1, geometry, 1);

    mesh->setOptimalIterationOrder();
    return mesh;
}






template <typename GeneratorT>
static shared_ptr<MeshGenerator> readTrivialGenerator(XMLReader& reader)
{
    reader.requireTagEnd();
    return make_shared<GeneratorT>();
}


static shared_ptr<MeshGenerator> readRectilinearMesh2DDivideGenerator(XMLReader& reader)
{
    auto result = make_shared<RectilinearMesh2DDivideGenerator>();

    while (reader.requireTagOrEnd()) {
        if (reader.getNodeName() == "prediv") {
            std::vector<std::string> divs; divs.reserve(2);
            std::string text = reader.requireAttribute("value");
            boost::split(divs, text, boost::is_any_of(", \n"), boost::token_compress_on);
            if (divs.size() == 1) result->setPreDivision(boost::lexical_cast<size_t>(divs[0]));
            else if (divs.size() == 2) result->setPreDivision(boost::lexical_cast<size_t>(divs[0]), boost::lexical_cast<size_t>(divs[1]));
            else throw XMLUnexpectedElementException("one or two integers");
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "postdiv") {
            std::vector<std::string> divs; divs.reserve(2);
            std::string text = reader.requireAttribute("value");
            boost::split(divs, text, boost::is_any_of(", \n"), boost::token_compress_on);
            if (divs.size() == 1) result->setPostDivision(boost::lexical_cast<size_t>(divs[0]));
            else if (divs.size() == 2) result->setPostDivision(boost::lexical_cast<size_t>(divs[0]), boost::lexical_cast<size_t>(divs[1]));
            else throw XMLUnexpectedElementException("one or two integers");
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "limit_change") {
            result->limit_change = reader.getAttribute<bool>("value", true);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "warn_none") {
            result->warn_none = reader.getAttribute<bool>("value", true);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "warn_multiple") {
            result->warn_multiple = reader.getAttribute<bool>("value", true);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "warn_outside") {
            result->warn_outside = reader.getAttribute<bool>("value", true);
            reader.requireTagEnd();
        } else if (reader.getNodeName() == "refinements") {
            //TODO
        } else throw XMLUnexpectedElementException("'divide' generator configuration");
    }
    return result;
}

static RegisterMeshGeneratorReader rectilinearmesh2d_simplegenerator_reader("rectilinear2d.simple", readTrivialGenerator<RectilinearMesh2DSimpleGenerator>);
static RegisterMeshGeneratorReader rectilinearmesh3d_simplegenerator_reader("rectilinear3d.simple", readTrivialGenerator<RectilinearMesh3DSimpleGenerator>);

static RegisterMeshGeneratorReader rectilinearmesh2d_dividinggenerator_reader("rectilinear2d.divide", readRectilinearMesh2DDivideGenerator);


} // namespace plask