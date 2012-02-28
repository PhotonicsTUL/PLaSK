#include "path.h"

#include <plask/config.h>

namespace plask {

void PathHints::addHint(const Hint& hint) {
    addHint(hint.first, hint.second);
}

void PathHints::addHint(weak_ptr<GeometryElement> container, weak_ptr<GeometryElement> child) {
    hintFor[container].insert(child);
}

std::set<shared_ptr<GeometryElement>> PathHints::getChildren(shared_ptr<const GeometryElement> container) {
    std::set<shared_ptr<GeometryElement>> result;
    auto e = hintFor.find(const_pointer_cast<GeometryElement>(container));
    if (e == hintFor.end()) return result;
    if (e->first.expired()) {   //container was deleted, new container is under same address as was old one
        hintFor.erase(e);
        return result;
    }
    for (auto weak_child_iter = e->second.begin(); weak_child_iter != e->second.end(); ) {
        shared_ptr<GeometryElement> child = weak_child_iter->lock();
        if (!child)        //child was deleted
            e->second.erase(weak_child_iter++);
        else {
            result.insert(child);
            ++weak_child_iter;
        }
    }
    if (e->second.empty()) hintFor.erase(e);    //we remove all constraints
    return result;
}

std::set<shared_ptr<GeometryElement>> PathHints::getChildren(shared_ptr<const GeometryElement> container) const {
    std::set<shared_ptr<GeometryElement>> result;
    auto e = hintFor.find(const_pointer_cast<GeometryElement>(container));
    if (e == hintFor.end() || e->first.expired()) return result;
    for (auto child_weak: e->second) {
        shared_ptr<GeometryElement> child = child_weak.lock();
        if (child) result.insert(child);
    }
    return result;
}

void PathHints::cleanDeleted() {
    for(auto i = hintFor.begin(); i != hintFor.end(); )
        if (i->first.expired())
            hintFor.erase(i++);
        else {
            for (auto weak_child_iter = i->second.begin(); weak_child_iter != i->second.end(); ) {
                if (weak_child_iter->expired())
                    i->second.erase(weak_child_iter);
                else
                    ++weak_child_iter;
            }
            if (i->second.empty())
                hintFor.erase(i++);
            else
                ++i;
        }
}

//----------------- Path ------------------------------------------

void Path::addElements(const GeometryElement::Subtree* path_nodes) {
    if (!elements.empty() && elements.back() == path_nodes->element) {
        if (path_nodes->children.empty()) return;
        path_nodes = &(path_nodes->children[0]);
    }
    while (true) {
        elements.push_back(path_nodes->element);
        if (path_nodes->children.empty()) return;
        path_nodes = &(path_nodes->children[0]);
    }
}

void Path::addElements(const GeometryElement::Subtree& paths) {
    if (paths.isWithBranches())
        throw Exception("There are more than one path.");
    addElements(&paths);
}

/*Path& Path::operator+=(const GeometryElement::Subtree& paths) {
    if (paths.empty()) return;
    addElements(paths);
}

Path& Path::operator+=(const PathHints::Hint& hint);

Path& Path::operator+=(const GeometryElement& last) {
    if (elements.empty())
        elements.push_back(last);
    else {
        GeometryElement::Subtree path = elements.back()->findPathsTo(last);


        addElements();
    }
}*/

}   //namespace plask
