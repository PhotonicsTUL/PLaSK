#include "document.h"
#include "material.h"

Document::Document(QtAbstractPropertyBrowser& browser): treeModel(*this), propertiesBrowser(browser) {
}

void Document::open(const QString &fileName) {
    undoStack.clear();
    //TODO support file names with non-asci char
    manager.loadGeometryFromFile(fileName.toStdString(), &NameOnlyMaterial::getInstance);
    treeModel.refresh(*this);
}
