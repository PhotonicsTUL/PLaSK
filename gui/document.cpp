#include "document.h"

Document::Document(QtAbstractPropertyBrowser& browser): treeModel(*this), propertiesBrowser(browser) {
}

void Document::open(const QString &fileName) {
    //TODO support file names with non-asci char
    manager.loadFromFile(fileName.toStdString());
    treeModel.refresh(*this);
}
