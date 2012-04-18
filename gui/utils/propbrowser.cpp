#include "propbrowser.h"

BrowserWithManagers::BrowserWithManagers(QtAbstractPropertyBrowser& browser)
    : browser(browser), alignerFact(QStringList() << "left" << "right" << "center")
{
    browser.setFactoryForManager(&integer, &integerFact);
    browser.setFactoryForManager(&doubl, &doublFact);
    browser.setFactoryForManager(sizeF.subDoublePropertyManager(), &doublFact);
    browser.setFactoryForManager(&string, &stringFact);
    browser.setFactoryForManager(&aligner, &alignerFact);
}

void BrowserWithManagers::clear() {
    browser.clear();
    integer.clear();
    doubl.clear();
    sizeF.clear();
    string.clear();
    aligner.clear();
    qDeleteAll(objectsToClear);
    objectsToClear.clear();
}
