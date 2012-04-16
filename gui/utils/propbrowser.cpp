#include "propbrowser.h"

BrowserWithManagers::BrowserWithManagers(QtAbstractPropertyBrowser& browser)
    : browser(browser)
{
    browser.setFactoryForManager(&integer, &integerFact);
    browser.setFactoryForManager(&doubl, &doublFact);
    browser.setFactoryForManager(sizeF.subDoublePropertyManager(), &doublFact);
}

void BrowserWithManagers::clear() {
    browser.clear();
    integer.clear();
    doubl.clear();
    sizeF.clear();
    qDeleteAll(objectsToClear);
    objectsToClear.clear();
}
