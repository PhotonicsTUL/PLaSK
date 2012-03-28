#ifndef PLASK_GUI_UTILS_PROPBROWSER_H
#define PLASK_GUI_UTILS_PROPBROWSER_H

/** @file
 * This file includes utils functions and classes connected with QT property browser.
 */

#include <QtAbstractPropertyBrowser>

//managers
#include <QtSizeFPropertyManager>

/**
 * Includes property browser and set of property managers for it.
 */
struct BrowserWithManagers {

    QtAbstractPropertyBrowser& browser;

    // managers:
    QtSizeFPropertyManager sizeF;

    BrowserWithManagers(QtAbstractPropertyBrowser& browser);

private:    // factories:

};

#endif // PROPBROWSER_H
