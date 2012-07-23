#ifndef PLASK__UTILS_XML_WRITER_H
#define PLASK__UTILS_XML_WRITER_H

#include <string>
#include <deque>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "exceptions.h"

namespace plask {

/**
 * XML writer.
 */
class XMLWriter {
private:
    std::ostream out;

public:

    /// Element to write
    class Element {
    };

};

}   // namespace plask


#endif // PLASK__UTILS_XML_WRITER_H
