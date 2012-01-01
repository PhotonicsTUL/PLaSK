#include "irrxml/irrXML.h"

#include <string>
#include <boost/lexical_cast.hpp>

namespace plask {

typedef irr::io::IrrXMLReader XMLReader; 

namespace XML {

template <typename T>
inline T getAttribiute(XMLReader& reader, const char* name, T&& default_value) {
    const char* attr_str = reader.getAttributeValue(name);
    if (attr_str == nullptr) return default_value;
    return boost::lexical_cast<T>(attr_str);
}

}

}
