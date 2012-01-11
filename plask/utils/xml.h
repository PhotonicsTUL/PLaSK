#ifndef PLASK__XML_H
#define PLASK__XML_H

#include <irrxml/irrXML.h>

#include <string>
#include <boost/lexical_cast.hpp>

namespace plask {

/**
 * XML pull parser.
 */
typedef irr::io::IrrXMLReader XMLReader; 

namespace XML {

template <typename T>
inline T getAttribiute(XMLReader& reader, const char* name, T&& default_value) {
    const char* attr_str = reader.getAttributeValue(name);
    if (attr_str == nullptr) return default_value;
    return boost::lexical_cast<T>(attr_str);
}

template <typename T>
inline T getAttribiute(XMLReader& reader, const std::string& name, T&& default_value) {
    return getAttribiute<T>(reader, name.c_str(), std::forward<T>(default_value));
}

std::string requireAttr(XMLReader &source, const char* attr_name);

template <typename T>
inline T requireAttr(XMLReader& reader, const char* name) {
    return boost::lexical_cast<T>(requireAttr(reader, name));
}

template <typename T>
inline T requireAttr(XMLReader& reader, const std::string& name) {
    return requireAttr<T>(reader, name.c_str());
}

/**
 * Call reader.read(), one or more time (skip comments).
 * @throw XMLUnexpectedEndException if there is no next element
 */
void requireNext(XMLReader& reader);

void requireTag(XMLReader& reader);

void requireTagEnd(XMLReader& reader);

/**
 * Skip XML comments.
 * @return @c true if read non-comment or @c false if XML data end
 */
bool skipComments(XMLReader& reader);

///Allow to read XML from standard C++ input stream (std::istream).
struct StreamReaderCallback: public irr::io::IFileReadCallBack {
    
    ///Stream to read from.
    std::istream& input;
    
    ///Size of stream, number of bytes to read.
    int size;
    
    /**
     * @param input stream to read from
     */
    StreamReaderCallback(std::istream& input);
    
    virtual int read(void* buffer, int sizeToRead);

    virtual int getSize() { return size; }
    
};

}

}

#endif // PLASK__XML_H
