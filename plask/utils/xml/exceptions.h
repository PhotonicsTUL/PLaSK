#ifndef PLASK__UTILS_XML_EXCEPTIONS_H
#define PLASK__UTILS_XML_EXCEPTIONS_H

/** @file
This file includes definitions of exceptions used by XML reader and writer.
*/

#include <stdexcept>
#include "../format.h"
#include "../string.h"

namespace plask {

class XMLReader;

/**
 * Base class for all exceptions thrown by plask library.
 */
struct XMLException: public std::runtime_error {

    /**
     * @param reader current reader
     * @param msg error message
     */
    XMLException(const XMLReader& reader, const std::string& msg);

    /**
     * @param where indication where the error appeared
     * @param msg error message
     */
    XMLException(const std::string& where, const std::string& msg);

    /**
     * @param msg error message
     */
    XMLException(const std::string& msg);

};

/**
 * Exceptions of this class are thrown when the required attribute is not found in XML tag.
 */
struct XMLUnexpectedAttrException: public XMLException {
    /**
     * @param reader current reader
     * @param attr_name name of required attribute
     */
    XMLUnexpectedAttrException(const XMLReader& reader, const std::string& attr_name):
        XMLException(reader, "tag has unexpected attribute '" + attr_name + "'") {}
};

/**
 * Exceptions of this class are thrown when the required attribute is not found in XML tag.
 */
struct XMLNoAttrException: public XMLException {
    /**
     * @param reader current reader
     * @param attr_name name of required attribute
     */
    XMLNoAttrException(const XMLReader& reader, const std::string& attr_name):
        XMLException(reader, "tag has no required attribute '" + attr_name + "'") {}
};

/**
 * Exceptions of this class are thrown when the attribute has wrong value.
 */
struct XMLBadAttrException: public XMLException {
    /**
     * @param reader current reader
     * @param attr_name name of attribute
     * @param attr_value illegal value of attribute
     */
    XMLBadAttrException(const XMLReader& reader, const std::string& attr_name, const std::string& attr_value):
        XMLException(reader, "tag attribute '" + attr_name + "' has bad value \"" + attr_value + "\"") {}
};

/**
 * Exceptions of this class are thrown when XML file/data stream unexpected end.
 */
struct XMLUnexpectedEndException: public XMLException {
    XMLUnexpectedEndException(const XMLReader& reader):
        XMLException(reader, "unexpected end of data") {}
};

/**
 * Exceptions of this class are thrown when the type of XML element is different than expected.
 */
struct XMLUnexpectedElementException: public XMLException {
    /**
     * @param reader current reader
     * @param what_is_expected what is expected (typically tag with name, etc.)
     */
    XMLUnexpectedElementException(const XMLReader& reader, const std::string& what_is_expected):
        XMLException(reader, "expected " + what_is_expected) {}

    /**
     * @param reader current reader
     * @param what_is_expected what is expected (typically tag with name, etc.)
     * @param what_is_got what is got (typically tag with name, etc.)
     */
    XMLUnexpectedElementException(const XMLReader& reader, const std::string& what_is_expected, const std::string& what_is_got):
        XMLException(reader, "expected " + what_is_expected + ", got " + what_is_got + " instead") {}
};

/**
 * Exceptions of this class are thrown when illegal repetition of tag appears.
 */
struct XMLDuplicatedElementException: public XMLException {
    /**
     * @param reader current reader
     * @param duplicated name of duplicated thing
     */
    XMLDuplicatedElementException(const XMLReader& reader, const std::string& duplicated):
        XMLException(reader, duplicated + " should apprear only once in the current scope") {}
    /**
     * @param parent name of the parent tag
     * @param duplicated name of duplicated thing
     */
    XMLDuplicatedElementException(const std::string& parent, const std::string& duplicated):
        XMLException(parent, duplicated + " should apprear only once in the current scope") {}
};

/**
 * Exceptions of this class are thown if two optional attributes in XML conflict with each other.
 */
struct XMLConflictingAttributesException: public XMLException {
    /**
     * @param reader current reader
     * @param attr1 first attribute name
     * @param attr2 second attribute name
     */
    XMLConflictingAttributesException(const XMLReader& reader, const std::string& attr1, const std::string& attr2)
    : XMLException(reader, "conflicting attributes '" + attr1 + "' and '" + attr2 + "'") {}
};

/**
 * Exceptions of this type are thrown by XMLWriter class
 */
struct XMLWriterException: public std::runtime_error {
    /**
     * \param msg error message
     */
    XMLWriterException(const std::string& msg): std::runtime_error(msg) {}
};


} // namespace plask

#endif  //PLASK__UTILS_XML_EXCEPTIONS_H
