#ifndef PLASK__UTILS_XML_H
#define PLASK__UTILS_XML_H

#include <irrxml/irrXML.h>

#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>

namespace plask {

/**
 * XML pull parser.
 */
struct XMLReader {

    /// Enumeration for all xml nodes which are parsed by XMLReader
    enum NodeType {
            NODE_NONE = irr::io::EXN_NONE,   //<No xml node. This is usually the node if you did not read anything yet.
            NODE_ELEMENT = irr::io::EXN_ELEMENT,    //<A xml element, like <foo>
            NODE_ELEMENT_END = irr::io::EXN_ELEMENT_END,    //<End of an xml element, like </foo>
            NODE_TEXT = irr::io::EXN_TEXT,   //< Text within a xml element: <foo> this is the text. </foo>
            NODE_COMMENT =  irr::io::EXN_COMMENT,    //< An xml comment like &lt;!-- I am a comment --&gt; or a DTD definition.
            NODE_CDATA =  irr::io::EXN_CDATA,  //< An xml cdata section like &lt;![CDATA[ this is some CDATA ]]&gt;
            NODE_UNKNOWN = irr::io::EXN_UNKNOWN //< Unknown element
    };

private:
    irr::io::IrrXMLReader* irrReader;
    NodeType currentNodeType;

public:

    XMLReader(const char* file_name);

    XMLReader(std::istream& input);

    ~XMLReader() { delete irrReader; }

    NodeType getNodeType() const { return currentNodeType; }

    /** Reads forward to the next xml node.
        @return @c false only if there was no further node.
    */
    bool read();

    /// Returns if an element is an empty element, like \<foo /\>
    bool isEmptyElement() const { return irrReader->isEmptyElement(); }

    /**
     * Returns attribute count of the current XML node.

     * This is usually non 0 if the current node is NODE_ELEMENT, and the element has attributes.
     * @return amount of attributes of this xml node.
     */
    int getAttributeCount() const { return irrReader->getAttributeCount(); }

    //! Returns name of an attribute.
    /** @param idx: Zero based index, should be something between 0 and getAttributeCount()-1.
    @return Name of the attribute, 0 if an attribute with this index does not exist. */
    const char* getAttributeNameC(int idx) const { return irrReader->getAttributeName(idx); }

    //! Returns the value of an attribute.
    /** @param idx: Zero based index, should be something between 0 and getAttributeCount()-1.
    @return Value of the attribute, 0 if an attribute with this index does not exist. */
    const char* getAttributeValueC(int idx) const { return irrReader->getAttributeValue(idx); }

    //! Returns the value of an attribute.
    /** @param name: Name of the attribute.
    @return Value of the attribute, 0 if an attribute with this name does not exist. */
    const char* getAttributeValueC(const char* name) const { return irrReader->getAttributeValue(name); }

    //! Returns the value of an attribute in a safe way.
    /** Like getAttributeValue(), but does not
    return 0 if the attribute does not exist. An empty string ("") is returned then.
    @param name: Name of the attribute.
    @return Value of the attribute, and "" if an attribute with this name does not exist */
    const char* getAttributeValueOrEmptyC(const char* name) const { return irrReader->getAttributeValueSafe(name); }

    bool hasAttribute(const char* name) const { return getAttributeValueC(name) != 0; }

    bool hasAttribute(const std::string& name) const { return getAttributeValueC(name.c_str()) != 0; }

    //! Returns the name of the current node.
    /** Only non null, if the node type is NODE_ELEMENT.
    @return Name of the current node or 0 if the node has no name. */
    const char* getNodeNameC() const { return irrReader->getNodeName(); }

    std::string getNodeName() const { return irrReader->getNodeName(); }

    //! Returns data of the current node.
    /** Only non null if the node has some
    data and it is of type NODE_TEXT or NODE_UNKNOWN. */
    const char* getNodeDataC() const { return irrReader->getNodeData(); }

    template <typename T>
    inline T getAttribute(const char* name, T&& default_value) const {
        const char* attr_str = getAttributeValueC(name);
        if (attr_str == nullptr) return std::forward<T>(default_value);
        return boost::lexical_cast<T>(attr_str);
    }

    template <typename T>
    inline T getAttribute(const std::string& name, T&& default_value) const {
        return getAttribute<T>(name.c_str(), std::forward<T>(default_value));
    }

    boost::optional<std::string> getAttribute(const char* name) const;

    boost::optional<std::string> getAttribute(const std::string& name) const {
        return getAttribute(name.c_str());
    }

    template <typename T>
    boost::optional<T> getAttribute(const char* name) const {
        const char* attr_str = getAttributeValueC(name);
        if (attr_str == nullptr) return boost::optional<T>();
        return boost::lexical_cast<T>(attr_str);
    }

    template <typename T>
    boost::optional<T> getAttribute(const std::string& name) const {
        return getAttribute(name.c_str());
    }

    std::string requireAttribute(const char* attr_name) const;

    template <typename T>
    inline T requireAttribute(const char* name) const {
        return boost::lexical_cast<T>(requireAttribute(name));
    }

    template <typename T>
    inline T requireAttribute(const std::string& name) const {
        return requireAttribute<T>(name.c_str());
    }

    /**
     * Call read(), one or more time (skip comments).
     * @throw XMLUnexpectedEndException if there is no next element
     */
    void requireNext();

    void requireTag();

    void requireTagEnd(const std::string& tag);

    //void requireTagEndOrEmptyTag(const std::string& tag);

    /**
     * Skip XML comments.
     * @return @c true if read non-comment or @c false if XML data end
     */
    bool skipComments();
};

}

#endif // PLASK__UTILS_XML_H
