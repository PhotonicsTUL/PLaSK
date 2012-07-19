#ifndef PLASK__UTILS_XML_H
#define PLASK__UTILS_XML_H

#include <irrxml/irrXML.h>

#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>
#include <vector>
#include <unordered_set>

namespace plask {

/**
 * XML pull parser.
 *
 * It makes some checks while reading and throw exeptions when XML document is valid:
 * - it check open/close tags,
 * - it checks if all attribiutes was read.
 */
struct XMLReader {

    /// Enumeration for all xml nodes which are parsed by XMLReader
    enum NodeType {
            NODE_NONE = irr::io::EXN_NONE,                  ///< No xml node. This is usually the node if you did not read anything yet.
            NODE_ELEMENT = irr::io::EXN_ELEMENT,            ///< A xml element, like \<foo>
            NODE_ELEMENT_END = irr::io::EXN_ELEMENT_END,    ///< End of an xml element, like \</foo>
            NODE_TEXT = irr::io::EXN_TEXT,                  ///< Text within a xml element: \<foo> this is the text. \</foo>
            NODE_COMMENT =  irr::io::EXN_COMMENT,           ///< An xml comment like &lt;!-- I am a comment --&gt; or a DTD definition.
            NODE_CDATA =  irr::io::EXN_CDATA,               ///< An xml cdata section like &lt;![CDATA[ this is some CDATA ]]&gt;
            NODE_UNKNOWN = irr::io::EXN_UNKNOWN             ///< Unknown element
    };

private:
    /// xml reader, low level
    irr::io::IrrXMLReader* irrReader;

    /// current type of node
    NodeType currentNodeType;

    /// path from root to current tag
    std::vector<std::string> path;

    /// attribiutes which was read
    std::unordered_set<std::string> read_attribiutes;

public:

    /**
     * Construct XML reader to read XML from given file.
     * @param file_name name of file to read
     */
    XMLReader(const char* file_name);

    /**
     * Construct XML reader to read XML from given @p input stream.
     * @param input input stream
     */
    XMLReader(std::istream& input);

#if __cplusplus >= 201103L
    /**
     * Move constructor.
     * @param to_move object to move from, should not be used but only delete after move
     */
    XMLReader(XMLReader&& to_move);

    /**
     * Move assigment operator.
     * @param to_move object to move from, should not be used but only delete after move
     * @return *this
     */
    XMLReader& operator=(XMLReader&& to_move);

    /// Disallow copy of reader.
    XMLReader(const XMLReader& to_copy) = delete;
    XMLReader& operator=(const XMLReader& to_copy) = delete;
#else
private:
    XMLReader(const XMLReader&) {}
    XMLReader& operator=(const XMLReader&) { return *this; }
public:
#endif

    ~XMLReader() { delete irrReader; }  //if irrReader is not nullptr throw exception if path is not empty

    /**
     * Swap states of @c this and @p to_swap.
     * @param to_swap object to swap with this
     */
    void swap(XMLReader& to_swap);

    /**
     * Get current type of node.
     * @return current type of node
     */
    NodeType getNodeType() const { return currentNodeType; }

    /** Reads forward to the next xml node.
        @return @c false only if there was no further node.
    */
    bool read();

    /**
     * Check if node is empty, like \<foo /\>.
     *
     * Note that empty nodes are comunicate by parser two times: as NODE_ELEMENT and next as NODE_ELEMENT_END.
     * So for \<foo /\> parser work just like for \<foo>\</foo> and only this method allow to check which notation was used.
     * @return if an element is an empty element, like \<foo /\>
     */
    bool isEmptyElement() const { return irrReader->isEmptyElement(); }

    /**
     * Get vector of names of all opened tags from root to current one.
     * @return vector of names of all opened tags, first is root, last is current tag
     */
    const std::vector<std::string> getPath() const { return path; }

    /**
     * Returns attribute count of the current XML node.

     * This is usually non 0 if the current node is NODE_ELEMENT, and the element has attributes.
     * @return amount of attributes of this xml node.
     */
    int getAttributeCount() const { return irrReader->getAttributeCount(); }

    /* Returns name of an attribute.
    @param idx: Zero based index, should be something between 0 and getAttributeCount()-1.
    @return Name of the attribute, 0 if an attribute with this index does not exist. */
    //const char* getAttributeNameC(int idx) const { return irrReader->getAttributeName(idx); }

    /* Returns the value of an attribute.
     @param idx: Zero based index, should be something between 0 and getAttributeCount()-1.
     @return Value of the attribute, 0 if an attribute with this index does not exist. */
    //const char* getAttributeValueC(int idx) const { return irrReader->getAttributeValue(idx); }

    //! Returns the value of an attribute.
    /** @param name: Name of the attribute.
    @return Value of the attribute, 0 if an attribute with this name does not exist. */
    const char* getAttributeValueC(const std::string& name) const;

    /**
     * Mark argument with given name as read, so parser not throw an exeption if this attribiute will be not read.
     * @param name name of attribiute to ignore
     */
    void ignoreAttribute(const std::string& name) { getAttributeValueC(name); }

    /* Returns the value of an attribute in a safe way.

    Like getAttributeValue(), but does not
    return 0 if the attribute does not exist. An empty string ("") is returned then.
    @param name: Name of the attribute.
    @return Value of the attribute, and "" if an attribute with this name does not exist */
    //const char* getAttributeValueOrEmptyC(const char* name) const { return irrReader->getAttributeValueSafe(name); }

    //bool hasAttribute(const char* name) const { return getAttributeValueC(name) != 0; }

    /**
     * Check if current node has attribiute with given @p name.
     * @param name attribiute name
     * @return @c true only if current node has attribiute with given @p name
     */
    bool hasAttribute(const std::string& name) const { return getAttributeValueC(name) != 0; }

    /**
     * Returns the name of the current node.
     *
     * Only non null, if the node type is NODE_ELEMENT.
     * @return name of the current node or 0 if the node has no name.
     */
    const char* getNodeNameC() const { return irrReader->getNodeName(); }

    /**
     * Returns the name of the current node.
     *
     * Bechaviour is undefined if name is not defined.
     * @return name of the current node
     */
    std::string getNodeName() const { return irrReader->getNodeName(); }

    /** Returns data of the current node.
     *
     * Only non null if the node has some data and it is of type NODE_TEXT or NODE_UNKNOWN.
     * @return data of the current node
     */
    const char* getNodeDataC() const { return irrReader->getNodeData(); }

    /**
     * Get value of attribiute with given @p name, or @p default_value if attribiute with given @p name is not defined in current node.
     * @param name name of attribiute
     * @param default_value default value which will be return when attribiute with given @p name is not defined
     * @return attribiute with given @p name, or @p default_value if attribiute with given @p name is not defined in current node
     * @tparem required type of value, boost::lexical_cast\<T> will be used to obtain value of this type from string
     */
    template <typename T>
    inline T getAttribute(const std::string& name, T&& default_value) const {
        const char* attr_str = getAttributeValueC(name);
        if (attr_str == nullptr) return std::forward<T>(default_value);
        return boost::lexical_cast<T>(attr_str);
    }

    boost::optional<std::string> getAttribute(const std::string& name) const;

    template <typename T>
    boost::optional<T> getAttribute(const std::string& name) const {
        const char* attr_str = getAttributeValueC(name);
        if (attr_str == nullptr) return boost::optional<T>();
        return boost::lexical_cast<T>(attr_str);
    }

    std::string requireAttribute(const std::string& attr_name) const;

    template <typename T>
    inline T requireAttribute(const std::string& name) const {
        return boost::lexical_cast<T>(requireAttribute(name));
    }

    /**
     * Call read(), one or more time skipping comments.
     * @throw XMLUnexpectedEndException if there is no next element
     */
    void requireNext();

    /**
     * Call requireNext() and next check if current element is tag opening. Throw exception if it's not.
     */
    void requireTag();

    /**
     * Call requireNext() and next check if current element is tag closing. Throw exception if it's not.
     */
    void requireTagEnd();

    //void requireTagEndOrEmptyTag(const std::string& tag);

    /**
     * Skip XML comments.
     * @return @c true if read non-comment or @c false if XML data end
     */
    bool skipComments();
};

}   // namespace plask

namespace std {
inline void swap(plask::XMLReader& a, plask::XMLReader& b) { a.swap(b); }
}   // namespace std

#endif // PLASK__UTILS_XML_H
