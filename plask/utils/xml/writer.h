#ifndef PLASK__UTILS_XMLWRITER_H
#define PLASK__UTILS_XMLWRITER_H

#include <string>
#include <iostream>

#include "exceptions.h"
#include "../format.h"

namespace plask {

/**
 * Object of this class help produce valid XML documents.
 *
 * It hold std output stream and helps format valid XML data written to this stream.
 *
 * Example:
 * @code
 * XMLWriter w(some_out_stream);
 * w.writeHeader();
 * w.addElement("a").attr("x", 1).attr("y", 2); // \<a x="1" y="2"/>
 * {
 *  XMLWriter::Element outer(w, "o"); // or XMLWriter::Element outer = w.addElement("o");
 *  outer.attr("o_attr", "v");
 *  w.element("i").writeText("inner content");
 *  // here outer.attr(...) is not allowed and will throw an exception
 * }    // \<o o_attr="v">\<i>inner content\</i>\</o>
 * w.addElement("o").addElement("i1").attr("a", 1).addElement("i2");  // \<o>\<i1 a="1">\<i2/>\</i1>\</o>
 * w.addElement("o").addElement("i1").attr("a", 1).end().addElement("i2");    // \<o>\<i1 a="1"/>\<i2/>\</o>
 * @endcode
 */
struct XMLWriter {

    /**
     * Represent single XML element connected with writer.
     *
     * Constructor put in stream opening tag and destructor closing one.
     */
    struct Element {

        /// Tag name.
        std::string name;

        /// XML writer
        XMLWriter* writer;

        /// Parent element.
        Element* parent;

        /// @c true only if this tag is open and allow to append atribiutes
        bool attributesStillAlowed;

        /**
         * Construct element with given @p name, write to steam openning of element tag.
         * @param writer XML writer where element should be append
         * @param name name of elements tag
         */
        Element(XMLWriter& writer, const std::string& name);

        /**
         * Construct element with given @p name, write to steam openning of element tag.
         * @param writer XML writer where element should be append
         * @param name name of elements tag
         */
        Element(XMLWriter& writer, std::string&& name);

        /**
         * Construct element with given @p name, write to steam openning of element tag.
         * @param parent parent element, must by recently added, not closed one
         * @param name name of elements tag
         */
        Element(Element& parent, const std::string& name);

        /**
         * Construct element with given @p name, write to steam openning of element tag.
         * @param parent parent element, must by recently added, not closed one
         * @param name name of elements tag
         */
        Element(Element& parent, std::string&& name);

        /// Disallow to copy element.
        Element(const Element&) = delete;

        /// Disallow to copy element.
        Element& operator=(const Element&) = delete;

        /// Move is allowed.
        Element(Element&& to_move);

        /// Move is allowed.
        Element& operator=(Element&& to_move);

        /// Close element tag.
        ~Element();

        /**
         * @return 0 for root, 1 for child of root, and so on
         */
        std::size_t getLevel() const;

        /**
         * Append attribute to this element.
         * @param attr_name name of attribute to append
         * @param attr_value value of attribute to append
         */
        Element& attr(const std::string& attr_name, const std::string& attr_value);

        /**
         * Append attribute to this element.
         * @param attr_name name of attribute to append
         * @param attr_value value of attribute to append, will be change to string using boost::lexical_cast
         */
        template <typename ValueT>
        Element& attr(const std::string& attr_name, const ValueT& attr_value) {
            return attr(attr_name, str(attr_value));
        }

        /**
         * Append text content to this element.
         * @param str content to append
         */
        Element& writeText(const char* str);

        /**
         * Append text content to this element.
         * @param str content to append
         */
        Element& writeText(const std::string& str) { return writeText(str.c_str()); }

        /**
         * Append text content to this element.
         * @param value content to append, will be change to string using boost::lexical_cast
         */
        template <class T>
        Element& writeText(const T& value) {
            return writeText(str(std::forward<T>(value)));
        }

        Element& writeCDATA(const std::string& str);

        /**
         * Create sub-element of this.
         * @return sub-element of this
         */
        template <typename name_t>
        Element addElement(name_t&& name) { return Element(*this, std::forward<name_t>(name)); }

        /**
         * Create sub-element of this.
         * @return sub-element of this
         */
        template <typename name_t>
        Element addTag(name_t&& name) { return Element(*this, std::forward<name_t>(name)); }

        /**
         * Close this element.
         * @return parent of this element or invalid element which can only be delete if this represents the root element
         */
        Element& end();

    private:
        void writeOpening();    /// called only by constructors, write element opening and set this as current

        void writeClosing();    /// called by destructor, write element closing and set parent of this as current

        void disallowAttributes(); /// set attributesStillAlowed to false, and put '>' in out if necessary

        void ensureIsCurrent(); /// throw excpetion if this is not current element
    };

private:

    /// Output, destination stream
    std::ostream& out;

    void appendStr(const std::string& s) { out.write(s.data(), s.size()); }

    void appendStrQuoted(const char* s);

    void appendStrQuoted(const std::string& s) { appendStrQuoted(s.c_str()); }

    /// Current element.
    Element* current;

public:

    /**
     * Construct XML writer which will write content to given @p out stream.
     * @param out ouptut stream which will be used as destination to XML content
     */
    XMLWriter(std::ostream& out): out(out), current(0) {}

    ~XMLWriter() { assert(current == 0); }

    /// Append to stream XML document header.
    void writeHeader() {
        out << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;
    }

    /**
     * Append element/tag to stream.
     * @param name tag name
     * @return object which allow to set details of element (attributes, content, etc.)
     */
    template <typename name_t>
    Element addElement(name_t&& name) { return Element(*this, std::forward<name_t>(name)); }

    /**
     * Append element/tag to stream.
     * @param name tag name
     * @return object which allow to set details of element (attributes, content, etc.)
     */
    template <typename name_t>
    Element addTag(name_t&& name) { return Element(*this, std::forward<name_t>(name)); }

    /**
     * Write text content for the current element
     * \param text text to write
     */
    void writeText(const std::string& text) {
        if (!current) throw XMLWriterException("No tag is open");
        current->writeText(text);
    }

    /**
     * Write CDATA for the current element
     * \param cdata data to write
     */
    void writeCDATA(const std::string& cdata) {
        if (!current) throw XMLWriterException("No tag is open");
        current->writeCDATA(cdata);
    }
};

}   // namespace plask

#endif // PLASK__UTILS_XMLWRITER_H
