#ifndef PLASK__UTILS_XMLWRITER_H
#define PLASK__UTILS_XMLWRITER_H

#include <string>
#include <iostream>
#include <cstdio>

#include "exceptions.h"
#include "../format.h"

namespace plask {

/**
 * Object of this class help produce valid XML documents.
 *
 * It holds std output stream and helps format valid XML data written to this stream.
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
     * Base class for output (stream).
     */
    struct Output {

        /**
         * Write @n bytes from @b buffer.
         * @param buffer buffer with data with size @p n or more
         * @param n number of bytes to write
         */
        virtual void write(const char* buffer, std::size_t n) = 0;

        /**
         * Write @p n - 1 bytes from @p str.
         *
         * Usefull to writing const char[] literals.
         * @param str string, typicaly const char[] literal
         */
        template <int n>
        void puts(const char (&str)[n]) { write(str, n-1); }

        /**
         * Write one character @p c.
         *
         * Default implementation calls: <code>write(&c, 1);</code>
         * @param c character to write
         */
        virtual void put(char c) { write(&c, 1); }

        /**
         * Write new line character (end-line) to this output.
         */
        void newline() { put('\n'); }

    };

    /**
     * Represent single XML element connected with writer.
     *
     * Constructor put in stream opening tag and destructor closing one.
     */
    class Element {

        /// Tag name.
        std::string name;

        /// XML writer
        XMLWriter* writer;

        /// Parent element.
        Element* parent;

        /// @c true only if this tag is open and allow to append attributes
        bool attributesStillAlowed;

     public:

        /**
         * Construct element with given @p name, write to steam opening of element tag.
         * @param writer XML writer where element should be append
         * @param name name of elements tag
         */
        Element(XMLWriter& writer, const std::string& name);

        /**
         * Construct element with given @p name, write to steam opening of element tag.
         * @param writer XML writer where element should be append
         * @param name name of elements tag
         */
        Element(XMLWriter& writer, std::string&& name);

        /**
         * Construct element with given @p name, write to steam opening of element tag.
         * @param parent parent element, must by recently added, not closed one
         * @param name name of elements tag
         */
        Element(Element& parent, const std::string& name);

        /**
         * Construct element with given @p name, write to steam opening of element tag.
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
            return writeText(str(value));
        }

        Element& writeCDATA(const std::string& str);

        /**
         * Write spaces to the current indentation level
         */
        void indent ();

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

        /**
         * Check if tag attributes still can be appended
         * @return @c true if attributes can still be append to this element
         */
        bool canAppendAttributes() const { return this->attributesStillAlowed; }

        /**
         * Check if this is current element.
         * @return @c true only if this is current element
         */
        bool isCurrent() const { return writer->current == this; }

        /**
         * Check if this was ended or moved, and can't be used any more
         * @return @c true only if this was ended or moved
         */
        bool isEnded() const { return writer != 0; }

        /**
         * Get name of this element.
         * @return name of this
         */
        const std::string& getName() const { return name; }

        /**
         * Get writter used by this.
         * @return writter used by this
         */
        XMLWriter* getWriter() const { return writer; }

        /**
         * Get parent of this element.
         * @return parent of this element, @c nullptr if this represent the root
         */
        Element* getParent() const { return parent; }

    private:
        void writeOpening();    ///< called only by constructors, write element opening and set this as current

        void writeClosing();    ///< called by destructor, write element closing and set parent of this as current

        void disallowAttributes(); ///< set attributesStillAlowed to false, and put '>' in out if necessary

        void ensureIsCurrent(); ///< throw exception if this is not current element
    };

private:

    /// Output, destination stream
    Output* out;

    void appendStr(const std::string& s) { out->write(s.data(), s.size()); }

    void appendStrQuoted(const char* s);

    void appendStrQuoted(const std::string& s) { appendStrQuoted(s.c_str()); }

    /// Current element.
    Element* current;

    /// Indentation for each tag level
    std::size_t indentation;

public:

    /**
     * Construct XML writer which will write content to given @p out stream.
     * @param out ouptut stream which will be used as destination to XML content, will be not closed by this writer
     * @param indentation indentation for each tag level
     */
    XMLWriter(std::ostream& out, std::size_t indentation=2);

    /**
     * Construct XML writer which will write content to file with given name.
     * @param file_name name of file which will be used as destination to XML content
     * @param indentation indentation for each tag level
     */
    XMLWriter(const std::string& file_name, std::size_t indentation=2);

    /**
     * Construct XML writer which will write content to given C file.
     *
     * Writter will not close given descriptor.
     * @param cfile opened, C file descriptor, writer will not close it
     * @param indentation indentation for each tag level
     */
    XMLWriter(std::FILE* cfile, std::size_t indentation=2);

    /**
     * Construct XML writer which will write content to given @p out.
     * @param out ouptut which will be used as destination to XML content, will be delete (with @c delete operator) by writer destructor
     * @param indentation indentation for each tag level
     */
    XMLWriter(Output* out, std::size_t indentation=2);

    /// Delete output object.
    ~XMLWriter() { delete out; assert(current == 0); }

    /// Append to stream XML document header.
    void writeHeader() {
        out->puts("<?xml version=\"1.0\" encoding=\"utf-8\"?>");
        out->newline();
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
    template <typename text_t>
    void writeText(text_t&& text) {
        if (!current) throw XMLWriterException("No tag is open");
        current->writeText(std::forward<text_t>(text));
    }

    /**
     * Write CDATA for the current element
     * \param cdata data to write
     */
    template <typename text_t>
    void writeCDATA(text_t&& cdata) {
        if (!current) throw XMLWriterException("No tag is open");
        current->writeCDATA(std::forward<text_t>(cdata));
    }

    /**
     * Write spaces to the current indentation level
     */
    void indent () {
        if (current) current->indent();
    }

    /**
     * Get current element.
     * @return current elment or @p nullptr if there are no elements (root was not open jet or was already closed)
     */
    Element* getCurrent() {
        return current;
    }

};

/// Easier access to XML element type
typedef XMLWriter::Element XMLElement;

}   // namespace plask

#endif // PLASK__UTILS_XMLWRITER_H
