/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "writer.hpp"

#include "exceptions.hpp"

#include <fstream>

namespace plask {

template <typename ostream_t>
struct OStreamOutput: public XMLWriter::Output {

    ostream_t ostream;

    OStreamOutput(ostream_t& stream_ref): ostream(stream_ref) {
    }
    
    template <typename... StreamCtorArgT>
    OStreamOutput(StreamCtorArgT&&... stream_open_arg) {
        ostream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        ostream.open(std::forward<StreamCtorArgT>(stream_open_arg)...);
    }

    void write(const char *buffer, std::size_t n) override {
        ostream.write(buffer, n);
    }

    void put(char c) override {
        ostream.put(c);
    }

};

typedef OStreamOutput<std::ostream&> OStreamRef;
typedef OStreamOutput<std::ofstream> OFStream;

struct CFileOutput: public XMLWriter::Output {

    FILE* file;

    CFileOutput(FILE* file): file(file) {}

    void write(const char *buffer, std::size_t n) override {
        if (std::fwrite(buffer, 1, n, file) != n)
            throw XMLWriterException("XML writter can't write to C file descriptor.");
    }

    /*void put(char c) {
        std::fputc(c, file);
    }*/

};

XMLWriter::Element::Element(XMLWriter &writer, const std::string &name)
: name(name), writer(&writer), hasChildren(false) {
    writeOpening();
}

XMLWriter::Element::Element(XMLWriter &writer, std::string &&name)
: name(std::move(name)), writer(&writer), hasChildren(false) {
    writeOpening();
}

XMLWriter::Element::Element(XMLWriter::Element &parent, const std::string &name)
: name(name), writer(parent.writer), hasChildren(false) {
    parent.ensureIsCurrent();
    writeOpening();
}

XMLWriter::Element::Element(XMLWriter::Element &parent, std::string&& name)
: name(std::move(name)), writer(parent.writer), hasChildren(false) {
    parent.ensureIsCurrent();
    writeOpening();
}

XMLWriter::Element::Element(XMLWriter::Element&& to_move) {
    this->operator =(std::move(to_move));
/*    to_move.ensureIsCurrent();
    name = std::move(to_move.name);
    writer = to_move.writer;
    to_move.writer = 0;
    parent = to_move.parent;
    attributesStillAlowed = to_move.attributesStillAlowed;
    this->writer->current = this;*/
}

XMLWriter::Element &XMLWriter::Element::operator=(XMLWriter::Element && to_move) {
    to_move.ensureIsCurrent();
    name = std::move(to_move.name);
    writer = to_move.writer;
    parent = to_move.parent;
    attributesStillAlowed = to_move.attributesStillAlowed;
    hasChildren = to_move.hasChildren;
    to_move.writer = 0;
    this->writer->current = this;
    return *this;
}

XMLWriter::Element::~Element() {
    if (!writer) return;    // element already moved
    writeClosing();
}

std::size_t XMLWriter::Element::getLevel() const {
    std::size_t result = 0;
    for (Element* i = this->parent; i != 0; i = i->parent) ++result;
    return result;
}

XMLWriter::Element &XMLWriter::Element::attr(const std::string &attr_name, const std::string &attr_value) {
    if (!attributesStillAlowed)
        throw XMLWriterException(format("Can't append attribute \"{1}\" to \"{0}\" XML element because this element has already non-empty content.", name, attr_name));
    writer->out->put(' ');
    writer->appendStr(attr_name);
    writer->out->puts("=\"");
    writer->appendStrQuoted(attr_value);
    writer->out->put('"');
    return *this;
}

XMLWriter::Element &XMLWriter::Element::writeText(const char *str) {
    ensureIsCurrent();
    disallowAttributes();
    writer->appendStrQuoted(str);
    return *this;
}

XMLWriter::Element &XMLWriter::Element::writeCDATA(const std::string& str) {
    ensureIsCurrent();
    disallowAttributes();
    writer->out->puts("<![CDATA[");
    writer->appendStr(str);
    writer->out->puts("]]>");
    return *this;
}

void XMLWriter::Element::indent () {
    if (disallowAttributes()) writer->out->newline();
    std::size_t l = (getLevel() + 1) * writer->indentation;
    while (l > 0) { writer->out->put(' '); --l; }
}

XMLWriter::Element &XMLWriter::Element::end() {
    ensureIsCurrent();
    writeClosing();
    Element* current = writer->current; //new current tag, parent of this
    this->writer = 0;   // to not close a tag by destructor
    return current ? *current : *this;
}

void XMLWriter::Element::writeOpening() {
    attributesStillAlowed = true;
    parent = writer->current;
    if (writer->current) {
        writer->current->hasChildren = true;
        if (writer->current->disallowAttributes())  //parent has nothing inside?
            writer->out->newline();
    }
    writer->current = this;
    std::size_t l = getLevel() * writer->indentation;
    while (l > 0) { writer->out->put(' '); --l; }
    writer->out->put('<');
    writer->appendStr(name);
}

void XMLWriter::Element::writeClosing()
{
    if (attributesStillAlowed) {   //empty tag?
        writer->out->puts("/>");
    } else {
        if (hasChildren) {
            std::size_t l = getLevel() * writer->indentation;
            while (l > 0) { writer->out->put(' '); --l; }
        }
        writer->out->puts("</");
        writer->appendStr(name);
        writer->out->put('>');
    }
    writer->out->newline();   //TODO ??
    writer->current = this->parent;
}

bool XMLWriter::Element::disallowAttributes() {
    if (attributesStillAlowed) {
        writer->out->put('>');
        //writer->out->newline();   //TODO ??
        writer->current->attributesStillAlowed = false;
        return true;
    } else
        return false;
}

void XMLWriter::Element::ensureIsCurrent() {
    if (this != writer->current)
        throw XMLWriterException("Operation is not permitted as the XML element \""+ name +"\" is not the last one in the stack");
}

XMLWriter::XMLWriter(std::ostream& out, std::size_t indentation)
    : out(new OStreamRef(out)), current(0), indentation(indentation) {}

XMLWriter::XMLWriter(const std::string& file_name, std::size_t indentation)
    : out(new OFStream(file_name.c_str())), current(0), indentation(indentation)
{}

XMLWriter::XMLWriter(std::FILE* cfile, std::size_t indentation)
    : out(new CFileOutput(cfile)), current(0), indentation(indentation)
{}

XMLWriter::XMLWriter(Output* out, std::size_t indentation)
    : out(out), current(0), indentation(indentation)
{}

void XMLWriter::appendStrQuoted(const char *s) {
    for (; *s; ++s)
        switch (*s) {
            case '"': out->puts("&quot;"); break;
            case '<': out->puts("&lt;"); break;
            case '>': out->puts("&gt;"); break;
            case '&': out->puts("&amp;"); break;
            case '\'': out->puts("&apos;"); break;
            default: out->put(*s); break;
        }
}

}   // namespace plask
