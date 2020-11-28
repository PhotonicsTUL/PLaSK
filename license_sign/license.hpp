#ifndef PLASK_LICENSE_SIGN__LICENSE_H
#define PLASK_LICENSE_SIGN__LICENSE_H

// This file should not be included in any plask .h file because it can be not distributed to solvers' developers.

#include "../plask/exceptions.hpp"
#include "../plask/utils/xml.hpp"
#include "../plask/utils/warnings.hpp"

#include "sha1.hpp"

namespace plask {

#define PLASK_LICENSE_SIGNATURE_TAG_NAME "signature"

/**
 * Read data from license without verifying it.
 *
 * @param src source license XML
 * @param content_cb callback called for each content node (with @p src argument), do nothing by default
 */
inline static void readLicenseData(XMLReader& src, XMLWriter* PLASK_UNUSED(dst),
                                   std::function<void (XMLReader& src)> content_cb = [](XMLReader&){}) {
    while (src.next())
        if (src.getNodeType() == XMLReader::NODE_TEXT) content_cb(src);
}

/**
 * Check if license has valid signature or append signature to license XML.
 *
 * Only inteprete @c signature tag (and only if it has level 1 - is child of root), content of other tags can be interpreted by @p content_cb.
 *
 * Throw exception if there are duplicated @c signature tag, @p src is ill-formated or @p content_cb throws exception.
 * @param src source license XML
 * @param dst optional destination license XML, if given all content of @p src, completed with proper signature tag, will be added to it
 * @param content_cb callback called for each content node (with @p src argument), do nothing by default
 * @return @c true only if @p src has proper signature
 */
inline static bool processLicense(XMLReader& src, XMLWriter* dst,
                                  std::function<void (XMLReader& src)> content_cb = [](XMLReader&){}) {
    plask::optional<std::string> read_signature;
    std::string calculated_signature;
    std::string to_sign = "CxoAMhusG8KNnwuBELW432yR\n";
    std::deque<XMLElement> writtenPath;    //unused if dst is nullptr
    while (src.next())
        switch (src.getNodeType()) {

            case XMLReader::NODE_ELEMENT:
                if (src.getLevel() == 2 && src.getNodeName() == PLASK_LICENSE_SIGNATURE_TAG_NAME) {
                    if (read_signature) src.throwException("duplicated <" PLASK_LICENSE_SIGNATURE_TAG_NAME "> tag in license file");
                    read_signature = src.requireTextInCurrentTag(); //this will move src to tag end, so this tag will be skiped in dst
                    break;
                }
                if (dst) writtenPath.push_back(dst->addElement(src.getNodeName()));
                to_sign += 'N';
                to_sign += src.getNodeName();
                for (auto& a : src.getAttributes()) {
                    if (dst) dst->getCurrent()->attr(a.first, a.second);
                    to_sign += 'A';
                    to_sign += a.first;
                    to_sign += '=';
                    to_sign += a.second;
                }
                break;

            case XMLReader::NODE_TEXT: {
                content_cb(src);
                std::string text = src.getTextContent();
                if (dst) dst->writeText(text);
                to_sign += 'T';
                to_sign += text;
                break;
            }

            case XMLReader::NODE_ELEMENT_END:
                if (src.getLevel() == 1) {  //end of root?
                    unsigned char hash[20];
                    sha1::calc(to_sign.data(), int(to_sign.size()), hash);
                    calculated_signature.reserve(40);
                    for (unsigned char h: hash) {
                        const char* to_hex = "0123456789ABCDEF";
                        calculated_signature += to_hex[h % 16]; //lower half of byte
                        calculated_signature += to_hex[h / 16]; //upper half of byte
                    }
                    if (dst) {
                        dst->getCurrent()->addElement(PLASK_LICENSE_SIGNATURE_TAG_NAME).writeText(calculated_signature);
                    }
                }
                if (dst) writtenPath.pop_back();
                break;
        }

    return read_signature ? *read_signature == calculated_signature : false;
}

}   // namespace plask



#endif // PLASK_LICENSE_SIGN__LICENSE_H