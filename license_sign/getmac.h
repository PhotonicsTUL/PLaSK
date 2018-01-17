#ifndef PLASK_GETMAC_H
#define PLASK_GETMAC_H

// This file should not be included in any plask .h file because it can be not distributed to solvers' developers.

#include <vector>
#include <string>
#include <array>

namespace plask {

typedef std::array<unsigned char, 6> mac_address_t;

}   // namespace plask

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)

//#define WIN32_LEAN_AND_MEAN
#include <plask/utils/minimal_windows.h>
#include <intrin.h>
#include <iphlpapi.h>

namespace plask {

//http://stackoverflow.com/questions/16858782/how-to-obtain-almost-unique-system-identifier-in-a-cross-platform-way
//http://msdn.microsoft.com/en-us/library/windows/desktop/aa366062%28v=vs.85%29.aspx
inline std::vector<mac_address_t> getMacs() {
    std::vector<mac_address_t> result;

    IP_ADAPTER_INFO AdapterInfo[32];
    DWORD dwBufLen = sizeof( AdapterInfo );

    DWORD dwStatus = GetAdaptersInfo( AdapterInfo, &dwBufLen );
    if ( dwStatus != ERROR_SUCCESS )
        return result; // no adapters.

    PIP_ADAPTER_INFO pAdapterInfo = AdapterInfo;
    while (pAdapterInfo) {
        if (pAdapterInfo->Type == MIB_IF_TYPE_ETHERNET && pAdapterInfo->AddressLength == 6) {
            result.emplace_back();
            memcpy(result.back().data(), pAdapterInfo->Address, 6);
        }
        pAdapterInfo = pAdapterInfo->Next;
    }

    return result;
}

}   // namespace plask

#else   // Linux code
//TODO, MACOS: http://oroboro.com/unique-machine-fingerprint/

#include <ifaddrs.h>
#include <net/if.h>
#include <netpacket/packet.h>

namespace plask {

inline std::vector<mac_address_t> getMacs() {
    std::vector<mac_address_t> result;

    //Code comes from:
    //http://stackoverflow.com/questicons/1779715/how-to-get-mac-address-of-your-machine-using-a-c-program
    //http://stackoverflow.com/questions/16858782/how-to-obtain-almost-unique-system-identifier-in-a-cross-platform-way
    ifaddrs* ifaddr = nullptr;

    if (getifaddrs(&ifaddr) != -1) {
         for (ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
             if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_PACKET && !(ifa->ifa_flags & IFF_LOOPBACK)) {
                  struct sockaddr_ll *s = (struct sockaddr_ll*)ifa->ifa_addr;
                  result.emplace_back();
                  memcpy(result.back().data(), s->sll_addr+s->sll_halen-6, 6);
             }
         }
         freeifaddrs(ifaddr);
    }
    return result;
}

}   // namespace plask

#endif  // Linux code

namespace plask {

inline std::string macToString(const mac_address_t& mac, bool colons=false) {
    std::string res;
    res.reserve(2*6 + 5);
    for (unsigned char c: mac) {
        const char* to_hex = "0123456789ABCDEF";
        res += to_hex[c / 16]; //upper half of byte
        res += to_hex[c % 16]; //lower half of byte
        if (colons && res.size() != 2*6 + 5) res += ':';
    }
    return res;
}

inline unsigned char fromHex(char c, const std::string& mac_str) {
    if ('0' <= c && c <= '9') return c - '0';
    if ('A' <= c && c <= 'F') return c - 'A' + 10;
    if ('a' <= c && c <= 'f') return c - 'a' + 10;
    throw std::invalid_argument("\"" + mac_str + "\" is not well-formated mac address. It includes invalid character where hex digit is expected.");
}

inline mac_address_t macFromString(const std::string& mac_str) {
    mac_address_t res;
    int parsed = 0;
    for (std::size_t i = 0; i < mac_str.size(); ++i) {
        const char c = mac_str[i];
        if (std::isspace(c)) continue;
        if (parsed == 6) throw std::invalid_argument("\"" + mac_str + "\" is not well-formated mac address.");
        if (c == ':') continue;
        res[parsed] = fromHex(c, mac_str);
        ++i;
        if (i == mac_str.size()) throw std::invalid_argument("\"" + mac_str + "\" is not well-formated mac address (unexpected end).");
        res[parsed] = (res[parsed] << 4) | fromHex(mac_str[i], mac_str);
        ++parsed;
    }
    if (parsed != 6) throw std::invalid_argument("\"" + mac_str + "\" is not well-formated mac address (unexpected end).");
    return res;
}

}


#endif // PLASK_GETMAC_H
