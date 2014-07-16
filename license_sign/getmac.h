#ifndef PLASK_GETMAC_H
#define PLASK_GETMAC_H

#include <vector>
#include <string>
#include <array>

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>
#include <iphlpapi.h>

//http://stackoverflow.com/questions/16858782/how-to-obtain-almost-unique-system-identifier-in-a-cross-platform-way
//http://msdn.microsoft.com/en-us/library/windows/desktop/aa366062%28v=vs.85%29.aspx
std::vector<std::array<unsigned char, 6>> getMacs() {
    std::vector<std::array<unsigned char, 6>> result;

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


#else   // Linux code
//TODO, MACOS: http://oroboro.com/unique-machine-fingerprint/

#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#include <netinet/in.h>
#include <string.h>

std::vector<std::array<unsigned char, 6>> getMacs() {
    std::vector<std::array<unsigned char, 6>> result;

    //Code comes from:
    //http://stackoverflow.com/questions/1779715/how-to-get-mac-address-of-your-machine-using-a-c-program
    //http://stackoverflow.com/questions/16858782/how-to-obtain-almost-unique-system-identifier-in-a-cross-platform-way
    struct ifreq ifr;
    struct ifconf ifc;
    char buf[256 * sizeof(struct ifreq)];

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    //if (sock == -1) { /handle error };

    ifc.ifc_len = sizeof(buf);
    ifc.ifc_buf = buf;
    if (ioctl(sock, SIOCGIFCONF, &ifc) == -1) {
        //handle error
    }

    struct ifreq* it = ifc.ifc_req;
    const struct ifreq* const end = it + (ifc.ifc_len / sizeof(struct ifreq));

    for (; it != end; ++it) {
        strcpy(ifr.ifr_name, it->ifr_name);
        if (ioctl(sock, SIOCGIFFLAGS, &ifr) == 0) {
            if (! (ifr.ifr_flags & IFF_LOOPBACK)) { // don't count loopback
                if (ioctl(sock, SIOCGIFHWADDR, &ifr) == 0) {
                    result.emplace_back();
                    memcpy(result.back().data(), ifr.ifr_hwaddr.sa_data, 6);
                }
            }
        }
       // else {  handle error }
    }
    close( sock );

    return result;
}

#endif  // Linux code

std::string macToString(const std::array<unsigned char, 6>& mac) {
    std::string res;
    res.reserve(2*6 + 5);
    for (unsigned char c: mac) {
        const char* to_hex = "0123456789ABCDEF";
        res += to_hex[c / 16]; //upper half of byte
        res += to_hex[c % 16]; //lower half of byte
        if (res.size() != 2*6 + 5) res += ':';
    }
    return res;
}


#endif // PLASK_GETMAC_H
