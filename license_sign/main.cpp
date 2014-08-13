#include "license.h"

#include <sstream>
#include <fstream>

#include "getmac.h"

void print_macs() {
    std::cout << std::endl << "Note: Detected mac adresses:" << std::endl;
    for (auto& m: plask::getMacs()) std::cout << ' ' << plask::macToString(m) << std::endl;
}

int main(int argc, char *argv[]) {

    try {

        if (argc != 2) {
            std::cout << "Usage: " << argv[0] << " file_to_sign.xml" << std::endl;
            print_macs();
            return 1;
        }

        std::stringstream dst_ss;
        {
            plask::XMLReader src(argv[1]);
            plask::XMLWriter dst(dst_ss);

            if (plask::processLicense(src, &dst))
                return 0; //file already has proper signature
        }

        std::ofstream out(argv[1]);
        std::string to_write = dst_ss.str();
        out.write(to_write.data(), to_write.size());

    } catch (std::exception& e) {
        std::cerr << e.what();
        return 2;
    }

    print_macs();
    return 0;
}
