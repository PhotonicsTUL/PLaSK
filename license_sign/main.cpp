#include "license.h"

#include <sstream>
#include <fstream>

int main(int argc, char *argv[]) {
    try {

        if (argc != 2) {
            std::cout << "Usage: " << argv[0] << " file_to_sign.xml" << std::endl;
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

    return 0;
}
