#include <cstdio>
using namespace std;

#include "log.h"

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#    include <windows.h>
#else
#   include <unistd.h>
#endif

namespace plask {

#ifdef NDEBUG
LogLevel maxLoglevel = LOG_DETAIL;
#else
LogLevel maxLoglevel = LOG_DEBUG;
#endif

bool forcedLoglevel = false;

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)

    /// Class writing colorful log into stderr
    class StderrLogger: public Logger {
        static const HANDLE hstderr;
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        unsigned short BACKGROUND, DEFAULT;

        enum Colors: unsigned short
        {
            BLACK = 0,
            BLUE = 1,
            GREEN = 2,
            CYAN = 3,
            RED = 4,
            MAGENTA = 5,
            BROWN = 6,
            WHITE = 7,
            GRAY = 8,
            BRIGHT_BLUE = 9,
            BRIGHT_GREEN = 10,
            BRIGHT_CYAN = 11,
            BRIGHT_RED = 12,
            BRIGHT_MAGENTA = 13,
            YELLOW = 14,
            BRIGHT_WHITE = 15
        };

        inline void color(Colors fg) {
            SetConsoleTextAttribute(hstderr, BACKGROUND | fg);
        }

        const char* head(LogLevel level) {
            switch (level) {
                case LOG_ERROR:         color(BRIGHT_RED); return "ERROR         ";
                case LOG_CRITICAL_ERROR:color(BRIGHT_RED); return "CRITICAL ERROR";
                case LOG_WARNING:       color(BROWN); return "WARNING       ";
                case LOG_INFO:          color(BRIGHT_CYAN); return "INFO          ";
                case LOG_RESULT:        color(GREEN); return "RESULT        ";
                case LOG_DATA:          color(CYAN); return "DATA          ";
                case LOG_DETAIL:        color((Colors)DEFAULT); return "DETAIL        ";
                case LOG_ERROR_DETAIL:  color(RED); return "ERROR DETAIL  ";
                case LOG_DEBUG:         color(GRAY); return "DEBUG         ";
            }
            return "UNSPECIFIED   "; // mostly to silence compiler warning than to use in the real life
        }

      public:

        StderrLogger() {
            GetConsoleScreenBufferInfo(hstderr, &csbi);
            BACKGROUND = csbi.wAttributes & 0xF0;
            DEFAULT = csbi.wAttributes & 0x0F;
        }

        virtual void writelog(LogLevel level, const std::string& msg) {
            fprintf(stderr, "%s: %s\n", head(level), msg.c_str());
            SetConsoleTextAttribute(hstderr, csbi.wAttributes);
        }
    };
    const HANDLE StderrLogger::hstderr = GetStdHandle(STD_ERROR_HANDLE);

#else

    /// Class writing colorful log into stderr
    class StderrLogger: public Logger {

        #define DEFAULT "\033[00m"
        #define BLACK   "\033[30m"
        #define RED     "\033[31m"
        #define GREEN   "\033[32m"
        #define BROWN  "\033[33m"
        #define BLUE    "\033[34m"
        #define MAGENTA "\033[35m"
        #define CYAN    "\033[36m"
        #define WHITE   "\033[37m"
        #define GRAY   "\033[30;01m"
        #define BRIGHT_RED     "\033[31;01m"
        #define BRIGHT_GREEN   "\033[32;01m"
        #define YELLOW  "\033[33;01m"
        #define BRIGHT_BLUE    "\033[34;01m"
        #define BRIGHT_MAGENTA "\033[35;01m"
        #define BRIGHT_CYAN    "\033[36;01m"
        #define BRIGHT_WHITE   "\033[37;01m"

        static const bool tty;

        static const char* head(LogLevel level) {
            if (tty) {
                switch (level) {
                    case LOG_CRITICAL_ERROR:return BRIGHT_RED "CRITICAL ERROR";
                    case LOG_ERROR:         return BRIGHT_RED "ERROR         ";
                    case LOG_WARNING:       return BROWN "WARNING       ";
                    case LOG_INFO:          return BRIGHT_BLUE "INFO          ";
                    case LOG_RESULT:        return GREEN "RESULT        ";
                    case LOG_DATA:          return CYAN "DATA          ";
                    case LOG_DETAIL:        return DEFAULT "DETAIL        ";
                    case LOG_ERROR_DETAIL:  return RED "ERROR DETAIL  ";
                    case LOG_DEBUG:         return GRAY "DEBUG         ";
                }
            } else {
                switch (level) {
                    case LOG_ERROR:         return "ERROR         ";
                    case LOG_CRITICAL_ERROR:return "CRITICAL ERROR";
                    case LOG_WARNING:       return "WARNING       ";
                    case LOG_INFO:          return "INFO          ";
                    case LOG_RESULT:        return "RESULT        ";
                    case LOG_DATA:          return "DATA          ";
                    case LOG_DETAIL:        return "DETAIL        ";
                    case LOG_ERROR_DETAIL:  return "ERROR DETAIL  ";
                    case LOG_DEBUG:         return "DEBUG         ";
                }
            }
            return "UNSPECIFIED   "; // mostly to silence compiler warning than to use in the real life
        }

      public:

        virtual void writelog(LogLevel level, const std::string& msg) {
            fprintf(stderr, "%s: %s%s\n", head(level), msg.c_str(), tty? DEFAULT : "");
        }
    };
    const bool StderrLogger::tty = isatty(fileno(stderr));

#endif

shared_ptr<Logger> default_logger { new StderrLogger() };

void writelog(LogLevel level, const std::string& msg) {
    if (level <= maxLoglevel) {
        #pragma omp critical(writelog)
        default_logger->writelog(level, msg);
    }
}

} // namespace plask
