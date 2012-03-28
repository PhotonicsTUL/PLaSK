#ifndef PLASK__LOG_LOG_H
#define PLASK__LOG_LOG_H

namespace plask {

enum LogLevel {
    INFO, WARNING, DEBUG, CHART
};

template<typename ...Args>
void log(LogLevel level, Args&&... params);
    // format(std::forward<Args>(params)...);

}   // namespace plask


#endif // LOG_H
