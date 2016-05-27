/*
 * Author: Dyinnz.HUST.Hustunique
 * Date: 2015-10-31
 * Email: ml_143@sina.com
 * Github: https://github.com/dyinnz
 * Description:
 *    A simple logger written in C++
 */

#pragma once

#include <cstdio>
#include <cstdarg>

#define DEBUG

namespace dy_logger {

class Logger {
  public:
    enum Level {
      kDebug     = 0,
      kLog       = 1,
      kError     = 4,
      };

    static constexpr int kMaxLevel = 3;

    Logger() {
      _fps[kDebug] = stdout;
      _fps[kLog]   = stdout;
      _fps[kError] = stdout;
    }
    ~Logger() {}

  public:
    void debug(const char *format, ...) {
#ifdef DEBUG
      if (kDebug >= _filter_level) {
        fprintf(_fps[kDebug], "[debug] ");
        va_list va;
        va_start(va, format);
        vfprintf(_fps[kDebug], format, va);
        va_end(va);
      }
#endif
    }

    void log(const char *format, ...) {
      if (kLog >= _filter_level) {
        fprintf(_fps[kLog], "[log] ");
        va_list va;
        va_start(va, format);
        vfprintf(_fps[kLog], format, va);
        va_end(va);
      }
    }

    void error(const char *format, ...) {
      if (kError >= _filter_level) {
        fprintf(_fps[kError], "[error] ");
        va_list va;
        va_start(va, format);
        vfprintf(_fps[kError], format, va);
        va_end(va);
      }
    }

    void set_filter_level(Level filter) { _filter_level = filter; }
    Level filter_level() { return _filter_level; }

    void set_level_fp(Level level, FILE *fp) { _fps[level] = fp; }
    FILE *level_fp(Level level) { return _fps[level]; }

  private:
    FILE *_fps[kMaxLevel];
    Level _filter_level {kLog};

  private: 
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};

#define func_debug(__logger, __format, ...) do { \
  __logger.debug("%s():" __format, __func__, ##__VA_ARGS__); \
} while (false);

} // end of namespace
