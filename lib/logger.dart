// logger.dart
import 'dart:io';

class Logger {
  // ANSI color codes
  static const _reset  = '\x1B[0m';
  static const _blue   = '\x1B[34m';
  static const _green  = '\x1B[32m';
  static const _yellow = '\x1B[33m';
  static const _red    = '\x1B[31m';
  static const _cyan   = '\x1B[36m';

  /// Generic logger. You can always call Logger.log(...) if you need a custom color.
  static void log(
      String message, {
        String prefix = 'ℹ️',
        String color  = _cyan,
      }) {
    // The timestamp has been removed.
    stdout.writeln('$color$prefix $message$_reset');
  }

  /// Convenience methods for each color:
  static void blue(String message, {String prefix = 'ℹ️'}) {
    log(message, prefix: prefix, color: _blue);
  }

  static void green(String message, {String prefix = 'ℹ️'}) {
    log(message, prefix: prefix, color: _green);
  }

  static void yellow(String message, {String prefix = 'ℹ️'}) {
    log(message, prefix: prefix, color: _yellow);
  }

  static void red(String message, {String prefix = 'ℹ️'}) {
    log(message, prefix: prefix, color: _red);
  }

  static void cyan(String message, {String prefix = 'ℹ️'}) {
    log(message, prefix: prefix, color: _cyan);
  }
}