
## `Logger` Utility

This is a simple, static utility class for printing color-coded messages to the standard console output (`stdout`).

### How It Works

The class is entirely static, meaning you do not need to create an instance of `Logger` to use it. You can call its methods directly.

It uses ANSI color codes to change the text color in the terminal.

### Core Method

* `Logger.log(String message, {String prefix = 'ℹ️', String color})`
    * This is the generic method that all other methods use.
    * It prints the `message` to the console.
    * You can optionally provide a `prefix` (which defaults to 'ℹ️') and a specific ANSI `color` code.

### Convenience Methods

To make logging easier, the class provides several shortcuts that call `Logger.log` with a pre-defined color:

* **`Logger.blue(String message, {String prefix = 'ℹ️'})`**

* **`Logger.green(String message, {String prefix = 'ℹ️'})`**

* **`Logger.yellow(String message, {String prefix = 'ℹ️'})`**

* **`Logger.red(String message, {String prefix = 'ℹ️'})`**

* **`Logger.cyan(String message, {String prefix = 'ℹ️'})`**

### Example Usage

```dart
// Import the logger (assuming it's in a 'utils' folder)
import '.../utils/logger.dart';

void main() {
  Logger.green('Operation successful.', prefix: '✅');
  Logger.yellow('Warning: Low memory.', prefix: '⚠️');
  Logger.red('Critical error occurred!', prefix: '🔥');
  Logger.blue('Starting background process...');
}
```