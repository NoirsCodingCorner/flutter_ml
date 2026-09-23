#ifndef FLUTTER_PLUGIN_FLUTTER_ML_PLUGIN_H_
#define FLUTTER_PLUGIN_FLUTTER_ML_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace flutter_ml {

class FlutterMlPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar);

  FlutterMlPlugin();

  virtual ~FlutterMlPlugin();

  // Disallow copy and assign.
  FlutterMlPlugin(const FlutterMlPlugin&) = delete;
  FlutterMlPlugin& operator=(const FlutterMlPlugin&) = delete;

  // Called when a method is called on this plugin's channel from Dart.
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue> &method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace flutter_ml

#endif  // FLUTTER_PLUGIN_FLUTTER_ML_PLUGIN_H_
