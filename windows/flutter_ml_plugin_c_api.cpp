#include "include/flutter_ml/flutter_ml_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "flutter_ml_plugin.h"

void FlutterMlPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  flutter_ml::FlutterMlPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
