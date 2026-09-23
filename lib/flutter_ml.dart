
import 'flutter_ml_platform_interface.dart';

class FlutterMl {
  Future<String?> getPlatformVersion() {
    return FlutterMlPlatform.instance.getPlatformVersion();
  }
}
