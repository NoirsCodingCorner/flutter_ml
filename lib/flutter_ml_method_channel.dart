import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'flutter_ml_platform_interface.dart';

/// An implementation of [FlutterMlPlatform] that uses method channels.
class MethodChannelFlutterMl extends FlutterMlPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('flutter_ml');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
