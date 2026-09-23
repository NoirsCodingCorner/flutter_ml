import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'flutter_ml_method_channel.dart';

abstract class FlutterMlPlatform extends PlatformInterface {
  /// Constructs a FlutterMlPlatform.
  FlutterMlPlatform() : super(token: _token);

  static final Object _token = Object();

  static FlutterMlPlatform _instance = MethodChannelFlutterMl();

  /// The default instance of [FlutterMlPlatform] to use.
  ///
  /// Defaults to [MethodChannelFlutterMl].
  static FlutterMlPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [FlutterMlPlatform] when
  /// they register themselves.
  static set instance(FlutterMlPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
