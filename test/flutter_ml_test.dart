import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_ml/flutter_ml.dart';
import 'package:flutter_ml/flutter_ml_platform_interface.dart';
import 'package:flutter_ml/flutter_ml_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockFlutterMlPlatform
    with MockPlatformInterfaceMixin
    implements FlutterMlPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final FlutterMlPlatform initialPlatform = FlutterMlPlatform.instance;

  test('$MethodChannelFlutterMl is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelFlutterMl>());
  });

  test('getPlatformVersion', () async {
    FlutterMl flutterMlPlugin = FlutterMl();
    MockFlutterMlPlatform fakePlatform = MockFlutterMlPlatform();
    FlutterMlPlatform.instance = fakePlatform;

    expect(await flutterMlPlugin.getPlatformVersion(), '42');
  });
}
