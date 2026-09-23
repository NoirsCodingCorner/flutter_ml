import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import '../test/gpuFunctions_test.dart' as gpu_tests;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  gpu_tests.main();
}