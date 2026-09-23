import 'dart:io';

void main() {
  final directory = Directory('lib');
  final outputFile = File('all_code.txt');
  final sink = outputFile.openWrite();

  final files = directory.listSync(recursive: true);

  for (var entity in files) {
    if (entity is File && entity.path.endsWith('.dart')) {
      // Generierte Dateien ignorieren
      if (entity.path.endsWith('.g.dart') || entity.path.endsWith('.freezed.dart')) {
        continue;
      }

      sink.writeln('\n--- Dateiname: ${entity.path} ---');
      sink.writeln(entity.readAsStringSync());
    }
  }

  sink.close();
  print('Fertig! Alle Dateien wurden in all_code.txt zusammengefasst.');
}