import 'dart:math' as math;
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/rendering.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]).then((_) => runApp(const MyApp()));
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'NID Detector',
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  void _openSidePicker(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.credit_card),
              title: const Text('Front side'),
              onTap: () {
                Navigator.pop(ctx);
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (_) => NidCaptureYoloViewPage(
                      title: 'Front',
                      modelAssetPath: 'assets/front_nid_model.tflite',
                      labelsAssetPath: 'assets/front_nid_labels.txt',
                      requiredLabels: [
                        'name',
                        'date_of_birth',
                        'nid_number',
                        'nid_front_image',
                      ],
                    ),
                  ),
                );
              },
            ),
            ListTile(
              leading: const Icon(Icons.credit_card_rounded),
              title: const Text('Back side'),
              onTap: () {
                Navigator.pop(ctx);
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (_) => NidCaptureYoloViewPage(
                      title:'Back',
                      modelAssetPath: 'assets/back_nid_model.tflite',
                      labelsAssetPath: 'assets/back_nid_labels.txt',
                    ),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('NID Tracker')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 280,
                child: FilledButton(
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => const NidLiveDetectPage(
                          title: 'NID Front - Live Detection',
                          modelAssetPath: 'assets/front_nid_model.tflite',
                          labelsAssetPath: 'assets/front_nid_labels.txt',
                        ),
                      ),
                    );
                  },
                  child: const Text('Open NID Front Detector'),
                ),
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: 280,
                child: OutlinedButton(
                  onPressed: () {
                    // Example for reusing with a different model/labels.
                    // Update the asset paths once you add the back model files to pubspec assets.
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => const NidLiveDetectPage(
                          title: 'NID Back - Live Detection',
                          modelAssetPath: 'assets/back_nid_model.tflite',
                          labelsAssetPath: 'assets/back_nid_labels.txt',
                        ),
                      ),
                    );
                  },
                  child: const Text('Open NID Back Detector'),
                ),
              ),
              const SizedBox(height: 24),
              SizedBox(
                width: 280,
                child: FilledButton.tonal(
                  onPressed: () => _openSidePicker(context),
                  child: const Text('Use Camera'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class NidLiveDetectPage extends StatefulWidget {
  final String title;
  final String modelAssetPath;
  final String labelsAssetPath;
  const NidLiveDetectPage({
    super.key,
    required this.title,
    required this.modelAssetPath,
    required this.labelsAssetPath,
  });

  @override
  State<NidLiveDetectPage> createState() => _NidLiveDetectPageState();
}

class _NidLiveDetectPageState extends State<NidLiveDetectPage> {
  final _controller = YOLOViewController();
  List<YOLOResult> _results = const [];
  Map<String, YOLOResult> _latestByLabel = {};
  double _zoom = 1.0;

  // Absolute model file path copied from assets
  String? _modelFilePath;

  // Labels loaded from assets/labels.txt
  List<String> _labels = const [];
  double? _fps;
  int _lastEventMs = 0;

  @override
  void initState() {
    super.initState();
    _prepareModelPath();
    _loadLabels();
    // Apply more permissive thresholds to ensure early visibility
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _controller.setThresholds(confidenceThreshold: 0.60, iouThreshold: 0.50, numItemsThreshold: 100);
    });
  }

  Future<void> _prepareModelPath() async {
    try {
      final dir = await getApplicationSupportDirectory();
      final modelsDir = Directory('${dir.path}/models');
      if (!await modelsDir.exists()) {
        await modelsDir.create(recursive: true);
      }
      final baseName = widget.modelAssetPath.split('/').last;
      final outFile = File('${modelsDir.path}/$baseName');
      // Always copy on first run; overwrite if file missing or size differs
      final data = await rootBundle.load(widget.modelAssetPath);
      if (!await outFile.exists() || (await outFile.length()) != data.lengthInBytes) {
        await outFile.writeAsBytes(data.buffer.asUint8List(), flush: true);
      }
      if (mounted) setState(() => _modelFilePath = outFile.path);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Model copy failed: $e')),
      );
    }
  }

  Future<void> _loadLabels() async {
    try {
      final txt = await rootBundle.loadString(widget.labelsAssetPath);
      final lines = txt.split(RegExp(r'\r?\n')).where((l) => l.trim().isNotEmpty).toList();
      if (mounted) {
        setState(() {
          _labels = lines;
          // Recompute latest detections mapped by label using current _results
          _latestByLabel = {
            for (final r in _results)
              if (_labels.contains(_displayName(r))) _displayName(r): r,
          };
        });
      }
    } catch (_) {
      // ignore if labels not present
    }
  }

  String _displayName(YOLOResult r) {
    final name = r.className.trim();
    final looksNumeric = RegExp(r'^\d+$').hasMatch(name);
    if (name.isEmpty || looksNumeric) {
      if (r.classIndex >= 0 && r.classIndex < _labels.length) {
        return _labels[r.classIndex];
      }
    }
    return name.isEmpty && r.classIndex >= 0 && r.classIndex < _labels.length
        ? _labels[r.classIndex]
        : (name.isEmpty ? 'class_${r.classIndex}' : name);
  }

  @override
  Widget build(BuildContext context) {
    final ready = _modelFilePath != null; // wait until model file is ready
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        actions: [
          IconButton(
            tooltip: 'Switch camera',
            icon: const Icon(Icons.cameraswitch),
            onPressed: () => _controller.switchCamera(),
          ),
        ],
      ),
      body: !ready
          ? const Center(child: Text('Preparing model…'))
          : LayoutBuilder(
        builder: (context, constraints) {
          final screenSize = Size(constraints.maxWidth, constraints.maxHeight);
          return Stack(
            fit: StackFit.expand,
            children: [
              YOLOView(
                modelPath: _modelFilePath!,
                task: YOLOTask.detect,
                controller: _controller,
                // Enable native overlay too, for quicker visual feedback
                showNativeUI: false,
                useGpu: true,
                confidenceThreshold: 0.60,
                iouThreshold: 0.50,
                streamingConfig: const YOLOStreamingConfig.minimal(),
                onResult: (List<YOLOResult> results) {
                  final now = DateTime.now().millisecondsSinceEpoch;
                  // Make a fresh copy to force painter repaint comparisons
                  final listCopy = List<YOLOResult>.from(results);
                  setState(() {
                    _lastEventMs = now;
                    _results = listCopy;
                    _latestByLabel = {
                      for (final r in listCopy)
                        if (_labels.contains(_displayName(r))) _displayName(r): r,
                    };
                  });
                },
                onStreamingData: (Map<String, dynamic> stream) {
                  // Fallback path: parse raw stream when onResult is bypassed
                  try {
                    final now = DateTime.now().millisecondsSinceEpoch;
                    final dets = (stream['detections'] as List?) ?? const [];
                    final parsed = dets.whereType<Map>().map((m) => YOLOResult.fromMap(m)).toList();
                    setState(() {
                      _lastEventMs = now;
                      _fps = (stream['fps'] is num) ? (stream['fps'] as num).toDouble() : _fps;
                      _results = parsed; // already a fresh list
                      _latestByLabel = {
                        for (final r in parsed)
                          if (_labels.contains(_displayName(r))) _displayName(r): r,
                      };
                    });
                  } catch (_) {/*ignore parse errors*/}
                },
                onPerformanceMetrics: (m) {
                  setState(() { _fps = m.fps; _lastEventMs = DateTime.now().millisecondsSinceEpoch; });
                },
                onZoomChanged: (z) => setState(() => _zoom = z),
              ),
              // Status banner (FPS / No detections yet)
              Positioned(
                top: 12,
                left: 12,
                right: 12,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _StatusBanner(fps: _fps, lastEventMs: _lastEventMs),
                    const SizedBox(height: 6),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.45),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text('Detections: ${_results.length}', style: const TextStyle(color: Colors.white)),
                    ),
                  ],
                ),
              ),
              Align(
                alignment: Alignment.bottomCenter,
                child: _buildBottomPanel(),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildBottomPanel() {
    return SafeArea(
      top: false,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.45),
          borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
        ),
        child: DefaultTextStyle(
          style: const TextStyle(color: Colors.white, fontSize: 14),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Text('Detected fields', style: TextStyle(fontWeight: FontWeight.bold)),
                  const Spacer(),
                  Text('Zoom ${_zoom.toStringAsFixed(1)}x'),
                ],
              ),
              const SizedBox(height: 6),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  for (final key in _labels)
                    _FieldChip(
                      label: key,
                      value: _latestByLabel[key]?.confidence != null
                          ? '${(_latestByLabel[key]!.confidence * 100).toStringAsFixed(0)}%'
                          : '—',
                      color: _colorForLabel(key),
                    ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Color _colorForLabel(String label) {
    if (_labels.isEmpty) return Colors.grey;
    final idx = _labels.indexOf(label);
    final hue = (idx / _labels.length) * 360.0;
    return HSLColor.fromAHSL(1.0, hue, 0.8, 0.5).toColor();
  }
}

class _FieldChip extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _FieldChip({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.18),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color, width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle),
          ),
          const SizedBox(width: 6),
          Text(label, style: const TextStyle(color: Colors.white)),
          const SizedBox(width: 8),
          Text(value, style: const TextStyle(color: Colors.white70)),
        ],
      ),
    );
  }
}

class _StatusBanner extends StatelessWidget {
  final double? fps;
  final int lastEventMs;
  const _StatusBanner({required this.fps, required this.lastEventMs});

  @override
  Widget build(BuildContext context) {
    final now = DateTime.now().millisecondsSinceEpoch;
    final stale = now - lastEventMs > 3000; // No events in >3s
    final text = fps != null && !stale
        ? 'FPS ${fps!.toStringAsFixed(1)}'
        : 'No detections/events yet. Check camera permission and model path/format.';
    return Align(
      alignment: Alignment.topLeft,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.45),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(text, style: const TextStyle(color: Colors.white)),
      ),
    );
  }
}

class NidCaptureYoloViewPage extends StatefulWidget {
  final String title;
  final String modelAssetPath;
  final String labelsAssetPath;
  final List<String> requiredLabels;
  const NidCaptureYoloViewPage({
    super.key,
    required this.title,
    required this.modelAssetPath,
    required this.labelsAssetPath,
    this.requiredLabels = const [],
  });

  @override
  State<NidCaptureYoloViewPage> createState() => _NidCaptureYoloViewPageState();
}

class _NidCaptureYoloViewPageState extends State<NidCaptureYoloViewPage> {
  final _controller = YOLOViewController();
  List<YOLOResult> _results = const [];
  List<String> _labels = const [];
  String? _modelFilePath;
  bool _showBoundingBox = true;
  Set<String> _detectedLabels = {};
  bool _isCapturing = false;
  Uint8List? _capturedImageBytes;

  @override
  void initState() {
    super.initState();
    _prepareModelPath();
    _loadLabels();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _controller.setThresholds(confidenceThreshold: 0.60, iouThreshold: 0.50, numItemsThreshold: 100);
    });
  }

  Future<void> _prepareModelPath() async {
    try {
      final dir = await getApplicationSupportDirectory();
      final modelsDir = Directory('${dir.path}/models');
      if (!await modelsDir.exists()) {
        await modelsDir.create(recursive: true);
      }
      final baseName = widget.modelAssetPath.split('/').last;
      final outFile = File('${modelsDir.path}/$baseName');
      final data = await rootBundle.load(widget.modelAssetPath);
      if (!await outFile.exists() || (await outFile.length()) != data.lengthInBytes) {
        await outFile.writeAsBytes(data.buffer.asUint8List(), flush: true);
      }
      if (mounted) setState(() => _modelFilePath = outFile.path);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Model copy failed: $e')),
      );
    }
  }

  Future<void> _loadLabels() async {
    try {
      final txt = await rootBundle.loadString(widget.labelsAssetPath);
      final lines = txt.split(RegExp(r'\r?\n')).where((l) => l.trim().isNotEmpty).toList();
      if (mounted) {
        setState(() {
          _labels = lines;
        });
      }
    } catch (_) {}
  }

  void _onResult(List<YOLOResult> results) {
    final detected = <String>{};
    for (final r in results) {
      final name = r.className.trim();
      if (_labels.contains(name)) detected.add(name);
      else if (r.classIndex >= 0 && r.classIndex < _labels.length) detected.add(_labels[r.classIndex]);
    }
    setState(() {
      _results = List<YOLOResult>.from(results);
      _detectedLabels = detected;
    });
  }

  bool get _canCapture {
    if (widget.requiredLabels.isEmpty) return true;
    return widget.requiredLabels.every((label) => _detectedLabels.contains(label));
  }

  Future<void> _captureImage() async {
    if (_isCapturing) return;
    setState(() { _isCapturing = true; });
    try {
      final frame = await _controller.captureFrame();
      if (mounted && frame != null) {
        setState(() { _capturedImageBytes = frame; });
        _showCapturedImageBottomSheet(frame);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Capture failed: $e')),
        );
      }
    } finally {
      if (mounted) setState(() { _isCapturing = false; });
    }
  }

  void _showCapturedImageBottomSheet(Uint8List bytes) {
    showModalBottomSheet(
      context: context,
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('Captured Image', style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 12),
              Image.memory(bytes, fit: BoxFit.contain),
              const SizedBox(height: 12),
              TextButton(
                onPressed: () => Navigator.pop(ctx),
                child: const Text('Close'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final ready = _modelFilePath != null;
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        actions: [
          IconButton(
            tooltip: 'Switch camera',
            icon: const Icon(Icons.cameraswitch),
            onPressed: () => _controller.switchCamera(),
          ),
        ],
      ),
      body: !ready
          ? const Center(child: Text('Preparing model…'))
          : Stack(
        fit: StackFit.expand,
        children: [
          YOLOView(
            modelPath: _modelFilePath!,
            task: YOLOTask.detect,
            controller: _controller,
            showNativeUI: false,
            useGpu: true,
            confidenceThreshold: 0.60,
            iouThreshold: 0.50,
            streamingConfig: const YOLOStreamingConfig.minimal(),
            onResult: _onResult,
          ),
          Positioned(
            top: 12,
            left: 12,
            right: 12,
            child: Row(
              children: [
                Switch(
                  value: _showBoundingBox,
                  onChanged: (v) => setState(() => _showBoundingBox = v),
                ),
                const SizedBox(width: 8),
                const Text('Show Bounding Box', style: TextStyle(color: Colors.white)),
                const Spacer(),
                ElevatedButton.icon(
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Capture Image'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _canCapture ? Colors.teal : Colors.grey,
                  ),
                  onPressed: _canCapture && !_isCapturing ? _captureImage : null,
                ),
              ],
            ),
          ),
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              color: Colors.black.withOpacity(0.45),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Required Labels:', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 6),
                  Wrap(
                    spacing: 8,
                    children: [
                      for (final label in widget.requiredLabels)
                        Chip(
                          label: Text(label, style: const TextStyle(color: Colors.white)),
                          backgroundColor: _detectedLabels.contains(label) ? Colors.green : Colors.red,
                        ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
