import 'dart:async';
import 'package:flutter/material.dart';
import '../../services/api_service.dart';
import '../../utils/theme.dart';

class GenerateEmbeddingsScreen extends StatefulWidget {
  const GenerateEmbeddingsScreen({super.key});

  @override
  State<GenerateEmbeddingsScreen> createState() =>
      _GenerateEmbeddingsScreenState();
}

class _GenerateEmbeddingsScreenState extends State<GenerateEmbeddingsScreen> {
  bool _runLocal = true;
  bool _runAngle = false;
  bool _runAccessory = false;
  bool _runAngleCombos = false;
  bool _runRebuildDb = true;

  bool _isGenerating = false;
  String? _currentJobId;
  int _progress = 0;
  List<String> _logs = [];
  Timer? _pollingTimer;

  final ScrollController _scrollController = ScrollController();

  @override
  void dispose() {
    _pollingTimer?.cancel();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _startGeneration() async {
    if (_isGenerating) return;

    if (!_runLocal &&
        !_runAngle &&
        !_runAccessory &&
        !_runAngleCombos &&
        !_runRebuildDb) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select at least one task to run.')),
      );
      return;
    }

    setState(() {
      _isGenerating = true;
      _logs = ['Starting generation jobs...'];
      _progress = 0;
      _currentJobId = null;
    });

    try {
      final response = await ApiService().generateEmbeddings(
        runLocal: _runLocal,
        runAngle: _runAngle,
        runAccessory: _runAccessory,
        runAngleCombos: _runAngleCombos,
        runRebuildDb: _runRebuildDb,
      );

      final jobId = response['job_id'];
      if (jobId != null) {
        setState(() {
          _currentJobId = jobId;
        });
        _startPolling(jobId);
      } else {
        throw Exception('No job ID returned.');
      }
    } catch (e) {
      setState(() {
        _isGenerating = false;
        _logs.add('Error submitting job: \$e');
      });
      _scrollToBottom();
    }
  }

  void _startPolling(String jobId) {
    _pollingTimer?.cancel();
    _pollingTimer = Timer.periodic(const Duration(milliseconds: 1500), (timer) async {
      try {
        final jobInfo = await ApiService().getEmbeddingJob(jobId);
        if (mounted) {
          setState(() {
            _progress = jobInfo.progress;
            _logs = jobInfo.logs;
          });
          _scrollToBottom();

          if (jobInfo.status == 'completed' || jobInfo.status == 'failed') {
            timer.cancel();
            setState(() {
              _isGenerating = false;
              if (jobInfo.status == 'completed') {
                _progress = 100;
                _logs.add('All tasks completed successfully.');
              } else {
                _logs.add('Job failed: \${jobInfo.errorMessage}');
              }
            });
            _scrollToBottom();
          }
        }
      } catch (e) {
        // Suppress polling error visually, just log it internally
      }
    });
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Generate Embeddings & Augmentations'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  flex: 1,
                  child: Container(
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: colorScheme.surface,
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: AppTheme.softShadow,
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Configuration',
                          style: theme.textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 16),
                        _buildCheckbox(
                          'Local Augmentations',
                          'Deterministic basic transformations (light, blur, etc)',
                          _runLocal,
                          (v) => setState(() => _runLocal = v ?? false),
                        ),
                        _buildCheckbox(
                          'Angle Augmentations (Hosted)',
                          'Generates varying head poses using external API',
                          _runAngle,
                          (v) => setState(() => _runAngle = v ?? false),
                        ),
                        _buildCheckbox(
                          'Accessory Blockers',
                          'Add generated caps, masks, and sunglasses',
                          _runAccessory,
                          (v) => setState(() => _runAccessory = v ?? false),
                        ),
                        _buildCheckbox(
                          'Angle Combos',
                          'Apply deterministic transforms on top of Angle outputs',
                          _runAngleCombos,
                          (v) => setState(() => _runAngleCombos = v ?? false),
                        ),
                        const Divider(height: 32),
                        _buildCheckbox(
                          'Rebuild Identity DB',
                          'Create vector embeddings from all generated resources',
                          _runRebuildDb,
                          (v) => setState(() => _runRebuildDb = v ?? false),
                        ),
                        const SizedBox(height: 32),
                        SizedBox(
                          width: double.infinity,
                          child: FilledButton.icon(
                            onPressed: _isGenerating ? null : _startGeneration,
                            icon: _isGenerating
                                ? const SizedBox(
                                    width: 16,
                                    height: 16,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      color: Colors.white,
                                    ),
                                  )
                                : const Icon(Icons.flash_on_rounded),
                            label: Text(_isGenerating ? 'Running...' : 'Generate Selected'),
                            style: FilledButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 20),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 24),
                Expanded(
                  flex: 2,
                  child: Container(
                    height: 500,
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1E1E1E),
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: AppTheme.softShadow,
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text(
                              'Execution Logs',
                              style: TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                              ),
                            ),
                            if (_isGenerating || _progress > 0)
                              Text(
                                '\$_progress%',
                                style: TextStyle(
                                  color: Colors.greenAccent.shade400,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        if (_isGenerating || _progress > 0)
                          LinearProgressIndicator(
                            value: _progress / 100.0,
                            backgroundColor: Colors.white12,
                            color: Colors.greenAccent.shade400,
                            minHeight: 6,
                            borderRadius: BorderRadius.circular(3),
                          ),
                        const SizedBox(height: 16),
                        Expanded(
                          child: _logs.isEmpty
                              ? const Center(
                                  child: Text(
                                    'Awaiting execution...',
                                    style: TextStyle(color: Colors.white38),
                                  ),
                                )
                              : ListView.builder(
                                  controller: _scrollController,
                                  itemCount: _logs.length,
                                  itemBuilder: (context, index) {
                                    return Padding(
                                      padding: const EdgeInsets.only(bottom: 6.0),
                                      child: Text(
                                        '> \${_logs[index]}',
                                        style: const TextStyle(
                                          color: Colors.white70,
                                          fontFamily: 'monospace',
                                          fontSize: 13,
                                        ),
                                      ),
                                    );
                                  },
                                ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCheckbox(
    String title,
    String subtitle,
    bool value,
    ValueChanged<bool?> onChanged,
  ) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: CheckboxListTile(
        title: Text(
          title,
          style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
        ),
        subtitle: Text(
          subtitle,
          style: const TextStyle(fontSize: 12),
        ),
        value: value,
        onChanged: _isGenerating ? null : onChanged,
        contentPadding: EdgeInsets.zero,
        controlAffinity: ListTileControlAffinity.leading,
        activeColor: AppTheme.primaryColor,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
    );
  }
}
