import 'dart:html' as html;
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../models/subject.dart';
import '../../models/attendance.dart';
import '../../services/api_service.dart';
import '../../services/analysis_data_service.dart';
import '../../utils/theme.dart';
import '../../widgets/cards/attendance_card.dart';
import '../../widgets/video/video_preview_widget.dart';
import '../../widgets/tables/csv_data_table_widget.dart';
import 'attendance_list_screen.dart';

/// Attendance Summary Screen — compact stats, donut chart, scrollable date chips
class AttendanceSummaryScreen extends StatefulWidget {
  final Subject subject;

  const AttendanceSummaryScreen({
    super.key,
    required this.subject,
  });

  @override
  State<AttendanceSummaryScreen> createState() =>
      _AttendanceSummaryScreenState();
}

class _AttendanceSummaryScreenState extends State<AttendanceSummaryScreen>
    with SingleTickerProviderStateMixin {
  late DateTime _selectedDate;
  late List<DateTime> _availableDates;
  AttendanceSummaryResponse? _apiSummary;
  List<AnalysisResult> _availableRuns = [];
  AnalysisResult? _selectedRun;
  String _selectedCsvType = 'attendance';

  bool _isLoading = true;
  bool _videoExpanded = false;
  Map<String, dynamic>? _videoMetadata;

  final ApiService _apiService = ApiService();
  final AnalysisDataService _analysisDataService = AnalysisDataService();
  late AnimationController _animController;

  @override
  void initState() {
    super.initState();
    _availableDates = [];
    _selectedDate = DateTime.now();
    _animController = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    );
    _analysisDataService.addListener(_onLiveDataUpdate);
    _initData();
  }

  Future<void> _initData() async {
    await _loadAvailableDates();
    _loadSummary();
  }

  Future<void> _loadAvailableDates() async {
    final dateStrings =
        await _apiService.getAvailableDates(subject: widget.subject.name);
    if (mounted) {
      setState(() {
        if (dateStrings.isNotEmpty) {
          _availableDates = dateStrings.map((s) => DateTime.parse(s)).toList();
          // Sort descending so most recent is first
          _availableDates.sort((a, b) => b.compareTo(a));
          _selectedDate = _availableDates.first;
        } else {
          _availableDates = [];
          _selectedDate = DateTime.now();
        }
      });
    }
  }

  @override
  void dispose() {
    _animController.dispose();
    _analysisDataService.removeListener(_onLiveDataUpdate);
    super.dispose();
  }

  void _onLiveDataUpdate(LiveAttendanceData data) {
    if (mounted) {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _loadSummary() async {
    setState(() {
      _isLoading = true;
      _availableRuns = [];
      _selectedRun = null;
    });

    final dateStr = DateFormat('yyyy-MM-dd').format(_selectedDate);

    // 1. Check if we have live data from a just-completed analysis for this same date
    if (_analysisDataService.latestData != null &&
        _analysisDataService.latestData!.date == dateStr) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _selectedRun = _analysisDataService.latestResult;
        });
      }
      _animController.forward(from: 0);
      return;
    }

    // 2. Otherwise fetch historical runs from backend
    try {
      final runs = await _apiService.getAttendanceRuns(dateStr,
          subjectName: widget.subject.name);

      if (!mounted) return;

      if (runs.isEmpty) {
        // No runs for this specific date, but we might still be in "API mode"
        setState(() {
          _availableRuns = [];
          _selectedRun = null;
          _isLoading = false;
        });
        return;
      }

      setState(() {
        _availableRuns = runs;
        _selectedRun = runs.first; // Default to most recent
      });

      // Parse the CSVs for the selected run
      await _analysisDataService.parseFromResult(_selectedRun!);

      if (!mounted) return;

      setState(() {
        _isLoading = false;
        final live = _analysisDataService.latestData;
        if (live != null) {
          _videoMetadata = {
            'filename': _selectedRun!.topic ?? 'Class Recording',
            'timestamp': _selectedRun!.timestamp,
            'facesDetected': live.totalStudents,
            'recordedBy': 'System Camera',
          };
        }
      });
    } catch (e) {
      print('Error loading attendance for $dateStr: $e');
      if (!mounted) return;

      setState(() {
        _apiSummary = null;
        _selectedRun = null;
        _availableRuns = [];
        _isLoading = false;
        _videoMetadata = null;
      });
    }

    _animController.forward(from: 0);
  }

  LiveAttendanceData? get _parsedData => _analysisDataService.latestData;

  int get _presentCount => _parsedData?.presentCount ?? 0;

  int get _absentCount => _parsedData?.absentCount ?? 0;

  int get _totalStudents => _parsedData?.totalStudents ?? 0;

  double get _attendancePercentage => _parsedData?.attendancePercentage ?? 0.0;

  double get _averageAttention => _parsedData?.averageAttention ?? 0.0;

  int get _phoneUsageCount => _parsedData?.phoneUsageCount ?? 0;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final runIsSelected = _selectedRun != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Attendance Summary'),
        actions: [
          IconButton(
            icon: const Icon(Icons.calendar_month_rounded, size: 20),
            onPressed: () => _showDatePicker(context),
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _loadSummary,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 20),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 600),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Compact subject header
                  _buildSubjectHeader(theme, colorScheme),
                  const SizedBox(height: 16),

                  // Scrollable date chips
                  _buildDateChips(theme, colorScheme),
                  const SizedBox(height: 16),

                  if (_isLoading)
                    const Padding(
                      padding: EdgeInsets.all(40),
                      child: Center(child: CircularProgressIndicator()),
                    )
                  else if (_apiSummary != null || _parsedData != null) ...[
                    // Run Selection (if multiple)
                    if (_availableRuns.length > 1) ...[
                      _buildRunSelector(theme, colorScheme),
                      const SizedBox(height: 16),
                    ],

                    // ── Analysis Content ──────────────────────────────────────────
                    if (runIsSelected) ...[
                      // Premium view: Exactly like the Upload Video result screen
                      _buildDetailedAnalysisModules(theme, colorScheme),
                    ] else ...[
                      // View when run is not selected but summary exists (handled above usually)
                      _buildSummaryCards(),
                      const SizedBox(height: 16),
                      _buildDonutChart(theme, colorScheme),
                      const SizedBox(height: 16),
                    ],

                    // Quick stats (useful for both views)
                    _buildQuickStats(theme, colorScheme),
                    const SizedBox(height: 20),
                  ] else ...[
                    const Padding(
                      padding: EdgeInsets.symmetric(vertical: 48),
                      child: Center(
                        child: Column(
                          children: [
                            Icon(Icons.video_camera_back_outlined,
                                size: 48, color: Colors.grey),
                            SizedBox(height: 12),
                            Text(
                              'No analysis data yet',
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                                color: Colors.grey,
                              ),
                            ),
                            SizedBox(height: 6),
                            Text(
                              'Upload a classroom video to see attendance results here.',
                              textAlign: TextAlign.center,
                              style:
                                  TextStyle(fontSize: 13, color: Colors.grey),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }



  Widget _buildSubjectHeader(ThemeData theme, ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        gradient: AppTheme.subtleGradient,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: colorScheme.outlineVariant.withOpacity(0.2),
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: AppTheme.primaryColor.withOpacity(0.12),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(
              widget.subject.icon,
              size: 22,
              color: AppTheme.primaryColor,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.subject.name,
                  style: theme.textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  widget.subject.code,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: colorScheme.onSurfaceVariant,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDateChips(ThemeData theme, ColorScheme colorScheme) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Select Date',
          style: theme.textTheme.titleSmall?.copyWith(
            fontWeight: FontWeight.w600,
            fontSize: 13,
          ),
        ),
        const SizedBox(height: 8),
        SizedBox(
          height: 42,
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            itemCount: _availableDates.length,
            itemBuilder: (context, index) {
              final date = _availableDates[index];
              final isSelected = _isSameDay(date, _selectedDate);
              final dayFormat = DateFormat('EEE');
              final dateFormat = DateFormat('dd MMM');

              return Padding(
                padding: EdgeInsets.only(right: 6),
                child: GestureDetector(
                  onTap: () {
                    setState(() => _selectedDate = date);
                    _loadSummary();
                  },
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: isSelected
                          ? AppTheme.primaryColor
                          : colorScheme.surface,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                        color: isSelected
                            ? AppTheme.primaryColor
                            : colorScheme.outlineVariant.withOpacity(0.3),
                      ),
                      boxShadow: isSelected ? AppTheme.softShadow : null,
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          dayFormat.format(date),
                          style: theme.textTheme.labelSmall?.copyWith(
                            color: isSelected
                                ? Colors.white.withOpacity(0.8)
                                : colorScheme.onSurfaceVariant,
                            fontWeight: FontWeight.w500,
                            fontSize: 10,
                          ),
                        ),
                        const SizedBox(width: 4),
                        Text(
                          dateFormat.format(date),
                          style: theme.textTheme.labelMedium?.copyWith(
                            color: isSelected
                                ? Colors.white
                                : colorScheme.onSurface,
                            fontWeight: FontWeight.w600,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildSummaryCards() {
    return GridView.count(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisCount: 4,
      crossAxisSpacing: 8,
      mainAxisSpacing: 8,
      childAspectRatio: 0.68,
      children: [
        AttendanceCard(
          title: 'Present',
          value: _presentCount.toString(),
          icon: Icons.check_circle_rounded,
          type: AttendanceCardType.present,
        ),
        AttendanceCard(
          title: 'Absent',
          value: _absentCount.toString(),
          icon: Icons.cancel_rounded,
          type: AttendanceCardType.absent,
        ),
        AttendanceCard(
          title: 'Phone',
          value: _phoneUsageCount.toString(),
          icon: Icons.phone_android_rounded,
          type: AttendanceCardType.late,
        ),
        AttendanceCard(
          title: 'Rate',
          value: '${_attendancePercentage.toStringAsFixed(0)}%',
          icon: Icons.pie_chart_rounded,
          type: AttendanceCardType.percentage,
        ),
      ],
    );
  }

  Widget _buildDonutChart(ThemeData theme, ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: colorScheme.outlineVariant.withOpacity(0.25),
        ),
        boxShadow: AppTheme.softShadow,
      ),
      child: Row(
        children: [
          // Donut chart
          SizedBox(
            width: 100,
            height: 100,
            child: TweenAnimationBuilder<double>(
              tween: Tween(begin: 0, end: 1),
              duration: const Duration(milliseconds: 800),
              curve: Curves.easeOutCubic,
              builder: (context, animValue, _) {
                return PieChart(
                  PieChartData(
                    sectionsSpace: 2,
                    centerSpaceRadius: 28,
                    startDegreeOffset: -90,
                    sections: [
                      PieChartSectionData(
                        value: _presentCount.toDouble() * animValue,
                        color: AppTheme.presentColor,
                        radius: 16,
                        showTitle: false,
                      ),
                      PieChartSectionData(
                        value: _absentCount.toDouble() * animValue,
                        color: AppTheme.absentColor,
                        radius: 16,
                        showTitle: false,
                      ),
                      if (_presentCount == 0 && _absentCount == 0)
                        PieChartSectionData(
                          value: 1,
                          color: colorScheme.outlineVariant.withOpacity(0.3),
                          radius: 16,
                          showTitle: false,
                        ),
                    ],
                  ),
                );
              },
            ),
          ),
          const SizedBox(width: 20),
          // Legend
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _LegendItem(
                  color: AppTheme.presentColor,
                  label: 'Present',
                  value: _presentCount,
                ),
                const SizedBox(height: 8),
                _LegendItem(
                  color: AppTheme.absentColor,
                  label: 'Absent',
                  value: _absentCount,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAttentionStats(ThemeData theme, ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.tertiaryColor.withOpacity(0.06),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: AppTheme.tertiaryColor.withOpacity(0.15),
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: AppTheme.tertiaryColor.withOpacity(0.12),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(
              Icons.psychology_rounded,
              size: 20,
              color: AppTheme.tertiaryColor,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Average Attention',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: colorScheme.onSurfaceVariant,
                    fontSize: 11,
                  ),
                ),
                Text(
                  '${_averageAttention.toStringAsFixed(1)}%',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: AppTheme.tertiaryColor,
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: _averageAttention >= 70
                  ? AppTheme.presentColor.withOpacity(0.12)
                  : _averageAttention >= 50
                      ? AppTheme.warningColor.withOpacity(0.12)
                      : AppTheme.absentColor.withOpacity(0.12),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              _averageAttention >= 70
                  ? 'Good'
                  : _averageAttention >= 50
                      ? 'Fair'
                      : 'Low',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: _averageAttention >= 70
                    ? AppTheme.presentColor
                    : _averageAttention >= 50
                        ? AppTheme.warningColor
                        : AppTheme.absentColor,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQuickStats(ThemeData theme, ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: colorScheme.outlineVariant.withOpacity(0.25),
        ),
        boxShadow: AppTheme.softShadow,
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              children: [
                Text(
                  '$_totalStudents',
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: AppTheme.primaryColor,
                  ),
                ),
                Text(
                  'Total Students',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: colorScheme.onSurfaceVariant,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
          Container(
            height: 32,
            width: 1,
            color: colorScheme.outlineVariant.withOpacity(0.3),
          ),
          Expanded(
            child: Column(
              children: [
                Text(
                  '$_phoneUsageCount',
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: AppTheme.tertiaryColor,
                  ),
                ),
                Text(
                  'Phone Users',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: colorScheme.onSurfaceVariant,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildVideoMetadata(ThemeData theme, ColorScheme colorScheme) {
    return Container(
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: colorScheme.outlineVariant.withOpacity(0.25),
        ),
        boxShadow: AppTheme.softShadow,
      ),
      child: Column(
        children: [
          // Header - always visible
          InkWell(
            borderRadius: BorderRadius.circular(14),
            onTap: () => setState(() => _videoExpanded = !_videoExpanded),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(7),
                    decoration: BoxDecoration(
                      color: Colors.deepPurple.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(9),
                    ),
                    child: Icon(
                      Icons.videocam_rounded,
                      size: 16,
                      color: Colors.deepPurple,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Class Video',
                          style: theme.textTheme.titleSmall?.copyWith(
                            fontWeight: FontWeight.w600,
                            fontSize: 14,
                          ),
                        ),
                        Text(
                          _videoMetadata != null
                              ? 'Recording available'
                              : 'No video for this date',
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: _videoMetadata != null
                                ? AppTheme.presentColor
                                : colorScheme.onSurfaceVariant,
                            fontSize: 11,
                          ),
                        ),
                      ],
                    ),
                  ),
                  if (_videoMetadata != null) ...[
                    // Status badge
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: AppTheme.presentColor.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        'Processed',
                        style: theme.textTheme.labelSmall?.copyWith(
                          color: AppTheme.presentColor,
                          fontWeight: FontWeight.w600,
                          fontSize: 9,
                        ),
                      ),
                    ),
                    const SizedBox(width: 6),
                  ],
                  AnimatedRotation(
                    turns: _videoExpanded ? 0.5 : 0,
                    duration: const Duration(milliseconds: 200),
                    child: Icon(
                      Icons.keyboard_arrow_down_rounded,
                      color: colorScheme.onSurfaceVariant,
                      size: 20,
                    ),
                  ),
                ],
              ),
            ),
          ),
          // Expandable content
          AnimatedCrossFade(
            firstChild: _videoMetadata != null
                ? _buildVideoDetails(theme, colorScheme)
                : Padding(
                    padding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: colorScheme.surfaceContainerHighest
                            .withOpacity(0.3),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Column(
                        children: [
                          Icon(
                            Icons.videocam_off_outlined,
                            size: 28,
                            color: colorScheme.onSurfaceVariant,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'No video recorded for this date',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: colorScheme.onSurfaceVariant,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
            secondChild: const SizedBox.shrink(),
            crossFadeState: _videoExpanded
                ? CrossFadeState.showFirst
                : CrossFadeState.showSecond,
            duration: const Duration(milliseconds: 250),
          ),
        ],
      ),
    );
  }

  Widget _buildVideoDetails(ThemeData theme, ColorScheme colorScheme) {
    final meta = _videoMetadata ?? {};
    final durationSec = (meta['durationSeconds'] as num?)?.toInt() ?? 5400;
    final durationStr =
        '${(durationSec ~/ 3600).toString().padLeft(1, '0')}h ${((durationSec % 3600) ~/ 60).toString().padLeft(2, '0')}m';
    final fileSizeMb = (meta['fileSizeMb'] as num?)?.toInt() ?? 1024;
    final fileSizeStr = fileSizeMb >= 1024
        ? '${(fileSizeMb / 1024).toStringAsFixed(1)} GB'
        : '$fileSizeMb MB';

    final filename = meta['filename'] as String? ?? 'Class_Recording.mp4';
    final resolution = meta['resolution'] as String? ?? '1080p';
    final fps = meta['fps']?.toString() ?? '30';
    final facesDetected = (meta['facesDetected'] as num?)?.toInt() ?? 0;
    final studentsRecognized =
        (meta['studentsRecognized'] as num?)?.toInt() ?? 0;
    final recordedBy = meta['recordedBy'] as String? ?? 'System';

    return Padding(
      padding: const EdgeInsets.fromLTRB(14, 0, 14, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Filename + play row
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.deepPurple.withOpacity(0.06),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(6),
                  decoration: BoxDecoration(
                    color: Colors.deepPurple.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: const Icon(Icons.play_arrow_rounded,
                      size: 16, color: Colors.deepPurple),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    filename,
                    style: theme.textTheme.bodySmall?.copyWith(
                      fontWeight: FontWeight.w500,
                      fontSize: 11,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                Text(
                  durationStr,
                  style: theme.textTheme.labelSmall?.copyWith(
                    color: colorScheme.onSurfaceVariant,
                    fontSize: 10,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),

          // Compact stats — all in one row
          Row(
            children: [
              _CompactMeta(Icons.sd_storage_rounded, fileSizeStr,
                  AppTheme.tertiaryColor),
              const SizedBox(width: 6),
              _CompactMeta(Icons.high_quality_rounded, resolution,
                  AppTheme.warningColor),
              const SizedBox(width: 6),
              _CompactMeta(
                  Icons.speed_rounded, '$fps FPS', AppTheme.presentColor),
            ],
          ),
          const SizedBox(height: 6),

          // Processing results in one row
          Row(
            children: [
              _CompactMeta(Icons.face_rounded, '$facesDetected faces',
                  Colors.deepPurple),
              const SizedBox(width: 6),
              _CompactMeta(Icons.how_to_reg_rounded,
                  '$studentsRecognized recognized', AppTheme.presentColor),
              const SizedBox(width: 6),
              _CompactMeta(Icons.person_outline_rounded, recordedBy,
                  colorScheme.onSurfaceVariant),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildViewDetailsButton(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton.icon(
        onPressed: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => AttendanceListScreen(
                subject: widget.subject,
                date: _selectedDate,
              ),
            ),
          );
        },
        icon: const Icon(Icons.list_alt_rounded, size: 18),
        label: const Text('View Detailed List'),
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 14),
        ),
      ),
    );
  }

  void _showDatePicker(BuildContext context) async {
    final picked = await showDatePicker(
      context: context,
      initialDate: _selectedDate,
      firstDate: DateTime.now().subtract(const Duration(days: 365)),
      lastDate: DateTime.now(),
    );

    if (picked != null && mounted) {
      setState(() => _selectedDate = picked);
      _loadSummary();
    }
  }

  bool _isSameDay(DateTime a, DateTime b) {
    return a.year == b.year && a.month == b.month && a.day == b.day;
  }

  Widget _buildRunSelector(ThemeData theme, ColorScheme colorScheme) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Select Session',
          style: theme.textTheme.titleSmall?.copyWith(
            fontWeight: FontWeight.w600,
            fontSize: 13,
          ),
        ),
        const SizedBox(height: 8),
        SizedBox(
          height: 38,
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            itemCount: _availableRuns.length,
            itemBuilder: (context, index) {
              final run = _availableRuns[index];
              final isSelected = _selectedRun?.runId == run.runId;

              return Padding(
                padding: const EdgeInsets.only(right: 6),
                child: ChoiceChip(
                  label: Text(
                    run.topic ?? 'Session ${index + 1}',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight:
                          isSelected ? FontWeight.bold : FontWeight.normal,
                    ),
                  ),
                  selected: isSelected,
                  onSelected: (selected) {
                    if (selected) {
                      setState(() => _selectedRun = run);
                      _analysisDataService.parseFromResult(run).then((_) {
                        if (mounted) setState(() {});
                      });
                    }
                  },
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildDetailedAnalysisModules(
      ThemeData theme, ColorScheme colorScheme) {
    final run = _selectedRun!;
    final data = _analysisDataService.latestData;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (data != null) ...[
          _buildLiveStatsSection(theme, colorScheme, data),
          const SizedBox(height: 24),
          if (data.records.isNotEmpty) ...[
            _buildStudentList(theme, colorScheme, data.records),
            const SizedBox(height: 24),
          ],
        ],

        // ── Speech & Seating Specialized Modules ─────────────────────────────
        _buildExtraAnalysisModules(theme, colorScheme, run),
        const SizedBox(height: 24),

        // ── Downloads ────────────────────────────────────────────────────────
        _buildDownloadArtifactsSection(theme, colorScheme, run),
        const SizedBox(height: 32),
      ],
    );
  }

  Widget _buildDownloadTile(BuildContext context,
      {required IconData icon,
      required Color color,
      required String title,
      required String subtitle,
      String? url}) {
    final colorScheme = Theme.of(context).colorScheme;
    final theme = Theme.of(context);
    final isAvailable = url != null && url.isNotEmpty;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: isAvailable
              ? color.withOpacity(0.3)
              : colorScheme.outlineVariant.withOpacity(0.2),
        ),
        boxShadow: isAvailable
            ? [
                BoxShadow(
                  color: color.withOpacity(0.08),
                  blurRadius: 8,
                  offset: const Offset(0, 3),
                )
              ]
            : null,
      ),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        leading: Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: color.withOpacity(isAvailable ? 0.15 : 0.08),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, color: isAvailable ? color : Colors.grey, size: 24),
        ),
        title: Text(
          title,
          style: theme.textTheme.titleSmall?.copyWith(
            fontWeight: FontWeight.bold,
            color: isAvailable ? null : colorScheme.onSurfaceVariant,
          ),
        ),
        subtitle: Text(
          isAvailable ? subtitle : 'Artifact not available in this run',
          style: theme.textTheme.bodySmall?.copyWith(
            color: colorScheme.onSurfaceVariant,
            fontSize: 11,
          ),
        ),
        trailing: isAvailable
            ? IconButton.filled(
                onPressed: () => _openUrl(url),
                icon: const Icon(Icons.download_rounded, size: 20),
                style: IconButton.styleFrom(
                  backgroundColor: color,
                  foregroundColor: Colors.white,
                ),
              )
            : const Icon(Icons.lock_clock_outlined,
                color: Colors.grey, size: 18),
      ),
    );
  }

  Widget _buildExtraAnalysisModules(
      ThemeData theme, ColorScheme colorScheme, AnalysisResult run) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // ── Speech Topic Analysis ──
        if (run.hasSpeechVideo || run.hasSpeechCsv) ...[
          _buildDetailedSectionHeader(theme, '🎙️ Speech Topic Analysis',
              Icons.record_voice_over_outlined),
          const SizedBox(height: 12),
          if (run.hasSpeechVideo)
            _buildDownloadTile(
              context,
              icon: Icons.record_voice_over_rounded,
              color: Colors.teal,
              title: 'Speech Topic Video',
              subtitle: 'Annotated with class-related / off-topic labels',
              url: run.speechVideoUrl,
            ),
          if (run.hasSpeechVideo && run.hasSpeechCsv)
            const SizedBox(height: 10),
          if (run.hasSpeechCsv)
            _buildDownloadTile(
              context,
              icon: Icons.text_snippet_outlined,
              color: Colors.teal,
              title: 'Speech Topic Segments (CSV)',
              subtitle: 'Timestamped class-related segments',
              url: run.speechCsvUrl,
            ),
          const SizedBox(height: 24),
        ],

        // ── Seat Map ──
        if (run.hasSeatMapPng || run.hasSeatMapJson) ...[
          _buildDetailedSectionHeader(
              theme, '🗺️ Seat Map', Icons.map_outlined),
          const SizedBox(height: 12),
          if (run.hasSeatMapPng) ...[
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.network(
                run.seatMapPngUrl!,
                fit: BoxFit.contain,
                errorBuilder: (_, __, ___) => const SizedBox.shrink(),
              ),
            ),
            const SizedBox(height: 10),
          ],
          if (run.hasSeatMapJson)
            _buildDownloadTile(
              context,
              icon: Icons.event_seat_rounded,
              color: Colors.indigo,
              title: 'Seat Map (JSON)',
              subtitle: 'Student-to-seat assignment data',
              url: run.seatMapJsonUrl,
            ),
          const SizedBox(height: 24),
        ],

        // ── Seating Timeline & Events ──
        if (run.hasSeatingTimeline || run.hasAttendanceEvents) ...[
          _buildDetailedSectionHeader(
              theme, '📋 Seating & Events', Icons.timeline_rounded),
          const SizedBox(height: 12),
          if (run.hasSeatingTimeline)
            _buildDownloadTile(
              context,
              icon: Icons.timeline_rounded,
              color: Colors.deepPurple,
              title: 'Seating Timeline (CSV)',
              subtitle: 'Occupation history over time',
              url: run.seatingTimelineUrl,
            ),
          if (run.hasSeatingTimeline && run.hasAttendanceEvents)
            const SizedBox(height: 10),
          if (run.hasAttendanceEvents)
            _buildDownloadTile(
              context,
              icon: Icons.event_note_rounded,
              color: Colors.deepOrange,
              title: 'Attendance Events (CSV)',
              subtitle: 'Entry, exit & seat-shift events',
              url: run.attendanceEventsUrl,
            ),
          const SizedBox(height: 24),
        ],
      ],
    );
  }

  Widget _buildCsvExplorerSection(
      ThemeData theme, ColorScheme colorScheme, AnalysisResult run) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildDetailedSectionHeader(
            theme, 'Detailed Reports Explorer', Icons.analytics_outlined),
        const SizedBox(height: 12),
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(
            children: [
              _csvChip('attendance', 'Attendance List'),
              _csvChip('activity', 'Activity Log'),
              if (run.hasSpeechCsv) _csvChip('speech', 'Speech Analysis'),
              if (run.hasSeatingTimeline) _csvChip('seating', 'Seating Map'),
            ],
          ),
        ),
        const SizedBox(height: 16),
        _buildReportTable(run),
      ],
    );
  }

  Widget _buildDownloadArtifactsSection(
      ThemeData theme, ColorScheme colorScheme, AnalysisResult run) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildDetailedSectionHeader(
            theme, 'Download Results', Icons.file_download_outlined),
        const SizedBox(height: 12),
        if (run.hasAttentionVideo) ...[
          _buildDownloadTile(
            context,
            icon: Icons.video_file_rounded,
            color: Colors.purple,
            title: 'Attention & Seating Video',
            subtitle: 'Annotated analysis recording',
            url: run.attentionVideoUrl,
          ),
          const SizedBox(height: 10),
        ],
        if (run.hasActivityVideo) ...[
          _buildDownloadTile(
            context,
            icon: Icons.movie_filter_rounded,
            color: Colors.orange,
            title: 'Activity Tracking Video',
            subtitle: 'Pose detection recording',
            url: run.activityVideoUrl,
          ),
          const SizedBox(height: 10),
        ],
        _buildDownloadTile(
          context,
          icon: Icons.table_chart_rounded,
          color: Colors.green,
          title: 'Attendance Report (CSV)',
          subtitle: 'attendance_report.csv',
          url: run.attendanceCsvUrl,
        ),
        if (run.hasActivityCsv) ...[
          const SizedBox(height: 10),
          _buildDownloadTile(
            context,
            icon: Icons.bar_chart_rounded,
            color: Colors.blue,
            title: 'Activity Summary (CSV)',
            subtitle: 'person_activity_summary.csv',
            url: run.activityCsvUrl,
          ),
        ],
      ],
    );
  }

  Widget _buildLiveStatsSection(
      ThemeData theme, ColorScheme colorScheme, LiveAttendanceData d) {
    final pct = d.attendancePercentage;
    final pctColor = pct >= 75
        ? Colors.green
        : pct >= 50
            ? Colors.orange
            : Colors.red;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text('Attendance Summary',
                style: theme.textTheme.titleMedium
                    ?.copyWith(fontWeight: FontWeight.w600)),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: Colors.green.withAlpha(25),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.green.withAlpha(60)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.check_circle_outline_rounded,
                      size: 10, color: Colors.green),
                  const SizedBox(width: 4),
                  Text('Live from CSV',
                      style: TextStyle(
                          fontSize: 10,
                          color: Colors.green.shade700,
                          fontWeight: FontWeight.w600)),
                ],
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            _statCard(theme, '${d.presentCount}', 'Present', Colors.green,
                Icons.check_circle_rounded),
            const SizedBox(width: 10),
            _statCard(theme, '${d.absentCount}', 'Absent', Colors.red,
                Icons.cancel_rounded),
            const SizedBox(width: 10),
            _statCard(theme, '${d.totalStudents}', 'Total', Colors.blue,
                Icons.people_rounded),
            const SizedBox(width: 10),
            _statCard(theme, '${pct.toStringAsFixed(0)}%', 'Rate', pctColor,
                Icons.pie_chart_rounded),
          ],
        ),
        if (d.averageAttention > 0 || d.sittingCount > 0) ...[
          const SizedBox(height: 10),
          Row(
            children: [
              if (d.averageAttention > 0)
                Expanded(
                  child: _metricTile(
                    theme,
                    colorScheme,
                    icon: Icons.psychology_rounded,
                    color: Colors.purple,
                    label: 'Avg Attention',
                    value: '${d.averageAttention.toStringAsFixed(1)}%',
                  ),
                ),
              if (d.averageAttention > 0 && d.sittingCount > 0)
                const SizedBox(width: 10),
              if (d.sittingCount > 0)
                Expanded(
                  child: _metricTile(
                    theme,
                    colorScheme,
                    icon: Icons.event_seat_rounded,
                    color: Colors.teal,
                    label: 'Sitting',
                    value: '${d.sittingCount}',
                  ),
                ),
            ],
          ),
        ],
      ],
    );
  }

  Widget _statCard(
      ThemeData theme, String value, String label, Color color, IconData icon) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
        decoration: BoxDecoration(
          color: color.withAlpha(18),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withAlpha(50)),
        ),
        child: Column(
          children: [
            Icon(icon, color: color, size: 20),
            const SizedBox(height: 4),
            Text(value,
                style: theme.textTheme.titleMedium
                    ?.copyWith(fontWeight: FontWeight.bold, color: color)),
            Text(label,
                style: theme.textTheme.labelSmall
                    ?.copyWith(color: color, fontSize: 10)),
          ],
        ),
      ),
    );
  }

  Widget _metricTile(ThemeData theme, ColorScheme colorScheme,
      {required IconData icon,
      required Color color,
      required String label,
      required String value}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withAlpha(40)),
      ),
      child: Row(
        children: [
          Icon(icon, color: color, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(value,
                    style: theme.textTheme.titleSmall
                        ?.copyWith(fontWeight: FontWeight.bold, color: color)),
                Text(label,
                    style: theme.textTheme.labelSmall?.copyWith(
                        color: colorScheme.onSurfaceVariant, fontSize: 10)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStudentList(
      ThemeData theme, ColorScheme colorScheme, List<StudentRecord> records) {
    const maxVisible = 8;
    final visible = records.take(maxVisible).toList();

    return Container(
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(14, 12, 14, 6),
            child: Text('Participants (${records.length})',
                style: theme.textTheme.titleSmall
                    ?.copyWith(fontWeight: FontWeight.w600)),
          ),
          const Divider(height: 1),
          ...visible.map((r) => ListTile(
                dense: true,
                visualDensity: VisualDensity.compact,
                leading: CircleAvatar(
                  radius: 14,
                  backgroundColor:
                      (r.isPresent ? Colors.green : Colors.red).withAlpha(25),
                  child: Text(
                    r.name.isNotEmpty ? r.name[0].toUpperCase() : '?',
                    style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: r.isPresent ? Colors.green : Colors.red),
                  ),
                ),
                title: Text(r.name,
                    style: theme.textTheme.bodySmall
                        ?.copyWith(fontWeight: FontWeight.w500)),
                trailing: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (r.attentionScore > 0)
                      Text('${r.attentionScore.toStringAsFixed(0)}%',
                          style: TextStyle(
                              fontSize: 11,
                              color: colorScheme.onSurfaceVariant)),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 7, vertical: 2),
                      decoration: BoxDecoration(
                        color: (r.isPresent ? Colors.green : Colors.red)
                            .withAlpha(20),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        r.isPresent ? 'Present' : 'Absent',
                        style: TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.w600,
                            color: r.isPresent
                                ? Colors.green.shade700
                                : Colors.red.shade700),
                      ),
                    ),
                  ],
                ),
              )),
          if (records.length > maxVisible)
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 4, 14, 10),
              child: Text(
                  '+ ${records.length - maxVisible} more (view full table below)',
                  style: theme.textTheme.bodySmall?.copyWith(
                      color: colorScheme.onSurfaceVariant, fontSize: 10)),
            ),
        ],
      ),
    );
  }

  Widget _buildVideoColumn(
      ThemeData theme, String label, String url, String subtitle) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: theme.textTheme.labelLarge?.copyWith(
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
              Text(
                subtitle,
                style: theme.textTheme.bodySmall?.copyWith(
                  fontSize: 10,
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
        VideoPreviewWidget(
          videoUrl: url,
          aspectRatio: 16 / 9,
          autoPlay: false,
        ),
      ],
    );
  }

  Widget _buildDetailedSectionHeader(
      ThemeData theme, String title, IconData icon) {
    return Row(
      children: [
        Icon(icon, size: 18, color: theme.colorScheme.primary),
        const SizedBox(width: 8),
        Text(
          title,
          style:
              theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold),
        ),
      ],
    );
  }

  Widget _csvChip(String type, String label) {
    final isSelected = _selectedCsvType == type;
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: ChoiceChip(
        label: Text(label, style: const TextStyle(fontSize: 11)),
        selected: isSelected,
        onSelected: (selected) {
          if (selected) setState(() => _selectedCsvType = type);
        },
      ),
    );
  }

  Widget _buildReportTable(AnalysisResult run) {
    String? url;
    String title = 'Report Data';

    switch (_selectedCsvType) {
      case 'attendance':
        url = run.attendanceCsvUrl;
        title = 'Student Attendance & Attention';
        break;
      case 'activity':
        url = run.activityCsvUrl;
        title = 'Activity Summary';
        break;
      case 'speech':
        url = run.speechCsvUrl;
        title = 'Speech Segments';
        break;
      case 'seating':
        url = run.seatingTimelineUrl;
        title = 'Seating Timeline';
        break;
    }

    if (url == null) return const SizedBox.shrink();

    return CsvDataTableWidget(csvUrl: url, title: title);
  }

  void _openUrl(String url) {
    html.window.open(url, '_blank');
  }
}

/// Legend item for the donut chart
class _LegendItem extends StatelessWidget {
  final Color color;
  final String label;
  final int value;

  const _LegendItem({
    required this.color,
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Row(
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(3),
          ),
        ),
        const SizedBox(width: 8),
        Text(
          label,
          style: theme.textTheme.bodySmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
            fontSize: 12,
          ),
        ),
        const Spacer(),
        Text(
          value.toString(),
          style: theme.textTheme.titleSmall?.copyWith(
            fontWeight: FontWeight.w600,
            fontSize: 14,
          ),
        ),
      ],
    );
  }
}

/// Ultra-compact icon+text metadata pill for the video section
class _CompactMeta extends StatelessWidget {
  final IconData icon;
  final String text;
  final Color color;

  const _CompactMeta(this.icon, this.text, this.color);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        decoration: BoxDecoration(
          color: color.withOpacity(0.06),
          borderRadius: BorderRadius.circular(7),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 12, color: color),
            const SizedBox(width: 4),
            Expanded(
              child: Text(
                text,
                style: theme.textTheme.labelSmall?.copyWith(
                  fontSize: 9,
                  fontWeight: FontWeight.w500,
                  color: color,
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
