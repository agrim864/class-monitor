import 'package:flutter/material.dart';
import '../../services/api_service.dart';
import '../../services/database_service.dart';
import '../../models/subject.dart';
import '../../utils/theme.dart';
import '../../widgets/common/empty_state.dart';
import '../../widgets/tables/csv_data_table_widget.dart';
import 'package:intl/intl.dart';

class DataExplorerScreen extends StatefulWidget {
  const DataExplorerScreen({super.key});

  @override
  State<DataExplorerScreen> createState() => _DataExplorerScreenState();
}

class _DataExplorerScreenState extends State<DataExplorerScreen> {
  String _selectedSubjectId = '';
  List<Subject> _subjects = [];
  bool _isLoading = true;

  AnalysisResult? _selectedRun;
  List<AnalysisResult> _availableRuns = [];
  String _selectedCsvTab = 'final_summary';

  @override
  void initState() {
    super.initState();
    _loadSubjects();
  }

  Future<void> _loadSubjects() async {
    setState(() => _isLoading = true);
    try {
      final subjects = await DatabaseService().getSubjects();
      if (subjects.isNotEmpty) {
        _subjects = subjects;
        _selectedSubjectId = subjects.first.id;
      }
      await _loadRuns();
    } catch (e) {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _loadRuns() async {
    if (!mounted || _subjects.isEmpty) return;
    setState(() => _isLoading = true);

    try {
      final selectedSubject = _subjects.firstWhere((s) => s.id == _selectedSubjectId);
      final dates = await ApiService().getAvailableDates(subject: selectedSubject.name);
      
      final runs = <AnalysisResult>[];
      for (final date in dates) {
        try {
          final dailyRuns = await ApiService().getAttendanceRuns(date, subjectName: selectedSubject.name);
          runs.addAll(dailyRuns);
        } catch (_) {}
      }

      runs.sort((a, b) {
        final dateA = a.timestamp != null ? DateTime.tryParse(a.timestamp!) ?? DateTime.fromMillisecondsSinceEpoch(0) : DateTime.fromMillisecondsSinceEpoch(0);
        final dateB = b.timestamp != null ? DateTime.tryParse(b.timestamp!) ?? DateTime.fromMillisecondsSinceEpoch(0) : DateTime.fromMillisecondsSinceEpoch(0);
        return dateB.compareTo(dateA);
      });
      
      if (mounted) {
        setState(() {
          _availableRuns = runs;
          _selectedRun = runs.isNotEmpty ? runs.first : null;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _availableRuns = [];
          _selectedRun = null;
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;
    final runIsSelected = _selectedRun != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Data Reports Explorer'),
      ),
      body: _isLoading && _subjects.isEmpty
          ? const Center(child: CircularProgressIndicator())
          : _subjects.isEmpty
              ? const EmptyState(
                  icon: Icons.table_chart_outlined,
                  title: 'No Subjects Found',
                  subtitle: 'Add a subject first to view reports.',
                )
              : Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      _buildSubjectHeader(theme, colorScheme),
                      const SizedBox(height: 16),
                      if (_isLoading)
                        const Expanded(child: Center(child: CircularProgressIndicator()))
                      else if (_availableRuns.isEmpty)
                        Expanded(
                          child: EmptyState(
                            icon: Icons.monitor_outlined,
                            title: 'No CSV Reports Yet',
                            subtitle: 'Upload a classroom video to generate analysis datasets.',
                            onAction: _loadRuns,
                            actionLabel: 'Refresh',
                          ),
                        )
                      else ...[
                        if (_availableRuns.length > 1)
                          _buildRunSelector(theme, colorScheme),
                        const SizedBox(height: 16),
                        if (runIsSelected) ...[
                          _buildCsvTabBar(theme, colorScheme),
                          const SizedBox(height: 16),
                          Expanded(child: _buildSelectedCsvTable(theme)),
                        ]
                      ],
                    ],
                  ),
                ),
    );
  }

  Widget _buildSubjectHeader(ThemeData theme, ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: colorScheme.outlineVariant.withOpacity(0.5)),
        boxShadow: AppTheme.softShadow,
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: _selectedSubjectId,
          isExpanded: true,
          icon: Icon(Icons.keyboard_arrow_down_rounded, color: colorScheme.onSurface),
          items: _subjects.map((subject) {
            return DropdownMenuItem(
              value: subject.id,
              child: Text(
                subject.name,
                style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
              ),
            );
          }).toList(),
          onChanged: (value) {
            if (value != null && value != _selectedSubjectId) {
              setState(() => _selectedSubjectId = value);
              _loadRuns();
            }
          },
        ),
      ),
    );
  }

  Widget _buildRunSelector(ThemeData theme, ColorScheme colorScheme) {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        children: _availableRuns.map((run) {
          final isSelected = _selectedRun?.timestamp == run.timestamp;
          final runDate = run.timestamp != null ? DateTime.tryParse(run.timestamp!) ?? DateTime.now() : DateTime.now();
          final dateStr = DateFormat('MMM d, yyyy - h:mm a').format(runDate);
          
          return Padding(
            padding: const EdgeInsets.only(right: 8.0),
            child: ChoiceChip(
              label: Text(run.topic ?? dateStr),
              selected: isSelected,
              onSelected: (selected) {
                if (selected) setState(() => _selectedRun = run);
              },
              backgroundColor: colorScheme.surfaceContainerHighest,
              selectedColor: AppTheme.primaryColor.withOpacity(0.15),
              labelStyle: TextStyle(
                color: isSelected ? AppTheme.primaryColor : colorScheme.onSurfaceVariant,
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              ),
              side: BorderSide(
                color: isSelected ? AppTheme.primaryColor : Colors.transparent,
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildCsvTabBar(ThemeData theme, ColorScheme colorScheme) {
    final run = _selectedRun!;
    
    final Map<String, String> tabs = {};
    if (run.hasFinalStudentSummary) tabs['final_summary'] = 'Final Summaries';
    if (run.hasAttendanceCsv) tabs['attendance'] = 'Attendance Report';
    if (run.hasAttendanceEvents) tabs['events'] = 'Attendance Events';
    if (run.hasActivityCsv) tabs['activity'] = 'Activity Summary';
    if (run.hasSeatShifts) tabs['seat_shifts'] = 'Seat Shifts';
    if (run.hasSeatingTimeline) tabs['seating_timeline'] = 'Seating Timeline';
    if (run.hasSpeechCsv) tabs['speech'] = 'Speech Analysis';

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!tabs.containsKey(_selectedCsvTab) && tabs.isNotEmpty) {
        setState(() => _selectedCsvTab = tabs.keys.first);
      }
    });

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        children: tabs.entries.map((entry) {
          final isSelected = _selectedCsvTab == entry.key;
          return Padding(
            padding: const EdgeInsets.only(right: 12.0),
            child: InkWell(
              borderRadius: BorderRadius.circular(20),
              onTap: () => setState(() => _selectedCsvTab = entry.key),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: isSelected ? colorScheme.primary : colorScheme.surface,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: isSelected ? colorScheme.primary : colorScheme.outline.withOpacity(0.2),
                  ),
                ),
                child: Text(
                  entry.value,
                  style: theme.textTheme.labelMedium?.copyWith(
                    fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
                    color: isSelected ? colorScheme.onPrimary : colorScheme.onSurface,
                  ),
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildSelectedCsvTable(ThemeData theme) {
    final run = _selectedRun!;
    String? url;
    String title = 'Data Report';

    switch (_selectedCsvTab) {
      case 'final_summary':
        url = run.finalStudentSummaryUrl;
        title = 'Final Output (Merged Analytics)';
        break;
      case 'attendance':
        url = run.attendanceCsvUrl;
        title = 'Student Attendance List';
        break;
      case 'events':
        url = run.attendanceEventsUrl;
        title = 'Continuous Attendance Log';
        break;
      case 'activity':
        url = run.activityCsvUrl;
        title = 'Pose & Activity Summary';
        break;
      case 'seat_shifts':
        url = run.seatShiftsUrl;
        title = 'Out of Seat Shifts';
        break;
      case 'seating_timeline':
        url = run.seatingTimelineUrl;
        title = 'Dynamic Seating Flow';
        break;
      case 'speech':
        url = run.speechCsvUrl;
        title = 'Recognized Speech Transcript';
        break;
    }

    if (url == null || url.isEmpty) {
      return Center(
        child: Text('Report not available for this run.', style: theme.textTheme.bodyMedium),
      );
    }

    return Card(
      margin: EdgeInsets.zero,
      elevation: 0,
      color: Colors.transparent,
      child: CsvDataTableWidget(
        key: ValueKey(url),
        csvUrl: url,
        title: title,
      ),
    );
  }
}
