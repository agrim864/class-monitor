import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';
import '../../models/attendance.dart';
import '../../models/subject.dart';
import '../../services/database_service.dart';
import '../../services/api_service.dart';
import '../../utils/theme.dart';

/// Attendance Analytics Screen — polished with compact cards and refined charts
class AttendanceAnalyticsScreen extends StatefulWidget {
  const AttendanceAnalyticsScreen({super.key});

  @override
  State<AttendanceAnalyticsScreen> createState() =>
      _AttendanceAnalyticsScreenState();
}

class _AttendanceAnalyticsScreenState extends State<AttendanceAnalyticsScreen> {
  String _selectedSubjectId = '';
  List<Subject> _subjects = [];
  List<AttendanceAnalytics> _analytics = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadSubjectsAndAnalytics();
  }

  Future<void> _loadSubjectsAndAnalytics() async {
    setState(() => _isLoading = true);
    try {
      final subjects = await DatabaseService().getSubjects();
      if (subjects.isNotEmpty) {
        _subjects = subjects;
        _selectedSubjectId = subjects.first.id;
      }
      await _loadAnalytics();
    } catch (e) {
      print('Error loading subjects: \$e');
      setState(() => _isLoading = false);
    }
  }
  Future<void> _loadAnalytics() async {
    if (!mounted) return;
    setState(() => _isLoading = true);

    try {
      final selectedSubject = _subjects.firstWhere(
        (s) => s.id == _selectedSubjectId,
        orElse: () => _subjects.first,
      );

      final dates = await ApiService().getAvailableDates(subject: selectedSubject.name);
      
      final List<AttendanceAnalytics> loaded = [];
      // Take up to the 10 most recent dates
      final recentDates = dates.take(10).toList().reversed.toList();

      for (final dateStr in recentDates) {
        try {
          final summary = await ApiService().getAttendance(
            dateStr,
            subjectId: selectedSubject.id,
            subjectName: selectedSubject.name,
          );
          loaded.add(AttendanceAnalytics(
            date: DateTime.parse(dateStr),
            attendancePercentage: summary.averageAttention > 0 // backend gives attention percentage too, but we need attendance
                ? ((summary.presentCount / summary.totalStudents) * 100)
                : 0.0,
            presentCount: summary.presentCount,
            totalStudents: summary.totalStudents,
          ));
        } catch (e) {
          print('Error fetching attendance for $dateStr: $e');
        }
      }

      if (mounted) {
        setState(() {
          _analytics = loaded;
          _isLoading = false;
        });
      }
    } catch (e) {
      print('Error in _loadAnalytics: $e');
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Analytics'),
        automaticallyImplyLeading: false,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildSubjectSelector(theme, colorScheme),
            const SizedBox(height: 16),
            _buildAverageCard(theme, colorScheme),
            const SizedBox(height: 16),
            _buildTrendSection(theme, colorScheme),
            const SizedBox(height: 16),
            _buildBarChartSection(theme, colorScheme),
          ],
        ),
      ),
    );
  }

  Widget _buildSubjectSelector(ThemeData theme, ColorScheme colorScheme) {
    if (_subjects.isEmpty) return const SizedBox();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 14),
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: colorScheme.outlineVariant.withOpacity(0.3),
        ),
        boxShadow: AppTheme.softShadow,
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: _selectedSubjectId,
          isExpanded: true,
          borderRadius: BorderRadius.circular(12),
          icon: Icon(Icons.keyboard_arrow_down_rounded,
              color: colorScheme.onSurfaceVariant, size: 20),
          items: _subjects.map((subject) {
            return DropdownMenuItem(
              value: subject.id,
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(4),
                    decoration: BoxDecoration(
                      color: AppTheme.primaryColor.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Icon(subject.icon,
                        size: 16, color: AppTheme.primaryColor),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      subject.name,
                      overflow: TextOverflow.ellipsis,
                      style: theme.textTheme.bodyMedium?.copyWith(
                        fontSize: 13,
                      ),
                    ),
                  ),
                ],
              ),
            );
          }).toList(),
          onChanged: (value) {
            if (value != null) {
              setState(() => _selectedSubjectId = value);
              _loadAnalytics();
            }
          },
        ),
      ),
    );
  }

  Widget _buildAverageCard(ThemeData theme, ColorScheme colorScheme) {
    final average = _analytics.isEmpty
        ? 0.0
        : _analytics
                .map((a) => a.attendancePercentage)
                .reduce((a, b) => a + b) /
            _analytics.length;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: AppTheme.primaryGradient,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: AppTheme.primaryColor.withOpacity(0.2),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(
              Icons.analytics_rounded,
              size: 28,
              color: Colors.white,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${average.toStringAsFixed(1)}%',
                  style: theme.textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
                Text(
                  'Average Attendance · ${_analytics.length} classes',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: Colors.white.withOpacity(0.7),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTrendSection(ThemeData theme, ColorScheme colorScheme) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Attendance Trend',
          style: theme.textTheme.titleSmall?.copyWith(
            fontWeight: FontWeight.w600,
            fontSize: 14,
          ),
        ),
        const SizedBox(height: 10),
        Container(
          height: 180,
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 8),
          decoration: BoxDecoration(
            color: colorScheme.surface,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
              color: colorScheme.outlineVariant.withOpacity(0.25),
            ),
            boxShadow: AppTheme.softShadow,
          ),
          child: _isLoading
              ? const Center(child: CircularProgressIndicator())
              : _analytics.isEmpty
                  ? const Center(child: Text('No data available'))
                  : _buildLineChart(colorScheme),
        ),
      ],
    );
  }

  Widget _buildLineChart(ColorScheme colorScheme) {
    return LineChart(
      LineChartData(
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: 25,
          getDrawingHorizontalLine: (value) {
            return FlLine(
              color: colorScheme.outlineVariant.withOpacity(0.15),
              strokeWidth: 1,
            );
          },
        ),
        titlesData: FlTitlesData(
          show: true,
          rightTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 24,
              interval: 2,
              getTitlesWidget: (value, meta) {
                final index = value.toInt();
                if (index >= 0 && index < _analytics.length) {
                  return Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Text(
                      DateFormat('dd').format(_analytics[index].date),
                      style: TextStyle(
                        color: colorScheme.onSurfaceVariant,
                        fontSize: 9,
                      ),
                    ),
                  );
                }
                return const SizedBox();
              },
            ),
          ),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              interval: 25,
              reservedSize: 32,
              getTitlesWidget: (value, meta) {
                return Text(
                  '${value.toInt()}%',
                  style: TextStyle(
                    color: colorScheme.onSurfaceVariant,
                    fontSize: 9,
                  ),
                );
              },
            ),
          ),
        ),
        borderData: FlBorderData(show: false),
        minX: 0,
        maxX: (_analytics.length - 1).toDouble(),
        minY: 0,
        maxY: 100,
        lineBarsData: [
          LineChartBarData(
            spots: _analytics.asMap().entries.map((entry) {
              return FlSpot(
                entry.key.toDouble(),
                entry.value.attendancePercentage,
              );
            }).toList(),
            isCurved: true,
            color: AppTheme.primaryColor,
            barWidth: 2.5,
            isStrokeCapRound: true,
            dotData: FlDotData(
              show: true,
              getDotPainter: (spot, percent, barData, index) {
                return FlDotCirclePainter(
                  radius: 3,
                  color: AppTheme.primaryColor,
                  strokeWidth: 1.5,
                  strokeColor: colorScheme.surface,
                );
              },
            ),
            belowBarData: BarAreaData(
              show: true,
              color: AppTheme.primaryColor.withOpacity(0.08),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBarChartSection(ThemeData theme, ColorScheme colorScheme) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Class-wise Attendance',
          style: theme.textTheme.titleSmall?.copyWith(
            fontWeight: FontWeight.w600,
            fontSize: 14,
          ),
        ),
        const SizedBox(height: 10),
        Container(
          height: 180,
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 8),
          decoration: BoxDecoration(
            color: colorScheme.surface,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
              color: colorScheme.outlineVariant.withOpacity(0.25),
            ),
            boxShadow: AppTheme.softShadow,
          ),
          child: _isLoading
              ? const Center(child: CircularProgressIndicator())
              : _analytics.isEmpty
                  ? const Center(child: Text('No data available'))
                  : _buildBarChart(colorScheme),
        ),
      ],
    );
  }

  Widget _buildBarChart(ColorScheme colorScheme) {
    return BarChart(
      BarChartData(
        alignment: BarChartAlignment.spaceAround,
        maxY: 100,
        barTouchData: BarTouchData(
          enabled: true,
          touchTooltipData: BarTouchTooltipData(
            getTooltipItem: (group, groupIndex, rod, rodIndex) {
              return BarTooltipItem(
                '${rod.toY.toStringAsFixed(1)}%',
                const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 11,
                ),
              );
            },
          ),
        ),
        titlesData: FlTitlesData(
          show: true,
          rightTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 24,
              getTitlesWidget: (value, meta) {
                final index = value.toInt();
                if (index >= 0 && index < _analytics.length) {
                  return Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Text(
                      DateFormat('dd').format(_analytics[index].date),
                      style: TextStyle(
                        color: colorScheme.onSurfaceVariant,
                        fontSize: 9,
                      ),
                    ),
                  );
                }
                return const SizedBox();
              },
            ),
          ),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              interval: 25,
              reservedSize: 32,
              getTitlesWidget: (value, meta) {
                return Text(
                  '${value.toInt()}%',
                  style: TextStyle(
                    color: colorScheme.onSurfaceVariant,
                    fontSize: 9,
                  ),
                );
              },
            ),
          ),
        ),
        borderData: FlBorderData(show: false),
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: 25,
          getDrawingHorizontalLine: (value) {
            return FlLine(
              color: colorScheme.outlineVariant.withOpacity(0.15),
              strokeWidth: 1,
            );
          },
        ),
        barGroups: _analytics.asMap().entries.map((entry) {
          final percentage = entry.value.attendancePercentage;
          return BarChartGroupData(
            x: entry.key,
            barRods: [
              BarChartRodData(
                toY: percentage,
                color: _getBarColor(percentage),
                width: 14,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(4),
                  topRight: Radius.circular(4),
                ),
              ),
            ],
          );
        }).toList(),
      ),
    );
  }

  Color _getBarColor(double percentage) {
    if (percentage >= 85) return AppTheme.presentColor;
    if (percentage >= 70) return AppTheme.warningColor;
    return AppTheme.absentColor;
  }
}
