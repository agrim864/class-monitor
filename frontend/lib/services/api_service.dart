import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../models/subject.dart';
import '../models/user.dart';

/// REST client for the local classroom-monitor FastAPI backend.
class ApiService {
  static String _baseUrl = 'http://localhost:8000';
  static String? authToken;

  static String get baseUrl => _baseUrl;

  static set baseUrl(String value) {
    _baseUrl =
        value.endsWith('/') ? value.substring(0, value.length - 1) : value;
  }

  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  Map<String, String> get _jsonHeaders => {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        if (authToken != null && authToken!.isNotEmpty)
          'Authorization': 'Bearer $authToken',
      };

  Map<String, String> get authHeaders => {
        'Accept': 'application/json',
        if (authToken != null && authToken!.isNotEmpty)
          'Authorization': 'Bearer $authToken',
      };

  Future<bool> healthCheck() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 6));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<AuthResponse> login(String email, String password) async {
    final response = await http
        .post(
          Uri.parse('$baseUrl/api/auth/login'),
          headers: _jsonHeaders,
          body: json.encode({'email': email, 'password': password}),
        )
        .timeout(const Duration(seconds: 20));
    if (response.statusCode != 200) {
      throw Exception(_errorMessage(response, 'Login failed'));
    }
    final data = json.decode(response.body) as Map<String, dynamic>;
    final token = data['token']?.toString() ?? '';
    final userJson = data['user'] as Map<String, dynamic>;
    return AuthResponse(token: token, user: userFromJson(userJson));
  }

  Future<User> me() async {
    final response = await http
        .get(Uri.parse('$baseUrl/api/auth/me'), headers: authHeaders)
        .timeout(const Duration(seconds: 12));
    if (response.statusCode != 200) {
      throw Exception(_errorMessage(response, 'Session expired'));
    }
    final data = json.decode(response.body) as Map<String, dynamic>;
    return userFromJson(data['user'] as Map<String, dynamic>);
  }

  Future<void> logout() async {
    try {
      await http
          .post(Uri.parse('$baseUrl/api/auth/logout'), headers: authHeaders)
          .timeout(const Duration(seconds: 8));
    } catch (_) {
      // Local logout should still clear the browser session if the backend is offline.
    }
  }

  Future<List<Subject>> getSubjects() async {
    final response = await http
        .get(Uri.parse('$baseUrl/api/subjects'), headers: authHeaders)
        .timeout(const Duration(seconds: 15));
    if (response.statusCode != 200) {
      throw Exception(_errorMessage(response, 'Failed to load subjects'));
    }
    final data = json.decode(response.body) as List<dynamic>;
    return data
        .map((item) => subjectFromJson(item as Map<String, dynamic>))
        .toList();
  }

  Future<Subject> createSubject({
    required String name,
    required String code,
    required String description,
    required int iconIndex,
    required int totalStudents,
  }) async {
    final response = await http
        .post(
          Uri.parse('$baseUrl/api/subjects'),
          headers: _jsonHeaders,
          body: json.encode({
            'name': name,
            'code': code,
            'description': description,
            'icon_index': iconIndex,
            'total_students': totalStudents,
          }),
        )
        .timeout(const Duration(seconds: 15));
    if (response.statusCode != 200) {
      throw Exception(_errorMessage(response, 'Failed to create subject'));
    }
    return subjectFromJson(json.decode(response.body) as Map<String, dynamic>);
  }

  Future<void> deleteSubject(String subjectId) async {
    final response = await http
        .delete(Uri.parse('$baseUrl/api/subjects/$subjectId'),
            headers: authHeaders)
        .timeout(const Duration(seconds: 15));
    if (response.statusCode != 200) {
      throw Exception(_errorMessage(response, 'Failed to delete subject'));
    }
  }

  Future<Map<String, dynamic>> generateEmbeddings({
    bool runLocal = false,
    bool runAngle = false,
    bool runAccessory = false,
    bool runAngleCombos = false,
    bool runRebuildDb = false,
  }) async {
    final uri = Uri.parse('$baseUrl/api/embeddings/generate');
    final request = http.MultipartRequest('POST', uri);
    
    // Convert bool to "true"/"false" strings
    request.fields['run_local'] = runLocal ? "true" : "false";
    request.fields['run_angle'] = runAngle ? "true" : "false";
    request.fields['run_accessory'] = runAccessory ? "true" : "false";
    request.fields['run_angle_combos'] = runAngleCombos ? "true" : "false";
    request.fields['run_rebuild_db'] = runRebuildDb ? "true" : "false";

    final authHeadersStr = authHeaders['Authorization'];
    if (authHeadersStr != null) {
      request.headers['Authorization'] = authHeadersStr;
    }

    final response = await request.send();
    if (response.statusCode == 200 || response.statusCode == 201) {
      final respStr = await response.stream.bytesToString();
      return jsonDecode(respStr);
    } else {
      throw Exception('Failed to start embeddings generation');
    }
  }

  Future<EmbeddingJobResponse> getEmbeddingJob(String jobId) async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/embeddings/jobs/$jobId'),
      headers: authHeaders,
    );
    if (response.statusCode == 200) {
      return EmbeddingJobResponse.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to get embedding job details');
    }
  }

  Future<List<String>> getAvailableDates({String? subject}) async {
    try {
      final query = subject != null
          ? '?subject=${Uri.encodeQueryComponent(subject)}'
          : '';
      final response = await http
          .get(Uri.parse('$baseUrl/api/dates$query'))
          .timeout(const Duration(seconds: 10));
      if (response.statusCode != 200) return [];
      final data = json.decode(response.body) as List<dynamic>;
      return data.map((item) => item.toString()).toList();
    } catch (e) {
      print('Error fetching available dates: $e');
      return [];
    }
  }

  Future<AnalysisJobStatus> getAnalysisJobStatus(String jobId) async {
    final response = await http
        .get(Uri.parse('$baseUrl/api/analysis/jobs/$jobId'),
            headers: authHeaders)
        .timeout(const Duration(seconds: 15));
    if (response.statusCode != 200) {
      throw Exception('Failed to fetch analysis job: ${response.statusCode}');
    }
    return AnalysisJobStatus.fromJson(
        json.decode(response.body) as Map<String, dynamic>);
  }

  Future<Uint8List> downloadFile(String serverPath) async {
    final encodedPath = Uri.encodeQueryComponent(serverPath);
    final response = await http
        .get(Uri.parse('$baseUrl/api/files?path=$encodedPath'),
            headers: authHeaders)
        .timeout(const Duration(minutes: 5));
    if (response.statusCode != 200) {
      throw Exception('Download failed: ${response.statusCode}');
    }
    return response.bodyBytes;
  }

  Future<String> fetchTextFile(String serverPath) async {
    final bytes = await downloadFile(serverPath);
    return utf8.decode(bytes);
  }

  Future<String> askRagQuestion(String query, {String? courseId}) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/api/query-rag'),
        headers: authHeaders,
        body: {'query': query, 'course_id': courseId ?? 'global'},
      ).timeout(const Duration(seconds: 30));
      if (response.statusCode != 200) {
        throw Exception('RAG query failed (${response.statusCode})');
      }
      final jsonMap = json.decode(response.body) as Map<String, dynamic>;
      return jsonMap['answer']?.toString() ?? 'No response received.';
    } catch (e) {
      print('RAG Error: $e');
      return _getMockRagResponse(query);
    }
  }

  Future<List<AnalysisResult>> getAttendanceRuns(String date,
      {String? subjectName}) async {
    try {
      final query = subjectName != null
          ? '?subject=${Uri.encodeQueryComponent(subjectName)}'
          : '';
      final response = await http
          .get(Uri.parse('$baseUrl/api/attendance/$date$query'),
              headers: authHeaders)
          .timeout(const Duration(seconds: 10));
      if (response.statusCode == 404) return [];
      if (response.statusCode != 200) {
        throw Exception('Server error: ${response.statusCode}');
      }
      final data = json.decode(response.body) as List<dynamic>;
      return data
          .map((run) =>
              AnalysisResult.fromApiJson(run as Map<String, dynamic>, baseUrl))
          .toList();
    } catch (e) {
      print('getAttendanceRuns error: $e');
      return [];
    }
  }

  Future<AttendanceSummaryResponse> getAttendance(String date,
      {String subjectId = 'default', String? subjectName}) async {
    final runs = await getAttendanceRuns(date, subjectName: subjectName);
    if (runs.isEmpty || runs.first.attendanceCsvPath == null) {
      throw Exception('No analysis data found for $date');
    }
    final csvText = await fetchTextFile(runs.first.attendanceCsvPath!);
    return _parseAttendanceCsvText(csvText, date, subjectId);
  }

  String _getMockRagResponse(String query) {
    final lowerQuery = query.toLowerCase();
    if (lowerQuery.contains('cnn') || lowerQuery.contains('convolutional')) {
      return 'A Convolutional Neural Network (CNN) learns spatial features using shared filters and is widely used for images.';
    } else if (lowerQuery.contains('backpropagation')) {
      return 'Backpropagation computes gradients of the loss with respect to model weights using the chain rule.';
    } else if (lowerQuery.contains('overfitting')) {
      return 'Overfitting happens when a model memorizes training noise and performs worse on unseen data.';
    }
    return 'The study assistant endpoint is available, and this is a fallback response while the full RAG stack is being wired up.';
  }

  AttendanceSummaryResponse _parseAttendanceCsvText(
      String csv, String date, String subjectId) {
    final lines = csv.trim().split('\n');
    if (lines.length < 2) {
      return AttendanceSummaryResponse(
          date: date,
          subjectId: subjectId,
          totalStudents: 0,
          presentCount: 0,
          absentCount: 0,
          averageAttention: 0,
          phoneUsageCount: 0,
          records: []);
    }
    final header = lines[0].toLowerCase().split(',');
    int idxPresent = header.indexOf('present');
    int idxAttention = header.indexOf('attention_percentage');
    if (idxAttention < 0) idxAttention = header.indexOf('attention_score');
    final idxPhone = header.indexOf('usingphone_frames');
    int present = 0;
    double sumAttention = 0;
    int phoneUsers = 0;
    final total = lines.length - 1;
    for (int i = 1; i < lines.length; i++) {
      final cols = lines[i].split(',');
      if (idxPresent >= 0 && idxPresent < cols.length) {
        final presentValue = cols[idxPresent].toLowerCase().trim();
        if (['yes', 'present', 'true', '1'].contains(presentValue)) present++;
      }
      if (idxAttention >= 0 && idxAttention < cols.length) {
        sumAttention += double.tryParse(cols[idxAttention]) ?? 0;
      }
      if (idxPhone >= 0 &&
          idxPhone < cols.length &&
          (int.tryParse(cols[idxPhone]) ?? 0) > 0) {
        phoneUsers++;
      }
    }
    return AttendanceSummaryResponse(
      date: date,
      subjectId: subjectId,
      totalStudents: total,
      presentCount: present,
      absentCount: total - present,
      averageAttention: total > 0 ? sumAttention / total : 0,
      phoneUsageCount: phoneUsers,
      records: [],
    );
  }

  String _errorMessage(http.Response response, String fallback) {
    try {
      final data = json.decode(response.body) as Map<String, dynamic>;
      return data['detail']?.toString() ?? fallback;
    } catch (_) {
      return fallback;
    }
  }
}

class AuthResponse {
  final String token;
  final User user;

  const AuthResponse({required this.token, required this.user});
}

User userFromJson(Map<String, dynamic> json) {
  return User(
    id: json['id']?.toString() ?? '',
    name: json['name']?.toString() ?? 'User',
    email: json['email']?.toString() ?? '',
    role: json['role']?.toString() == 'student'
        ? UserRole.student
        : UserRole.instructor,
    department: json['department']?.toString() ?? 'Computer Science',
  );
}

Subject subjectFromJson(Map<String, dynamic> json) {
  final iconIndex = (json['icon_index'] as num?)?.toInt() ?? 0;
  const iconOptions = [
    Icons.account_tree_rounded,
    Icons.psychology_rounded,
    Icons.storage_rounded,
    Icons.hub_rounded,
    Icons.engineering_rounded,
    Icons.computer_rounded,
    Icons.code_rounded,
    Icons.calculate_rounded,
    Icons.science_rounded,
    Icons.architecture_rounded,
    Icons.school_rounded,
    Icons.analytics_rounded,
  ];
  final safeIconIndex = iconIndex.clamp(0, iconOptions.length - 1);
  return Subject(
    id: json['id']?.toString() ?? '',
    name: json['name']?.toString() ?? '',
    code: json['code']?.toString() ?? '',
    icon: iconOptions[safeIconIndex],
    instructorName: json['instructor_name']?.toString() ?? 'Instructor',
    instructorId: json['instructor_id']?.toString() ?? '',
    description: json['description']?.toString(),
    totalStudents: (json['total_students'] as num?)?.toInt() ?? 0,
    attendancePercentage:
        (json['attendance_percentage'] as num?)?.toDouble() ?? 0.0,
    totalClasses: (json['total_classes'] as num?)?.toInt() ?? 0,
  );
}

class AnalysisJobStatus {
  final String jobId;
  final String status;
  final double progress;
  final String currentStep;
  final String? errorMessage;
  final AnalysisResult? result;

  const AnalysisJobStatus({
    required this.jobId,
    required this.status,
    required this.progress,
    required this.currentStep,
    this.errorMessage,
    this.result,
  });

  bool get isCompleted => status == 'completed';
  bool get isFailed => status == 'failed';

  factory AnalysisJobStatus.fromJson(Map<String, dynamic> json) {
    final resultJson = json['result'];
    return AnalysisJobStatus(
      jobId: json['job_id']?.toString() ?? '',
      status: json['status']?.toString() ?? 'unknown',
      progress: ((json['progress'] as num?)?.toDouble() ?? 0.0) / 100.0,
      currentStep: json['current_step']?.toString() ?? '',
      errorMessage: json['error_message']?.toString(),
      result: resultJson is Map<String, dynamic>
          ? AnalysisResult.fromApiJson(resultJson, ApiService.baseUrl)
          : null,
    );
  }
}

class AnalysisResult {
  final String? faceIdentityVideoPath;
  final String? faceIdentityCsvPath;
  final String? attentionVideoPath;
  final String? attendanceCsvPath;
  final String? attendancePresencePath;
  final String? attentionMetricsPath;
  final String? activityVideoPath;
  final String? activityCsvPath;
  final String? deviceUsePath;
  final String? handRaisePath;
  final String? noteTakingPath;
  final String? visualSpeakingVideoPath;
  final String? visualSpeakingCsvPath;
  final String? lipReadingTranscriptPath;
  final String? speechTopicClassificationPath;
  final String? speechVideoPath;
  final String? speechCsvPath;
  final String? seatMapPngPath;
  final String? seatMapJsonPath;
  final String? seatingTimelinePath;
  final String? seatShiftsPath;
  final String? attendanceEventsPath;
  final String? finalStudentSummaryPath;
  final String? runManifestPath;
  final String logText;
  final String baseUrl;
  final String? logTextPath;
  final String? runId;
  final String? topic;
  final String? timestamp;

  AnalysisResult({
    this.faceIdentityVideoPath,
    this.faceIdentityCsvPath,
    this.attentionVideoPath,
    this.attendanceCsvPath,
    this.attendancePresencePath,
    this.attentionMetricsPath,
    this.activityVideoPath,
    this.activityCsvPath,
    this.deviceUsePath,
    this.handRaisePath,
    this.noteTakingPath,
    this.visualSpeakingVideoPath,
    this.visualSpeakingCsvPath,
    this.lipReadingTranscriptPath,
    this.speechTopicClassificationPath,
    this.speechVideoPath,
    this.speechCsvPath,
    this.seatMapPngPath,
    this.seatMapJsonPath,
    this.seatingTimelinePath,
    this.seatShiftsPath,
    this.attendanceEventsPath,
    this.finalStudentSummaryPath,
    this.runManifestPath,
    required this.logText,
    required this.baseUrl,
    this.logTextPath,
    this.runId,
    this.topic,
    this.timestamp,
  });

  factory AnalysisResult.fromApiJson(
      Map<String, dynamic> json, String baseUrl) {
    return AnalysisResult(
      faceIdentityVideoPath: json['faceIdentityVideoPath']?.toString(),
      faceIdentityCsvPath: json['faceIdentityCsvPath']?.toString(),
      attentionVideoPath: json['attentionVideoPath']?.toString(),
      attendanceCsvPath: json['attendanceCsvPath']?.toString(),
      attendancePresencePath: json['attendancePresencePath']?.toString(),
      attentionMetricsPath: json['attentionMetricsPath']?.toString(),
      activityVideoPath: json['activityVideoPath']?.toString(),
      activityCsvPath: json['activityCsvPath']?.toString(),
      deviceUsePath: json['deviceUsePath']?.toString(),
      handRaisePath: json['handRaisePath']?.toString(),
      noteTakingPath: json['noteTakingPath']?.toString(),
      visualSpeakingVideoPath: json['visualSpeakingVideoPath']?.toString(),
      visualSpeakingCsvPath: json['visualSpeakingCsvPath']?.toString(),
      lipReadingTranscriptPath: json['lipReadingTranscriptPath']?.toString(),
      speechTopicClassificationPath:
          json['speechTopicClassificationPath']?.toString(),
      speechVideoPath: json['speechVideoPath']?.toString(),
      speechCsvPath: json['speechCsvPath']?.toString(),
      seatMapPngPath: json['seatMapPngPath']?.toString(),
      seatMapJsonPath: json['seatMapJsonPath']?.toString(),
      seatingTimelinePath: json['seatingTimelinePath']?.toString(),
      seatShiftsPath: json['seatShiftsPath']?.toString(),
      attendanceEventsPath: json['attendanceEventsPath']?.toString(),
      finalStudentSummaryPath: json['finalStudentSummaryPath']?.toString(),
      runManifestPath: json['runManifestPath']?.toString(),
      logText: json['logText']?.toString() ?? '',
      baseUrl: baseUrl,
      logTextPath: json['logTextPath']?.toString(),
      runId: json['run_id']?.toString() ?? json['runId']?.toString(),
      topic: json['topic']?.toString(),
      timestamp: json['timestamp']?.toString(),
    );
  }

  factory AnalysisResult.fromCacheJson(Map<String, dynamic> json) {
    return AnalysisResult.fromApiJson(
        json, json['baseUrl'] as String? ?? ApiService.baseUrl);
  }

  Map<String, dynamic> toJson() => {
        'faceIdentityVideoPath': faceIdentityVideoPath,
        'faceIdentityCsvPath': faceIdentityCsvPath,
        'attentionVideoPath': attentionVideoPath,
        'attendanceCsvPath': attendanceCsvPath,
        'attendancePresencePath': attendancePresencePath,
        'attentionMetricsPath': attentionMetricsPath,
        'activityVideoPath': activityVideoPath,
        'activityCsvPath': activityCsvPath,
        'deviceUsePath': deviceUsePath,
        'handRaisePath': handRaisePath,
        'noteTakingPath': noteTakingPath,
        'visualSpeakingVideoPath': visualSpeakingVideoPath,
        'visualSpeakingCsvPath': visualSpeakingCsvPath,
        'lipReadingTranscriptPath': lipReadingTranscriptPath,
        'speechTopicClassificationPath': speechTopicClassificationPath,
        'speechVideoPath': speechVideoPath,
        'speechCsvPath': speechCsvPath,
        'seatMapPngPath': seatMapPngPath,
        'seatMapJsonPath': seatMapJsonPath,
        'seatingTimelinePath': seatingTimelinePath,
        'seatShiftsPath': seatShiftsPath,
        'attendanceEventsPath': attendanceEventsPath,
        'finalStudentSummaryPath': finalStudentSummaryPath,
        'runManifestPath': runManifestPath,
        'logText': logText,
        'logTextPath': logTextPath,
        'baseUrl': baseUrl,
        'run_id': runId,
        'topic': topic,
        'timestamp': timestamp,
      };

  bool get hasFaceIdentityVideo => faceIdentityVideoPath != null;
  bool get hasFaceIdentityCsv => faceIdentityCsvPath != null;
  bool get hasAttentionVideo => attentionVideoPath != null;
  bool get hasAttendanceCsv => attendanceCsvPath != null;
  bool get hasActivityVideo => activityVideoPath != null;
  bool get hasActivityCsv => activityCsvPath != null;
  bool get hasSpeechVideo => speechVideoPath != null;
  bool get hasSpeechCsv => speechCsvPath != null;
  bool get hasSeatMapPng => seatMapPngPath != null;
  bool get hasSeatMapJson => seatMapJsonPath != null;
  bool get hasSeatingTimeline => seatingTimelinePath != null;
  bool get hasAttendanceEvents => attendanceEventsPath != null;
  bool get hasSeatShifts => seatShiftsPath != null;
  bool get hasFinalStudentSummary => finalStudentSummaryPath != null;
  bool get hasVisualSpeakingCsv => visualSpeakingCsvPath != null;

  String? get faceIdentityVideoUrl => _resolveFileUrl(faceIdentityVideoPath);
  String? get faceIdentityCsvUrl => _resolveFileUrl(faceIdentityCsvPath);
  String? get attentionVideoUrl => _resolveFileUrl(attentionVideoPath);
  String? get attendanceCsvUrl => _resolveFileUrl(attendanceCsvPath);
  String? get attendancePresenceUrl => _resolveFileUrl(attendancePresencePath);
  String? get attentionMetricsUrl => _resolveFileUrl(attentionMetricsPath);
  String? get activityVideoUrl => _resolveFileUrl(activityVideoPath);
  String? get activityCsvUrl => _resolveFileUrl(activityCsvPath);
  String? get deviceUseUrl => _resolveFileUrl(deviceUsePath);
  String? get handRaiseUrl => _resolveFileUrl(handRaisePath);
  String? get noteTakingUrl => _resolveFileUrl(noteTakingPath);
  String? get visualSpeakingVideoUrl =>
      _resolveFileUrl(visualSpeakingVideoPath);
  String? get visualSpeakingCsvUrl => _resolveFileUrl(visualSpeakingCsvPath);
  String? get lipReadingTranscriptUrl =>
      _resolveFileUrl(lipReadingTranscriptPath);
  String? get speechTopicClassificationUrl =>
      _resolveFileUrl(speechTopicClassificationPath);
  String? get speechVideoUrl => _resolveFileUrl(speechVideoPath);
  String? get speechCsvUrl => _resolveFileUrl(speechCsvPath);
  String? get seatMapPngUrl => _resolveFileUrl(seatMapPngPath);
  String? get seatMapJsonUrl => _resolveFileUrl(seatMapJsonPath);
  String? get seatingTimelineUrl => _resolveFileUrl(seatingTimelinePath);
  String? get seatShiftsUrl => _resolveFileUrl(seatShiftsPath);
  String? get attendanceEventsUrl => _resolveFileUrl(attendanceEventsPath);
  String? get finalStudentSummaryUrl =>
      _resolveFileUrl(finalStudentSummaryPath);
  String? get runManifestUrl => _resolveFileUrl(runManifestPath);
  String? get pipelineLogUrl => _resolveFileUrl(logTextPath);

  String? _resolveFileUrl(String? path) {
    if (path == null || path.isEmpty) return null;
    return '$baseUrl/api/files?path=${Uri.encodeQueryComponent(path)}';
  }

  bool get isSuccess => logText.toLowerCase().contains('pipeline complete');
}

class EmbeddingJobResponse {
  final String jobId;
  final String status;
  final int progress;
  final String? errorMessage;
  final List<String> logs;

  EmbeddingJobResponse({
    required this.jobId,
    required this.status,
    required this.progress,
    this.errorMessage,
    required this.logs,
  });

  factory EmbeddingJobResponse.fromJson(Map<String, dynamic> json) {
    return EmbeddingJobResponse(
      jobId: json['job_id'] ?? '',
      status: json['status'] ?? 'queued',
      progress: json['progress'] ?? 0,
      errorMessage: json['error_message'],
      logs: List<String>.from(json['logs'] ?? []),
    );
  }
}

class AttendanceSummaryResponse {
  final String date;
  final String subjectId;
  final int totalStudents;
  final int presentCount;
  final int absentCount;
  final double averageAttention;
  final int phoneUsageCount;
  final List<dynamic> records;

  AttendanceSummaryResponse({
    required this.date,
    required this.subjectId,
    required this.totalStudents,
    required this.presentCount,
    required this.absentCount,
    required this.averageAttention,
    required this.phoneUsageCount,
    required this.records,
  });
}
