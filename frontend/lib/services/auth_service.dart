import 'dart:convert';
import 'dart:html' as html;

import '../models/user.dart';
import 'api_service.dart';

/// Local FastAPI-backed authentication service.
class AuthService {
  static final AuthService _instance = AuthService._internal();
  factory AuthService() => _instance;
  AuthService._internal() {
    _restore();
  }

  static const _sessionKey = 'classroom_monitor_local_session';

  User? _currentUser;
  String? _token;

  User? get currentUser => _currentUser;
  bool get isLoggedIn => _currentUser != null && _token != null;
  String? get token => _token;

  Future<String?> login(
      String username, String password, bool asInstructor) async {
    try {
      final result = await ApiService().login(username.trim(), password);
      if (asInstructor && !result.user.isInstructor) {
        return 'This account is not an instructor account.';
      }
      if (!asInstructor && !result.user.isStudent) {
        return 'This account is not a student account.';
      }
      _currentUser = result.user;
      _token = result.token;
      ApiService.authToken = _token;
      _saveSession();
      return null;
    } catch (e) {
      return e.toString().replaceFirst('Exception: ', '');
    }
  }

  Future<void> logout() async {
    await ApiService().logout();
    _currentUser = null;
    _token = null;
    ApiService.authToken = null;
    html.window.localStorage.remove(_sessionKey);
  }

  String scopedKey(String base) {
    final uid = _currentUser?.id ?? 'anonymous';
    return '${base}_$uid';
  }

  void _saveSession() {
    final user = _currentUser;
    final token = _token;
    if (user == null || token == null) return;
    html.window.localStorage[_sessionKey] = json.encode({
      'token': token,
      'user': {
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'role': user.role == UserRole.instructor ? 'instructor' : 'student',
        'department': user.department,
      },
    });
  }

  void _restore() {
    try {
      final raw = html.window.localStorage[_sessionKey];
      if (raw == null || raw.isEmpty) return;
      final data = json.decode(raw) as Map<String, dynamic>;
      _token = data['token']?.toString();
      if (data['user'] is Map<String, dynamic>) {
        _currentUser = userFromJson(data['user'] as Map<String, dynamic>);
      }
      ApiService.authToken = _token;
    } catch (_) {
      _currentUser = null;
      _token = null;
      ApiService.authToken = null;
    }
  }
}
