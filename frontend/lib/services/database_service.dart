import '../models/subject.dart';
import 'api_service.dart';

/// Local backend-backed storage for subjects and enrollments.
class DatabaseService {
  static final DatabaseService _instance = DatabaseService._internal();
  factory DatabaseService() => _instance;
  DatabaseService._internal();

  Future<List<Subject>> getSubjects() async {
    try {
      return await ApiService().getSubjects();
    } catch (e) {
      print('Database error getting subjects: $e');
      return [];
    }
  }

  Future<Subject?> createSubject({
    required String name,
    required String code,
    required String description,
    required int iconIndex,
    required int totalStudents,
  }) async {
    try {
      return await ApiService().createSubject(
        name: name,
        code: code,
        description: description,
        iconIndex: iconIndex,
        totalStudents: totalStudents,
      );
    } catch (e) {
      print('Database error creating subject: $e');
      return null;
    }
  }

  Future<bool> deleteSubject(String subjectId) async {
    try {
      await ApiService().deleteSubject(subjectId);
      return true;
    } catch (e) {
      print('Database error deleting subject: $e');
      return false;
    }
  }

  Future<List<Map<String, dynamic>>> getEnrolledStudents(
      String subjectId) async {
    return [];
  }
}
