import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = 'http://192.168.1.2:8000';

  static const String videoFeedUrl = '$baseUrl/debug-face-feed';

  static String get cameraFrameUrl =>
      '$baseUrl/camera-frame?t=${DateTime.now().millisecondsSinceEpoch}';

  static Future<Map<String, dynamic>> _safeDecode(http.Response response) async {
    try {
      if (response.body.isEmpty) {
        return {'success': false, 'message': 'Empty response from server'};
      }

      final decoded = jsonDecode(response.body);

      if (decoded is Map<String, dynamic>) {
        return decoded;
      }

      return {'success': false, 'message': 'Invalid response format'};
    } catch (e) {
      return {'success': false, 'message': 'JSON decode error: $e'};
    }
  }

  static Future<Map<String, dynamic>> registerFromImage({
    required String name,
    required String userId,
    required String group,
    required String imagePath,
  }) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/register-image'),
      );

      request.fields['name'] = name.trim();
      request.fields['user_id'] = userId.trim();
      request.fields['group'] = group.trim();

      request.files.add(
        await http.MultipartFile.fromPath('image', imagePath),
      );

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      return await _safeDecode(response);
    } catch (e) {
      return {'success': false, 'message': 'Connection error: $e'};
    }
  }

  static Future<Map<String, dynamic>> registerFromVideo({
    required String name,
    required String userId,
    required String group,
    required String videoPath,
  }) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/register-video'),
      );

      request.fields['name'] = name.trim();
      request.fields['user_id'] = userId.trim();
      request.fields['group'] = group.trim();

      request.files.add(
        await http.MultipartFile.fromPath('video', videoPath),
      );

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      return await _safeDecode(response);
    } catch (e) {
      return {'success': false, 'message': 'Connection error: $e'};
    }
  }

  static Future<Map<String, dynamic>> registerLive({
    required String name,
    required String userId,
    required String group,
  }) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/register-live'),
      );

      request.fields['name'] = name.trim();
      request.fields['user_id'] = userId.trim();
      request.fields['group'] = group.trim();

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      return await _safeDecode(response);
    } catch (e) {
      return {'success': false, 'message': 'Connection error: $e'};
    }
  }


  static Future<Map<String, dynamic>> getUsers() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/users'));
      final data = await _safeDecode(response);
      data['users'] ??= [];
      return data;
    } catch (e) {
      return {'success': false, 'message': 'Connection error: $e', 'users': []};
    }
  }

  static Future<Map<String, dynamic>> getGroups() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/groups'));
      final data = await _safeDecode(response);
      data['groups'] ??= [];
      return data;
    } catch (e) {
      return {'success': false, 'message': 'Connection error: $e', 'groups': []};
    }
  }

  static Future<Map<String, dynamic>> getGroupDetails(String groupName) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/groups/${Uri.encodeComponent(groupName)}'),
      );
      return await _safeDecode(response);
    } catch (e) {
      return {'success': false, 'message': 'Connection error: $e'};
    }
  }

  static Future<Map<String, dynamic>> getAttendance() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/attendance'));
      final data = await _safeDecode(response);
      data['attendance'] ??= [];
      return data;
    } catch (e) {
      return {
        'success': false,
        'message': 'Connection error: $e',
        'attendance': [],
      };
    }
  }

  static Future<Map<String, dynamic>> startLiveScan() async {
    try {
      final response = await http.post(Uri.parse('$baseUrl/start-live-scan'));
      return await _safeDecode(response);
    } catch (e) {
      return {'success': false, 'message': 'Connection error: $e'};
    }
  }

  static Future<Map<String, dynamic>> setCameraMode(String mode) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/set-camera-mode'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'mode': mode}),
      );
      return await _safeDecode(response);
    } catch (e) {
      return {'success': false, 'message': 'Connection error: $e'};
    }
  }

  static Future<Map<String, dynamic>> stopLiveScan() async {
    try {
      final response = await http.post(Uri.parse('$baseUrl/stop-live-scan'));
      return await _safeDecode(response);
    } catch (e) {
      return {'success': false, 'message': 'Connection error: $e'};
    }
  }

  static Future<Map<String, dynamic>> getLiveResults() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/live-results'));
      final data = await _safeDecode(response);

      data['faces_detected'] ??= 0;
      data['known_count'] ??= 0;
      data['unknown_count'] ??= 0;
      data['too_far_count'] ??= 0;
      data['camera_mode'] ??= 'normal';
      data['last_updated'] ??= '';
      data['results'] ??= [];

      return data;
    } catch (e) {
      return {
        'success': false,
        'message': 'Connection error: $e',
        'faces_detected': 0,
        'known_count': 0,
        'unknown_count': 0,
        'too_far_count': 0,
        'camera_mode': 'normal',
        'last_updated': '',
        'results': [],
      };
    }
  }

  static Future<Map<String, dynamic>> deleteUser(String userId) async {
    try {
      final response = await http.delete(
        Uri.parse('$baseUrl/users/${Uri.encodeComponent(userId)}'),
      );

      return await _safeDecode(response);
    } catch (e) {
      return {'success': false, 'message': 'Connection error: $e'};
    }
  }
}