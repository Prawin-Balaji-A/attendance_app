import 'dart:async';
import 'package:flutter/material.dart';
import 'package:mjpeg_view/mjpeg_view.dart';

import '../services/api_service.dart';
import '../utils/app_colors.dart';

class MarkAttendanceScreen extends StatefulWidget {
  const MarkAttendanceScreen({super.key});

  @override
  State<MarkAttendanceScreen> createState() => _MarkAttendanceScreenState();
}

class _MarkAttendanceScreenState extends State<MarkAttendanceScreen> {
  bool isScanning = true;

  int facesDetected = 0;
  int knownCount = 0;
  int unknownCount = 0;

  String statusMessage = 'Auto live scan running';
  String lastUpdated = '';
  String currentCameraMode = 'normal';

  List<dynamic> detectedFaces = [];

  Timer? refreshTimer;

  @override
  void initState() {
    super.initState();
    startPolling();
  }

  @override
  void dispose() {
    refreshTimer?.cancel();
    super.dispose();
  }



  void startPolling() {
    refreshTimer?.cancel();

    refreshTimer = Timer.periodic(
      const Duration(seconds: 2),
          (_) async {
        final result = await ApiService.getLiveResults();

        if (!mounted) return;

        setState(() {
          statusMessage = result['message'] ?? 'Auto live scan running';
          facesDetected = result['faces_detected'] ?? 0;
          knownCount = result['known_count'] ?? 0;
          unknownCount = result['unknown_count'] ?? 0;
          currentCameraMode = result['camera_mode'] ?? 'normal';
          lastUpdated = result['last_updated'] ?? '';
          detectedFaces = result['results'] ?? [];
        });
      },
    );
  }



  void _openFullScreenCamera() {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => Scaffold(
          backgroundColor: Colors.black,
          body: Stack(
            children: [
              Center(
                child: InteractiveViewer(
                  panEnabled: true,
                  minScale: 1.0,
                  maxScale: 5.0,
                  child: RotatedBox(
                    quarterTurns: 1,
                    child: MjpegView(
                      uri: ApiService.videoFeedUrl,
                      fit: BoxFit.cover,
                    ),
                  ),
                ),
              ),
              Positioned(
                top: 50,
                right: 20,
                child: Container(
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    shape: BoxShape.circle,
                  ),
                  child: IconButton(
                    icon: const Icon(Icons.close_rounded, color: Colors.white, size: 28),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget cameraPreview() {
    return GestureDetector(
      onTap: _openFullScreenCamera,
      child: Stack(
        children: [
          Container(
            height: 260,
            width: double.infinity,
            clipBehavior: Clip.antiAlias,
            decoration: BoxDecoration(
              color: AppColors.primaryDark,
              borderRadius: BorderRadius.circular(28),
            ),
            child: MjpegView(
              uri: ApiService.videoFeedUrl,
              fit: BoxFit.contain,
            ),
          ),
          Positioned(
            right: 12,
            bottom: 12,
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Icon(
                Icons.fullscreen_rounded,
                color: Colors.white,
                size: 24,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget statusCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        color: AppColors.white,
        borderRadius: BorderRadius.circular(28),
      ),
      child: Column(
        children: [
          const Icon(
            Icons.sensors_rounded,
            size: 52,
            color: Colors.green,
          ),
          const SizedBox(height: 14),

          Text(
            statusMessage,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: AppColors.primaryDark,
            ),
          ),
          const SizedBox(height: 10),
          Text(
            'Faces detected: $facesDetected',
            style: const TextStyle(
              fontSize: 17,
              color: AppColors.textSecondary,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Known: $knownCount  |  Unknown: $unknownCount',
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 15,
              color: AppColors.textSecondary,
            ),
          ),
          const SizedBox(height: 10),
          Text(
            lastUpdated.isEmpty ? 'Last updated: --' : 'Last updated: $lastUpdated',
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 15,
              color: AppColors.textSecondary,
            ),
          ),
        ],
      ),
    );
  }



  Widget detectedFaceCard(dynamic face) {
    final bool known = face['known'] == true;
    final String status = face['status'] ?? 'unknown';

    final String displayName = known
        ? (face['name'] ?? 'Known')
        : 'Unknown';

    final String groupText = known ? (face['group'] ?? '') : '';
    final String message = face['message'] ?? '';

    final Color bgColor = known
        ? Colors.green.shade100
        : Colors.red.shade100;

    final Color iconColor = known
        ? Colors.green
        : Colors.red;

    final IconData icon = known
        ? Icons.verified_user_rounded
        : Icons.warning_amber_rounded;

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.white,
        borderRadius: BorderRadius.circular(22),
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 28,
            backgroundColor: bgColor,
            child: Icon(icon, color: iconColor),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  displayName,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: AppColors.primaryDark,
                  ),
                ),
                if (groupText.isNotEmpty) ...[
                  const SizedBox(height: 5),
                  Text(
                    groupText,
                    style: const TextStyle(color: AppColors.textSecondary),
                  ),
                ],
                const SizedBox(height: 5),
                Text(
                  message,
                  style: TextStyle(
                    color: iconColor,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget detectedFacesSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Detected Faces',
          style: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.bold,
            color: AppColors.primaryDark,
          ),
        ),
        const SizedBox(height: 16),
        if (detectedFaces.isEmpty)
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(30),
            decoration: BoxDecoration(
              color: AppColors.white,
              borderRadius: BorderRadius.circular(22),
            ),
            child: const Center(
              child: Text(
                'No faces detected',
                style: TextStyle(
                  fontSize: 16,
                  color: AppColors.textSecondary,
                ),
              ),
            ),
          ),
        ...detectedFaces.map((face) => detectedFaceCard(face)),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(18, 18, 18, 110),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          cameraPreview(),
          const SizedBox(height: 20),
          statusCard(),
          const SizedBox(height: 28),
          detectedFacesSection(),
        ],
      ),
    );
  }
}