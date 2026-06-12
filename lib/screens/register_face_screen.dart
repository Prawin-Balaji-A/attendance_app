import 'dart:io';
import 'dart:async';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:mjpeg_view/mjpeg_view.dart';

import '../services/api_service.dart';
import '../utils/app_colors.dart';

class RegisterFaceScreen extends StatefulWidget {
  const RegisterFaceScreen({super.key});

  @override
  State<RegisterFaceScreen> createState() => _RegisterFaceScreenState();
}

class _RegisterFaceScreenState extends State<RegisterFaceScreen> {
  final nameController = TextEditingController();
  final userIdController = TextEditingController();

  final ImagePicker picker = ImagePicker();

  bool isLoading = false;
  File? selectedImage;

  String selectedGroup = 'Software Intern';

  int currentStep = 0;

  final List<String> faceSteps = [
    'Look Straight (Include Shoulders)',
    'Turn Left (Turn Body & Face)',
    'Turn Right (Turn Body & Face)',
    'Look Slightly Up',
    'Look Slightly Down',
    'Look Straight (Vary Expression)',
    'Turn Further Left',
    'Turn Further Right',
    'Step Back slightly',
    'Walk closer slightly',
  ];

  String instruction = 'Step 1/10 : Look Straight (Include Shoulders)';

  bool validateInputs() {
    if (nameController.text.trim().isEmpty ||
        userIdController.text.trim().isEmpty) {
      showMessage('Enter name and ID');
      return false;
    }
    return true;
  }

  Future<void> pickImage(ImageSource source) async {
    if (!validateInputs()) return;

    try {
      final XFile? image = await picker.pickImage(
        source: source,
        imageQuality: 90,
      );

      if (image == null) return;

      setState(() {
        selectedImage = File(image.path);
      });

      await registerFromSelectedImage(image.path);
    } catch (e) {
      showMessage('Image error: $e');
    }
  }

  Future<void> registerFromSelectedImage(String imagePath) async {
    setState(() {
      isLoading = true;
      instruction =
      'Learning Face & Body features...\nStep ${currentStep + 1}/10';
    });

    final result = await ApiService.registerFromImage(
      name: nameController.text.trim(),
      userId: userIdController.text.trim(),
      group: selectedGroup,
      imagePath: imagePath,
    );

    if (!mounted) return;

    setState(() {
      isLoading = false;
    });

    showMessage(result['message'] ?? 'No response');

    if (result['success'] == true) {
      if (currentStep < faceSteps.length - 1) {
        setState(() {
          currentStep++;

          instruction =
          'Step ${currentStep + 1}/10 : ${faceSteps[currentStep]}';
        });
      } else {
        showMessage(
          'Registration Completed Successfully.\n'
              'Face & Body Features Saved.',
        );

        setState(() {
          currentStep = 0;
          selectedImage = null;
          instruction = 'Step 1/10 : Look Straight (Include Shoulders)';
        });

        nameController.clear();
        userIdController.clear();
      }
    }
  }

  Future<void> registerAutoCapture() async {
    if (!validateInputs()) return;

    // Show instructional dialog before opening camera
    final bool? proceed = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Important: Stand Back'),
          content: const Text(
            'For highest accuracy, please stand 1-2 meters away so your face AND upper body are visible.\n\n'
            'When recording starts, slowly turn your head and body left and right.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () => Navigator.of(context).pop(true),
              child: const Text('Got it, record!'),
            ),
          ],
        );
      },
    );

    if (proceed != true) return;

    try {
      final XFile? video = await picker.pickVideo(
        source: ImageSource.camera,
        maxDuration: const Duration(seconds: 10),
      );

      if (video == null) return;

      setState(() {
        isLoading = true;
        instruction = 'Learning Face & Body features...\nPlease wait.';
      });

      final result = await ApiService.registerFromVideo(
        name: nameController.text.trim(),
        userId: userIdController.text.trim(),
        group: selectedGroup,
        videoPath: video.path,
      );

    if (!mounted) return;

    setState(() {
      isLoading = false;
    });

    showMessage(result['message'] ?? 'No response');

      if (result['success'] == true) {
        setState(() {
          currentStep = 0;
          selectedImage = null;
          instruction = 'Step 1/10 : Look Straight (Include Shoulders)';
        });
        nameController.clear();
        userIdController.clear();
      } else {
        setState(() {
          instruction = 'Step ${currentStep + 1}/10 : ${faceSteps[currentStep]}';
        });
      }
    } catch (e) {
      showMessage('Video error: $e');
      setState(() {
        isLoading = false;
        instruction = 'Step ${currentStep + 1}/10 : ${faceSteps[currentStep]}';
      });
    }
  }

  Future<void> registerLiveCamera() async {
    if (!validateInputs()) return;

    final bool? proceed = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Register from Pi Camera'),
          content: const Text(
            'Please stand in front of the main Raspberry Pi camera.\n\n'
            'The system will automatically capture multiple frames of your face over the next 5 seconds to ensure perfect accuracy.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () => Navigator.of(context).pop(true),
              child: const Text('Start Live Registration'),
            ),
          ],
        );
      },
    );

    if (proceed != true) return;

    int secondsLeft = 5;
    setState(() {
      isLoading = true;
      instruction = 'Capturing live from Pi Camera...\n$secondsLeft seconds remaining';
    });

    Timer? countdownTimer;
    countdownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      setState(() {
        if (secondsLeft > 1) {
          secondsLeft--;
          instruction = 'Capturing live from Pi Camera...\n$secondsLeft seconds remaining';
        } else {
          instruction = 'Finalizing registration...';
        }
      });
    });

    try {
      final result = await ApiService.registerLive(
        name: nameController.text.trim(),
        userId: userIdController.text.trim(),
        group: selectedGroup,
      );

      countdownTimer.cancel();
      if (!mounted) return;

      setState(() {
        isLoading = false;
        instruction = 'Step 1/10 : Look Straight (Include Shoulders)';
      });

      showMessage(result['message'] ?? 'No response');

      if (result['success'] == true) {
        nameController.clear();
        userIdController.clear();
      }
    } catch (e) {
      countdownTimer?.cancel();
      showMessage('Registration error: $e');
      setState(() {
        isLoading = false;
        instruction = 'Step 1/10 : Look Straight (Include Shoulders)';
      });
    }
  }

  void showMessage(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg)),
    );
  }

  @override
  void dispose() {
    nameController.dispose();
    userIdController.dispose();
    super.dispose();
  }

  Widget imagePreview() {
    return Container(
      height: 260,
      width: double.infinity,
      clipBehavior: Clip.antiAlias,
      decoration: BoxDecoration(
        color: AppColors.primaryDark,
        borderRadius: BorderRadius.circular(28),
      ),
      child: selectedImage == null
          ? MjpegView(
              uri: ApiService.videoFeedUrl,
              fit: BoxFit.contain,
            )
          : Image.file(
              selectedImage!,
              fit: BoxFit.contain,
            ),
    );
  }

  Widget inputField(
      TextEditingController controller,
      String hint,
      IconData icon,
      ) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        color: AppColors.white,
        borderRadius: BorderRadius.circular(18),
      ),
      child: TextField(
        controller: controller,
        decoration: InputDecoration(
          hintText: hint,
          prefixIcon: Icon(icon),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.all(18),
        ),
      ),
    );
  }

  Widget groupDropdown() {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: AppColors.white,
        borderRadius: BorderRadius.circular(18),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: selectedGroup,
          isExpanded: true,
          icon: const Icon(Icons.arrow_drop_down),
          items: const [
            DropdownMenuItem(
              value: 'Hardware Fulltime',
              child: Text('Hardware Fulltime'),
            ),
            DropdownMenuItem(
              value: 'Software Fulltime',
              child: Text('Software Fulltime'),
            ),
            DropdownMenuItem(
              value: 'Hardware Intern',
              child: Text('Hardware Intern'),
            ),
            DropdownMenuItem(
              value: 'Software Intern',
              child: Text('Software Intern'),
            ),
          ],
          onChanged: (value) {
            if (value != null) {
              setState(() {
                selectedGroup = value;
              });
            }
          },
        ),
      ),
    );
  }

  Widget actionButton({
    required String text,
    required IconData icon,
    required VoidCallback? onPressed,
    required Color color,
  }) {
    return SizedBox(
      width: double.infinity,
      height: 56,
      child: ElevatedButton.icon(
        onPressed: isLoading ? null : onPressed,
        icon: isLoading
            ? const SizedBox(
          width: 20,
          height: 20,
          child: CircularProgressIndicator(
            strokeWidth: 2,
            color: AppColors.white,
          ),
        )
            : Icon(icon),
        label: Text(
          text,
          style: const TextStyle(
            fontWeight: FontWeight.bold,
          ),
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: color,
          foregroundColor: AppColors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(18),
          ),
        ),
      ),
    );
  }

  Widget progressCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.white,
        borderRadius: BorderRadius.circular(22),
      ),
      child: Column(
        children: [
          Text(
            instruction,
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: AppColors.primaryDark,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 15),
          LinearProgressIndicator(
            value: (currentStep + 1) / 10,
            minHeight: 10,
            borderRadius: BorderRadius.circular(10),
          ),
          const SizedBox(height: 10),
          Text(
            '${currentStep + 1} / 10 Face Angles',
            style: const TextStyle(
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(18, 18, 18, 110),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          imagePreview(),

          const SizedBox(height: 18),

          progressCard(),

          const SizedBox(height: 20),

          inputField(
            nameController,
            'Enter Name',
            Icons.person_rounded,
          ),

          inputField(
            userIdController,
            'Enter Employee ID / Reg No',
            Icons.badge_rounded,
          ),

          groupDropdown(),

          const SizedBox(height: 10),

          actionButton(
            text: isLoading
                ? 'Recording...'
                : 'Auto Capture from Phone (10 Sec)',
            icon: Icons.video_camera_front_rounded,
            color: Colors.green,
            onPressed: registerAutoCapture,
          ),

          const SizedBox(height: 14),

          actionButton(
            text: isLoading
                ? 'Capturing...'
                : 'Register from Pi Camera (Recommended)',
            icon: Icons.camera_indoor_rounded,
            color: Colors.blue,
            onPressed: registerLiveCamera,
          ),

          const SizedBox(height: 14),

          const Center(
            child: Text(
              'OR MANUAL UPLOAD',
              style: TextStyle(
                color: Colors.grey,
                fontWeight: FontWeight.bold,
                fontSize: 12,
              ),
            ),
          ),

          const SizedBox(height: 14),

          actionButton(
            text: isLoading
                ? 'Training...'
                : 'Capture ${currentStep + 1}/10 Using Camera',
            icon: Icons.camera_alt_rounded,
            color: AppColors.primary,
            onPressed: () => pickImage(ImageSource.camera),
          ),

          const SizedBox(height: 14),

          actionButton(
            text: isLoading
                ? 'Training...'
                : 'Upload ${currentStep + 1}/10 From Gallery',
            icon: Icons.photo_library_rounded,
            color: AppColors.primaryDark,
            onPressed: () => pickImage(ImageSource.gallery),
          ),

          const SizedBox(height: 20),

          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.orange.shade50,
              borderRadius: BorderRadius.circular(16),
            ),
            child: const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Registration Tips',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(height: 8),
                Text('• Good lighting'),
                Text('• Face clearly visible'),
                Text('• Remove heavy shadows'),
                Text('• Capture all 10 angles'),
                Text('• Keep camera steady'),
              ],
            ),
          ),
        ],
      ),
    );
  }
}