import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

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
    'Look Straight',
    'Turn Slightly Left',
    'Turn Slightly Right',
    'Look Slightly Up',
    'Look Slightly Down',
    'Look Straight (Smile/Vary Expression)',
    'Turn Further Left',
    'Turn Further Right',
    'Tilt Head Left',
    'Tilt Head Right',
  ];

  String instruction = 'Step 1/10 : Look Straight';

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
      'Training face model...\nStep ${currentStep + 1}/10';
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
              '10 Face Angles Saved.',
        );

        setState(() {
          currentStep = 0;
          selectedImage = null;
          instruction = 'Step 1/10 : Look Straight';
        });

        nameController.clear();
        userIdController.clear();
      }
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
          ? Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(
            Icons.face_retouching_natural_rounded,
            size: 70,
            color: AppColors.white,
          ),
          const SizedBox(height: 14),
          Text(
            'Face Capture\n${currentStep + 1}/10',
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: AppColors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      )
          : Image.file(
        selectedImage!,
        fit: BoxFit.cover,
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