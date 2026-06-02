import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../utils/app_colors.dart';

class GroupDetailsScreen extends StatefulWidget {
  final String groupName;

  const GroupDetailsScreen({
    super.key,
    required this.groupName,
  });

  @override
  State<GroupDetailsScreen> createState() => _GroupDetailsScreenState();
}

class _GroupDetailsScreenState extends State<GroupDetailsScreen> {
  bool isLoading = true;
  bool showPresent = true;

  int totalMembers = 0;
  int presentCount = 0;
  int absentCount = 0;

  List<Map<String, dynamic>> presentUsers = [];
  List<Map<String, dynamic>> absentUsers = [];

  @override
  void initState() {
    super.initState();
    loadDetails();
  }

  Future<void> loadDetails() async {
    setState(() {
      isLoading = true;
    });

    final result = await ApiService.getGroupDetails(widget.groupName);

    if (!mounted) return;

    if (result['success'] == true) {
      final group = result['group'] ?? {};

      setState(() {
        totalMembers = group['totalMembers'] ?? 0;
        presentCount = group['presentCount'] ?? 0;
        absentCount = group['absentCount'] ?? 0;

        presentUsers =
        List<Map<String, dynamic>>.from(group['presentUsers'] ?? []);
        absentUsers =
        List<Map<String, dynamic>>.from(group['absentUsers'] ?? []);

        isLoading = false;
      });
    } else {
      setState(() {
        totalMembers = 0;
        presentCount = 0;
        absentCount = 0;
        presentUsers = [];
        absentUsers = [];
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final selectedList = showPresent ? presentUsers : absentUsers;

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: AppColors.primaryDark,
        foregroundColor: AppColors.white,
        title: Text(widget.groupName),
        centerTitle: true,
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
        onRefresh: loadDetails,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(18),
          child: Column(
            children: [
              headerCard(totalMembers, presentCount, absentCount),
              const SizedBox(height: 18),
              switchButtons(),
              const SizedBox(height: 18),
              if (selectedList.isEmpty)
                Padding(
                  padding: const EdgeInsets.all(30),
                  child: Text(
                    showPresent
                        ? 'No present users yet'
                        : 'No absent users',
                    style: const TextStyle(color: AppColors.grey),
                  ),
                ),
              ListView.builder(
                itemCount: selectedList.length,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemBuilder: (context, index) {
                  return userCard(selectedList[index], showPresent);
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget headerCard(int total, int present, int absent) {
    int percent = total == 0 ? 0 : ((present / total) * 100).round();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF0F172A), Color(0xFF2563EB)],
        ),
        borderRadius: BorderRadius.circular(28),
      ),
      child: Column(
        children: [
          Text(
            widget.groupName,
            style: const TextStyle(
              color: AppColors.white,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 10),
          Text(
            '$percent% Attendance Today',
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 15,
            ),
          ),
          const SizedBox(height: 18),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              statItem('Total', total.toString()),
              statItem('Present', present.toString()),
              statItem('Absent', absent.toString()),
            ],
          ),
        ],
      ),
    );
  }

  Widget statItem(String title, String value) {
    return Column(
      children: [
        Text(
          value,
          style: const TextStyle(
            color: AppColors.white,
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
        Text(
          title,
          style: const TextStyle(color: Colors.white70),
        ),
      ],
    );
  }

  Widget switchButtons() {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton(
            onPressed: () {
              setState(() {
                showPresent = true;
              });
            },
            style: ElevatedButton.styleFrom(
              backgroundColor:
              showPresent ? AppColors.green : AppColors.white,
              foregroundColor:
              showPresent ? AppColors.white : AppColors.green,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
            ),
            child: const Text('Present'),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton(
            onPressed: () {
              setState(() {
                showPresent = false;
              });
            },
            style: ElevatedButton.styleFrom(
              backgroundColor:
              !showPresent ? AppColors.red : AppColors.white,
              foregroundColor:
              !showPresent ? AppColors.white : AppColors.red,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
            ),
            child: const Text('Absent'),
          ),
        ),
      ],
    );
  }

  Widget userCard(Map<String, dynamic> user, bool present) {
    String name = (user['name'] ?? 'Unknown').toString();
    String userId = (user['user_id'] ?? '').toString();
    String group = (user['group'] ?? '').toString();

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.white,
        borderRadius: BorderRadius.circular(22),
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 28,
            backgroundColor:
            present ? const Color(0xFFE8F8EF) : const Color(0xFFFFEEEE),
            child: Icon(
              present ? Icons.check_circle_rounded : Icons.cancel_rounded,
              color: present ? AppColors.green : AppColors.red,
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  name,
                  style: const TextStyle(
                    color: AppColors.primaryDark,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  '$userId • $group',
                  style: const TextStyle(
                    color: AppColors.grey,
                    fontSize: 13,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}