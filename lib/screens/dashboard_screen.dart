import 'package:flutter/material.dart';
import '../utils/app_colors.dart';
import '../services/api_service.dart';
import 'register_face_screen.dart';
import 'mark_attendance_screen.dart';
import 'manage_users_screen.dart';
import 'group_details_screen.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  int selectedIndex = 0;

  final GlobalKey<_PremiumHomePageState> homeKey = GlobalKey();

  final List<String> titles = const [
    'Admin Dashboard',
    'Register User',
    'Live Attendance',
    'Users',
  ];

  void changeTab(int index) {
    setState(() {
      selectedIndex = index;
    });

    if (index == 0) {
      Future.delayed(const Duration(milliseconds: 200), () {
        homeKey.currentState?.loadDashboard();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      PremiumHomePage(key: homeKey),
      const RegisterFaceScreen(),
      const MarkAttendanceScreen(),
      const ManageUsersScreen(),
    ];

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: AppColors.primaryDark,
        foregroundColor: AppColors.white,
        title: Text(titles[selectedIndex]),
        centerTitle: true,
        elevation: 0,
        actions: [
          if (selectedIndex == 0)
            IconButton(
              onPressed: () {
                homeKey.currentState?.loadDashboard();
              },
              icon: const Icon(Icons.refresh_rounded),
            ),
        ],
      ),
      body: pages[selectedIndex],
      bottomNavigationBar: Container(
        margin: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppColors.white,
          borderRadius: BorderRadius.circular(30),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.14),
              blurRadius: 20,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(30),
          child: BottomNavigationBar(
            currentIndex: selectedIndex,
            onTap: changeTab,
            type: BottomNavigationBarType.fixed,
            backgroundColor: AppColors.white,
            selectedItemColor: AppColors.primary,
            unselectedItemColor: AppColors.grey,
            items: const [
              BottomNavigationBarItem(
                icon: Icon(Icons.dashboard_rounded),
                label: 'Home',
              ),
              BottomNavigationBarItem(
                icon: Icon(Icons.person_add_alt_1_rounded),
                label: 'Register',
              ),
              BottomNavigationBarItem(
                icon: Icon(Icons.sensors_rounded),
                label: 'Live',
              ),
              BottomNavigationBarItem(
                icon: Icon(Icons.groups_rounded),
                label: 'Users',
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class PremiumHomePage extends StatefulWidget {
  const PremiumHomePage({super.key});

  @override
  State<PremiumHomePage> createState() => _PremiumHomePageState();
}

class _PremiumHomePageState extends State<PremiumHomePage> {
  bool isLoading = true;
  List<dynamic> groups = [];
  List<dynamic> attendance = [];

  @override
  void initState() {
    super.initState();
    loadDashboard();
  }

  Future<void> loadDashboard() async {
    if (mounted) {
      setState(() {
        isLoading = true;
      });
    }

    final groupResult = await ApiService.getGroups();
    final attendanceResult = await ApiService.getAttendance();

    if (!mounted) return;

    setState(() {
      groups = groupResult['groups'] ?? [];
      attendance =
          ((attendanceResult['attendance'] ?? []) as List).take(5).toList();
      isLoading = false;
    });
  }

  int safeInt(dynamic value) {
    if (value is int) return value;
    if (value is String) return int.tryParse(value) ?? 0;
    return 0;
  }

  @override
  Widget build(BuildContext context) {
    int totalGroups = groups.length;

    int totalUsers = groups.fold(
      0,
          (sum, g) => sum + safeInt(g['totalMembers']),
    );

    int totalPresent = groups.fold(
      0,
          (sum, g) => sum + safeInt(g['presentCount']),
    );

    int totalAbsent = groups.fold(
      0,
          (sum, g) => sum + safeInt(g['absentCount']),
    );

    int overallPercentage =
    totalUsers == 0 ? 0 : ((totalPresent / totalUsers) * 100).round();

    if (isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    return RefreshIndicator(
      onRefresh: loadDashboard,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(18, 18, 18, 110),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            premiumHeader(overallPercentage),
            const SizedBox(height: 18),

            Row(
              children: [
                topMetricCard('Groups', totalGroups.toString(),
                    Icons.grid_view_rounded, AppColors.primary),
                const SizedBox(width: 12),
                topMetricCard('Users', totalUsers.toString(),
                    Icons.people_alt_rounded, Colors.deepPurple),
              ],
            ),

            const SizedBox(height: 12),

            Row(
              children: [
                topMetricCard('Present', totalPresent.toString(),
                    Icons.verified_rounded, AppColors.green),
                const SizedBox(width: 12),
                topMetricCard('Absent', totalAbsent.toString(),
                    Icons.cancel_rounded, AppColors.red),
              ],
            ),

            const SizedBox(height: 24),

            const Text(
              'Groups Monitoring',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: AppColors.primaryDark,
              ),
            ),

            const SizedBox(height: 14),

            if (groups.isEmpty)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(30),
                  child: Text(
                    'No registered users yet',
                    style: TextStyle(color: AppColors.grey),
                  ),
                ),
              )
            else
              ListView.builder(
                itemCount: groups.length,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemBuilder: (context, index) {
                  return groupCard(groups[index]);
                },
              ),

            const SizedBox(height: 22),

            const Text(
              'Recent Attendance',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: AppColors.primaryDark,
              ),
            ),

            const SizedBox(height: 14),

            if (attendance.isEmpty)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(25),
                  child: Text(
                    'No attendance marked yet',
                    style: TextStyle(color: AppColors.grey),
                  ),
                ),
              )
            else
              ListView.builder(
                itemCount: attendance.length,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemBuilder: (context, index) {
                  return attendanceCard(attendance[index]);
                },
              ),
          ],
        ),
      ),
    );
  }

  Widget premiumHeader(int percentage) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF0F172A), Color(0xFF2563EB)],
        ),
        borderRadius: BorderRadius.circular(30),
      ),
      child: Row(
        children: [
          const Expanded(
            child: Text(
              'Live Attendance Monitor\nReal-time admin control panel',
              style: TextStyle(
                color: AppColors.white,
                fontSize: 22,
                fontWeight: FontWeight.bold,
                height: 1.4,
              ),
            ),
          ),
          CircleAvatar(
            radius: 42,
            backgroundColor: Colors.white24,
            child: Text(
              '$percentage%',
              style: const TextStyle(
                color: AppColors.white,
                fontWeight: FontWeight.bold,
                fontSize: 22,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget topMetricCard(String title, String value, IconData icon, Color color) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: AppColors.white,
          borderRadius: BorderRadius.circular(24),
        ),
        child: Row(
          children: [
            CircleAvatar(
              backgroundColor: color.withOpacity(0.12),
              child: Icon(icon, color: color),
            ),
            const SizedBox(width: 12),
            Flexible(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    value,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    title,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(color: AppColors.grey),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget groupCard(dynamic group) {
    String groupName = (group['groupName'] ?? 'Unknown Group').toString();

    if (groupName.trim().isEmpty) {
      groupName = 'Unknown Group';
    }

    int total = safeInt(group['totalMembers']);
    int present = safeInt(group['presentCount']);
    int absent = safeInt(group['absentCount']);
    int percent = total == 0 ? 0 : ((present / total) * 100).round();

    return InkWell(
      borderRadius: BorderRadius.circular(26),
      onTap: () async {
        await Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => GroupDetailsScreen(groupName: groupName),
          ),
        );

        loadDashboard();
      },
      child: Container(
        margin: const EdgeInsets.only(bottom: 16),
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: AppColors.white,
          borderRadius: BorderRadius.circular(26),
        ),
        child: Column(
          children: [
            Row(
              children: [
                CircleAvatar(
                  radius: 28,
                  backgroundColor: AppColors.primary.withOpacity(0.12),
                  child: Text(
                    groupName[0].toUpperCase(),
                    style: const TextStyle(
                      color: AppColors.primary,
                      fontWeight: FontWeight.bold,
                      fontSize: 22,
                    ),
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Text(
                    groupName,
                    style: const TextStyle(
                      color: AppColors.primaryDark,
                      fontSize: 19,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      '$percent%',
                      style: const TextStyle(
                        color: AppColors.green,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 4),
                    const Icon(
                      Icons.arrow_forward_ios_rounded,
                      size: 15,
                      color: AppColors.grey,
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 14),
            LinearProgressIndicator(value: percent / 100),
            const SizedBox(height: 14),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                smallInfo('Members', total.toString(), AppColors.primary),
                smallInfo('Present', present.toString(), AppColors.green),
                smallInfo('Absent', absent.toString(), AppColors.red),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget attendanceCard(dynamic item) {
    String name = (item['Name'] ?? 'Unknown').toString();
    String userId = (item['UserID'] ?? '').toString();
    String group = (item['Group'] ?? '').toString();

    String date = (item['Date'] ?? '').toString();
    String time = (item['Time'] ?? '').toString();

    if (date.isEmpty || time.isEmpty) {
      String timestamp = (item['Timestamp'] ?? '').toString();

      if (timestamp.contains(' ')) {
        final parts = timestamp.split(' ');
        date = parts[0];
        time = parts.length > 1 ? parts[1] : '';
      }
    }

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.white,
        borderRadius: BorderRadius.circular(22),
      ),
      child: Row(
        children: [
          const CircleAvatar(
            radius: 28,
            backgroundColor: Color(0xFFE8F8EF),
            child: Icon(Icons.check_circle_rounded, color: AppColors.green),
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
                const SizedBox(height: 4),
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
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                date,
                style: const TextStyle(
                  color: AppColors.primaryDark,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                time,
                style: const TextStyle(
                  color: AppColors.green,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget smallInfo(String title, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: TextStyle(
            color: color,
            fontWeight: FontWeight.bold,
            fontSize: 17,
          ),
        ),
        Text(
          title,
          style: const TextStyle(
            color: AppColors.grey,
            fontSize: 12,
          ),
        ),
      ],
    );
  }
}