import 'package:flutter/material.dart';
import '../utils/app_colors.dart';
import '../services/api_service.dart';

class ManageUsersScreen extends StatefulWidget {
  const ManageUsersScreen({super.key});

  @override
  State<ManageUsersScreen> createState() => _ManageUsersScreenState();
}

class _ManageUsersScreenState extends State<ManageUsersScreen> {
  bool isLoading = true;
  List<dynamic> users = [];
  String searchText = '';

  @override
  void initState() {
    super.initState();
    loadUsers();
  }

  Future<void> loadUsers() async {
    setState(() {
      isLoading = true;
    });

    final result = await ApiService.getUsers();

    if (!mounted) return;

    setState(() {
      users = result['users'] ?? [];
      isLoading = false;
    });
  }

  Future<void> confirmDelete(dynamic user) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Delete User'),
          content: Text(
            'Delete ${user['name']}?\n\nThis will also remove face data from the model.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel'),
            ),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.red,
                foregroundColor: AppColors.white,
              ),
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Delete'),
            ),
          ],
        );
      },
    );

    if (confirm != true) return;

    final result = await ApiService.deleteUser(
      (user['user_id'] ?? '').toString(),
    );

    if (!mounted) return;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(result['message'] ?? 'No response'),
        backgroundColor:
        result['success'] == true ? AppColors.green : AppColors.red,
      ),
    );

    if (result['success'] == true) {
      await loadUsers();
    }
  }

  @override
  Widget build(BuildContext context) {
    final filteredUsers = users.where((user) {
      final name = (user['name'] ?? '').toString().toLowerCase();
      final id = (user['user_id'] ?? '').toString().toLowerCase();
      final group = (user['group'] ?? '').toString().toLowerCase();
      final q = searchText.toLowerCase();

      return name.contains(q) || id.contains(q) || group.contains(q);
    }).toList();

    return RefreshIndicator(
      onRefresh: loadUsers,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(18, 18, 18, 110),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: TextField(
                    onChanged: (value) {
                      setState(() {
                        searchText = value;
                      });
                    },
                    decoration: InputDecoration(
                      hintText: 'Search by name, ID or group',
                      prefixIcon: const Icon(Icons.search_rounded),
                      filled: true,
                      fillColor: AppColors.white,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(20),
                        borderSide: BorderSide.none,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                IconButton(
                  style: IconButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: AppColors.white,
                  ),
                  onPressed: loadUsers,
                  icon: const Icon(Icons.refresh_rounded),
                ),
              ],
            ),

            const SizedBox(height: 20),

            Text(
              'Registered Users (${filteredUsers.length})',
              style: const TextStyle(
                fontSize: 21,
                fontWeight: FontWeight.bold,
                color: AppColors.primaryDark,
              ),
            ),

            const SizedBox(height: 14),

            if (isLoading)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(30),
                  child: CircularProgressIndicator(),
                ),
              )
            else if (filteredUsers.isEmpty)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(30),
                  child: Text(
                    'No users found',
                    style: TextStyle(color: AppColors.grey),
                  ),
                ),
              )
            else
              ListView.builder(
                itemCount: filteredUsers.length,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemBuilder: (context, index) {
                  final user = filteredUsers[index];

                  String name = (user['name'] ?? 'Unknown').toString();
                  String userId = (user['user_id'] ?? '').toString();
                  String group = (user['group'] ?? 'Unknown Group').toString();

                  if (name.trim().isEmpty) name = 'Unknown';
                  if (group.trim().isEmpty) group = 'Unknown Group';

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
                          radius: 30,
                          backgroundColor: AppColors.primary.withOpacity(0.12),
                          child: Text(
                            name[0].toUpperCase(),
                            style: const TextStyle(
                              color: AppColors.primary,
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                            ),
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
                                  fontSize: 17,
                                  fontWeight: FontWeight.bold,
                                  color: AppColors.primaryDark,
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
                        IconButton(
                          icon: const Icon(
                            Icons.delete_rounded,
                            color: AppColors.red,
                          ),
                          onPressed: () {
                            confirmDelete(user);
                          },
                        ),
                      ],
                    ),
                  );
                },
              ),
          ],
        ),
      ),
    );
  }
}