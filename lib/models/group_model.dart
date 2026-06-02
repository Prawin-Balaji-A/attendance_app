class GroupModel {
  final String groupName;
  final String groupType;
  final int totalMembers;
  final int presentCount;
  final int absentCount;

  GroupModel({
    required this.groupName,
    required this.groupType,
    required this.totalMembers,
    required this.presentCount,
    required this.absentCount,
  });

  int get attendancePercentage {
    if (totalMembers == 0) return 0;
    return ((presentCount / totalMembers) * 100).round();
  }
}