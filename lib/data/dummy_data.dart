import '../models/attendance_model.dart';
import '../models/user_model.dart';
import '../models/group_model.dart';

List<GroupModel> dummyGroupList = [
  GroupModel(
    groupName: 'CSE A',
    groupType: 'Class',
    totalMembers: 60,
    presentCount: 52,
    absentCount: 8,
  ),
  GroupModel(
    groupName: 'CSE B',
    groupType: 'Class',
    totalMembers: 58,
    presentCount: 49,
    absentCount: 9,
  ),
  GroupModel(
    groupName: 'IT A',
    groupType: 'Class',
    totalMembers: 55,
    presentCount: 46,
    absentCount: 9,
  ),
  GroupModel(
    groupName: 'Development Team',
    groupType: 'Company Team',
    totalMembers: 32,
    presentCount: 29,
    absentCount: 3,
  ),
  GroupModel(
    groupName: 'HR Team',
    groupType: 'Company Team',
    totalMembers: 12,
    presentCount: 11,
    absentCount: 1,
  ),
  GroupModel(
    groupName: 'Sales Team',
    groupType: 'Company Team',
    totalMembers: 24,
    presentCount: 19,
    absentCount: 5,
  ),
];

List<AttendanceModel> dummyAttendanceList = [
  AttendanceModel(
    name: 'Prawin',
    id: 'CSE001',
    department: 'CSE A',
    time: '09:10 AM',
    status: 'Present',
  ),
  AttendanceModel(
    name: 'Rafeeq',
    id: 'CSE002',
    department: 'CSE A',
    time: '09:15 AM',
    status: 'Present',
  ),
  AttendanceModel(
    name: 'Prem Vignesh',
    id: 'CSE003',
    department: 'CSE A',
    time: '--',
    status: 'Absent',
  ),
  AttendanceModel(
    name: 'Murugan',
    id: 'CSE004',
    department: 'CSE B',
    time: '09:25 AM',
    status: 'Present',
  ),
  AttendanceModel(
    name: 'Daniel',
    id: 'CSE005',
    department: 'CSE B',
    time: '--',
    status: 'Absent',
  ),
  AttendanceModel(
    name: 'Arun',
    id: 'DEV001',
    department: 'Development Team',
    time: '09:05 AM',
    status: 'Present',
  ),
  AttendanceModel(
    name: 'Meena',
    id: 'HR001',
    department: 'HR Team',
    time: '09:20 AM',
    status: 'Present',
  ),
];

List<UserModel> dummyUserList = [
  UserModel(
    name: 'Prawin',
    id: 'CSE001',
    department: 'CSE A',
    registeredDate: '25/05/2026',
  ),
  UserModel(
    name: 'Rafeeq',
    id: 'CSE002',
    department: 'CSE A',
    registeredDate: '25/05/2026',
  ),
  UserModel(
    name: 'Prem Vignesh',
    id: 'CSE003',
    department: 'CSE A',
    registeredDate: '25/05/2026',
  ),
  UserModel(
    name: 'Murugan',
    id: 'CSE004',
    department: 'CSE B',
    registeredDate: '25/05/2026',
  ),
  UserModel(
    name: 'Daniel',
    id: 'CSE005',
    department: 'CSE B',
    registeredDate: '25/05/2026',
  ),
  UserModel(
    name: 'Arun',
    id: 'DEV001',
    department: 'Development Team',
    registeredDate: '25/05/2026',
  ),
  UserModel(
    name: 'Meena',
    id: 'HR001',
    department: 'HR Team',
    registeredDate: '25/05/2026',
  ),
];