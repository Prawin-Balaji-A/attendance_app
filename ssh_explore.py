import paramiko
import json

def explore_pi():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect('192.168.1.2', username='admin', password='admin', timeout=5)
        stdin, stdout, stderr = client.exec_command('ls -la /home/admin/Desktop/attendance_backend')
        print(stdout.read().decode())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == '__main__':
    explore_pi()
