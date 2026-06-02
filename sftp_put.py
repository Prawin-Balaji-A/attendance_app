import paramiko
import sys

local_path = sys.argv[1]
remote_path = sys.argv[2]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.1.5', username='admin', password='admin')
sftp = ssh.open_sftp()
sftp.put(local_path, remote_path)
sftp.close()

# Restart the FastAPI server to apply changes.
commands = [
    "pkill -f uvicorn",
    "cd /home/admin/Desktop/attendance_app/backend && nohup .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
]

for cmd in commands:
    ssh.exec_command(cmd)

ssh.close()
print(f"Uploaded {local_path} to {remote_path} and restarted service")
