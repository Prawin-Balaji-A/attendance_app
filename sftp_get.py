import paramiko
import sys

remote_path = sys.argv[1]
local_path = sys.argv[2]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.1.5', username='admin', password='admin')
sftp = ssh.open_sftp()
sftp.get(remote_path, local_path)
sftp.close()
ssh.close()
print(f"Downloaded {remote_path} to {local_path}")
