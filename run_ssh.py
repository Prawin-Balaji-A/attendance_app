import paramiko
import sys

cmd = sys.argv[1]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.1.5', username='admin', password='admin')
stdin, stdout, stderr = ssh.exec_command(cmd)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    print("ERROR:", err, file=sys.stderr)
ssh.close()
