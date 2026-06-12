"""
check_server.py — Check Pi server status and tail logs.
Run: python check_server.py
"""

import time
import paramiko

HOST     = "192.168.1.2"
USER     = "admin"
PASSWORD = "admin"


def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)
    return ssh


def run(ssh, cmd, timeout=10):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err


if __name__ == "__main__":
    print(f"Checking Pi at {HOST}...\n")
    ssh = connect()

    print("=== Setup / Install Log (last 30 lines) ===")
    out, _ = run(ssh, "tail -30 /tmp/setup.log 2>/dev/null || echo '(no log yet)'")
    print(out)

    print("\n=== Server Log (last 20 lines) ===")
    out, _ = run(ssh, "tail -20 /tmp/attendance.log 2>/dev/null || echo '(no server log yet)'")
    print(out)

    print("\n=== Health Check ===")
    out, err = run(ssh, "curl -s --max-time 5 http://localhost:8000/health || echo 'NOT_RESPONDING'")
    if "ok" in out.lower():
        print("✅ Server is ONLINE!")
        print(out)
    else:
        print("⏳ Server not yet responding (may still be installing packages)")
        print(out or err)

    print("\n=== Running processes ===")
    out, _ = run(ssh, "ps aux | grep uvicorn | grep -v grep || echo 'uvicorn not running'")
    print(out)

    ssh.close()
