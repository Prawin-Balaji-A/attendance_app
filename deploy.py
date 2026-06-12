"""
deploy.py — Deploy backend files to Raspberry Pi via SFTP/SSH.
Run: python deploy.py
"""

import os
import time
import paramiko

HOST     = "192.168.1.2"
USER     = "admin"
PASSWORD = "admin"
REMOTE   = "/home/admin/Desktop/attendance_backend"
LOCAL    = os.path.join(os.path.dirname(__file__), "backend")


def ensure_remote_dir(sftp, path):
    parts = path.replace("\\", "/").split("/")
    current = ""
    for part in parts:
        if not part:
            continue
        current += "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def upload_dir(sftp, local_dir, remote_dir):
    ensure_remote_dir(sftp, remote_dir)
    for item in sorted(os.listdir(local_dir)):
        local_path  = os.path.join(local_dir, item)
        remote_path = remote_dir + "/" + item
        if os.path.isdir(local_path):
            upload_dir(sftp, local_path, remote_path)
        else:
            print(f"  Upload: {item}")
            sftp.put(local_path, remote_path)


def run_ssh(ssh, cmd, timeout=30):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    code = stdout.channel.recv_exit_status()
    return out, err, code


def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    return ssh


if __name__ == "__main__":
    print(f"[1/5] Connecting to {HOST}...")
    ssh = connect()

    # ── Step 1: Clear old Python/sh files, keep venv + data ───────────────
    print("[2/5] Clearing old files (preserving venv/ and data/)...")
    run_ssh(ssh, f"""
        find {REMOTE} -maxdepth 1 -name '*.py' -delete 2>/dev/null
        find {REMOTE} -maxdepth 1 -name '*.txt' -delete 2>/dev/null
        find {REMOTE} -maxdepth 1 -name '*.sh'  -delete 2>/dev/null
        rm -rf {REMOTE}/pipeline 2>/dev/null
        mkdir -p {REMOTE}/data {REMOTE}/models
    """, timeout=15)

    # ── Step 2: Upload all backend files ──────────────────────────────────
    print("[3/5] Uploading backend files...")
    sftp = ssh.open_sftp()
    upload_dir(sftp, LOCAL, REMOTE)
    sftp.close()

    # Make start.sh executable
    run_ssh(ssh, f"chmod +x {REMOTE}/start.sh", timeout=5)
    print("  All files uploaded.")

    # ── Step 3: Write a remote install+start script ───────────────────────
    print("[4/5] Writing remote setup script...")
    setup_script = f"""#!/bin/bash
cd {REMOTE}
echo "[setup] Creating virtualenv if needed..."
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo "[setup] Upgrading pip..."
pip install --upgrade pip -q
echo "[setup] Installing requirements..."
pip install -r requirements.txt
echo "[setup] DONE installing."

# Kill any existing server
pkill -f 'uvicorn main:app' 2>/dev/null || true
sleep 1

echo "[setup] Starting server..."
nohup uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 \
    > /tmp/attendance.log 2>&1 &
echo "Server started with PID $!"
"""
    sftp = ssh.open_sftp()
    with sftp.open(f"{REMOTE}/setup_and_start.sh", "w") as f:
        f.write(setup_script)
    sftp.close()
    run_ssh(ssh, f"chmod +x {REMOTE}/setup_and_start.sh", timeout=5)

    # ── Step 4: Run setup in background (nohup so SSH disconnect is safe) ─
    print("[5/5] Running pip install + server start in background on Pi...")
    print("      (This takes 3-10 minutes on Pi - packages are large)")
    run_ssh(ssh,
        f"nohup bash {REMOTE}/setup_and_start.sh > /tmp/setup.log 2>&1 &",
        timeout=10,
    )

    ssh.close()

    print()
    print("=" * 55)
    print("Files deployed. Pi is installing packages in background.")
    print()
    print("Monitor progress:")
    print("  python check_server.py")
    print("=" * 55)
