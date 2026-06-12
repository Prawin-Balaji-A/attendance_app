import paramiko
HOST='192.168.1.2'; USER='admin'; PASSWORD='admin'
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASSWORD, timeout=15)
def run(cmd, timeout=10):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(errors='replace').strip()

# Test color capture directly - save a test frame and check channel values
out = run("""python3 << 'PYEOF'
from picamera2 import Picamera2
import numpy as np
cam = Picamera2()

# Test RGB888
config = cam.create_preview_configuration(main={"size":(640,480),"format":"RGB888"})
cam.configure(config)
cam.start()
import time; time.sleep(1)
arr = cam.capture_array()
print("RGB888 shape:", arr.shape, "dtype:", arr.dtype)
print("RGB888 first pixel:", arr[240,320])
cam.stop()
cam.close()

# Test XRGB8888
cam2 = Picamera2()
config2 = cam2.create_preview_configuration(main={"size":(640,480),"format":"XRGB8888"})
cam2.configure(config2)
cam2.start()
time.sleep(1)
arr2 = cam2.capture_array()
print("XRGB8888 shape:", arr2.shape, "dtype:", arr2.dtype)
cam2.stop()
cam2.close()
PYEOF""", timeout=15)
print(out)
ssh.close()
