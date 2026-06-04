```bash
v4l2-ctl --list-devices


```
NVIDIA Tegra Video Input Device (platform:tegra-camrtc-ca):
	/dev/media0

Web Camera: Web Camera (usb-3610000.usb-2.1):
	/dev/video2
	/dev/video3
	/dev/media2

Web Camera: Web Camera (usb-3610000.usb-2.3):
	/dev/video0
	/dev/video1
	/dev/media1

sudo touch /etc/udev/rules.d/99-webcam.rules
```
# Front camera (USB port 2.1)
SUBSYSTEM=="video4linux", ENV{ID_PATH}=="platform-3610000.usb-usb-0:2.3:1.0", ENV{ID_V4L_CAPABILITIES}==":capture:", SYMLINK+="videotop", MODE="0666"

# Top camera (USB port 2.4)
SUBSYSTEM=="video4linux", ENV{ID_PATH}=="platform-3610000.usb-usb-0:2.1:1.0", ENV{ID_V4L_CAPABILITIES}==":capture:", SYMLINK+="videofront", MODE="0666"

# Wrist camera
# SUBSYSTEM=="video4linux", ENV{ID_PATH}=="platform-3610000.usb-usb-0:1.3:1.0", ENV{ID_V4L_CAPABILITIES}==":capture:", SYMLINK+="videowrist", MODE="0666"
```

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM_so101follower \
    --robot.id=cse_so101follower \
    --robot.cameras="{top: {type: opencv, index_or_path: '/dev/videotop', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}, front: {type: opencv, index_or_path: '/dev/videofront', width: 640, height: 480, fps: 30, backend: 200, fourcc: MJPG}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM_so101leader \
    --teleop.id=cse_so101_leader \
    --display_data=true \
    --display_compressed_images=true \
    --display_image_interval_s=0.5
```
