
# 🦾 XLeRobot

**XLeRobot** is a customized version of [🤗 LeRobot](https://github.com/huggingface/lerobot) tailored for easier setup and use on both Mac and Linux systems.
It provides additional setup guidance, calibration steps, and  improvements for getting your robot up and running quickly.

<br>

---

## 🚀 Installation & Setup

If you prefer to follow the original instructions, see:
🔗 [LeRobot Installation Docs](https://huggingface.co/docs/lerobot/installation)
🔗 [XLeRobot Installation Docs](https://xlerobot.readthedocs.io/en/latest/software/getting_started/install.html)

<br>

---

## 1. Install Conda

**Mac:**
```bash
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

**Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh
```

Then **open a new terminal**.

<br>

---

## 2. Create the Environment

Clone this repository:
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
Create a virtual environment with Python 3.10, using conda:

```bash
conda create -y -n lerobot python=3.10
```

Then activate your conda environment, you have to do this each time you open a shell to use lerobot:

```bash
conda activate lerobot
```

When using conda, install `ffmpeg` in your environment:

```bash
conda install ffmpeg -c conda-forge
```

⚠️ **IMPORTANT:** Run `conda activate lerobot` every time you open a new terminal.

<br>

---

## 3. Install the XLe Robot

```bash
git clone https://github.com/Minko82/xle-robot.git
cd xle-robot
pip install -e .
pip install 'lerobot[all]'
pip install -e ".[feetech]"
```

<br>

### Troubleshooting: 

If you encounter build errors, you may need to install additional dependencies. 

**Mac:**

```bash
brew install cmake pkg-config ffmpeg python 
```

**Linux:**

```bash
sudo apt-get install cmake build-essential python-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config
```

<br>

---
## 4. Arm Setup
We’ll first identify each arm and make sure their ports are stable for calibration.

Steps differ slightly on macOS and Linux. For Windows, you will have to install WSL and run Ubuntu on the top of your Windows system. Please see the below Windows instructions. Then you should follow the Linux instructions and setup the arms.

<details>
<summary><b> Windows Instructions</b></summary>

This guide gives a simple instruction for how to install **WSL**, run **Ubuntu**, and map **USB/Serial devices** into WSL using **usbipd-win**. 

System requirements:
**Windows 10 (21H2+)** or **Windows 11**

## 1. Install WSL

Open **PowerShell as Administrator**:

Right click Start → **Windows PowerShell (Admin)**

Run:

```powershell
wsl --install
```

To install a specific version (example: Ubuntu 22.04)（For ROS1, use Ubuntu20.04 or older version）

```powershell
wsl --install -d Ubuntu-22.04
```

List installed distros:

```powershell
wsl -l -v
```

## 2. First launch Ubuntu
Open **Ubuntu** from Start Menu and set username & password.

Update the system:

```bash
sudo apt update && sudo apt upgrade -y
```

## 3. USB Serial
WSL blocks the serial reading, we need to use USB/Serial mapping tool like usbipd-win
PowerShell (Admin):

```powershell
winget install --interactive --exact dorssel.usbipd-win
```
After that, restart the powershell for a new interface, then check USB devices:

```powershell
usbipd wsl list
```

## 4. Enable USB/IP support inside WSL
In Ubuntu:

```bash
sudo apt install linux-tools-virtual hwdata linux-cloud-tools-virtual -y
```

Verify:

```bash
usbip list -l
```

## 5. Attach USB/Serial devices to WSL

### 5.1 List devices

PowerShell (Admin):

```powershell
usbipd wsl list
```

Example output:

```
BUSID  DEVICE
1-3    USB Serial CH340 (COM3)
1-5    Intel RealSense Camera
```

### 5.2 Attach device to WSL

```powershell
usbipd wsl attach --busid 1-3
```

If you have multiple distros, specify:

```powershell
usbipd wsl attach --busid 1-3 --distribution Ubuntu
```
##  6. Access device inside WSL

In Ubuntu:

```bash
ls /dev/tty*
```

Common device names:

| Device Type | Linux Device |
|-------------|--------------|
| CH340, CP210x, FTDI | `/dev/ttyUSB0` |
| Arduino, Pico | `/dev/ttyACM0` |
| Built-in serial | `/dev/ttyS0` |

Fix permissions:

```bash
sudo usermod -aG dialout $USER
newgrp dialout
```
##  Done!

You can now head to the 🐧 Linux Instructions below and setup the arms.

</details>
   
<details>
<summary><b>🍎 macOS Instructions</b></summary>
   
## 1. Get the Right Arm Serial Number

**Connect the following:**

-   Right bus servo adapter → Dock → Computer

Then run:

``` bash
ioreg -p IOUSB -l | grep -iE "tty|serial"
```

**Example (partial) output:**

``` bash
|   +-o USB Single Serial@01140000  <class IOUSBHostDevice, ... >
|         "iSerialNumber" = 3
|         "USB Product Name" = "USB Single Serial"
|         "kUSBSerialNumberString" = "5A68013518"
|         "USBPortType" = 0
|         "kUSBProductString" = "USB Single Serial"
|         "USB Serial Number" = "5A68013518"
```

**Record the USB Serial Number**, for example:

-   **Right arm USB Serial Number:** `5A68013518`

<br>

## 2. Get the Left Arm Serial Number

1.  Unplug the right arm adapter.
2.  Plug in the left arm adapter.
3.  Run:

``` bash
ioreg -p IOUSB -l | grep -iE "tty|serial"
```

Then record the left arm's USB Serial Number, for example:

-   **Left arm USB Serial Number:** `5A68012794`

You will later see corresponding `/dev/tty.usbmodem*` device paths when
running `lerobot-find-port`.

<br>

## 3. Calibrate the Robot
*Run once per new computer used*

### 1. Connect the Hardware


Both arms must be calibrated, individually.

Connect the dock and the correct bus-servo adapter to your computer:

-   If calibrating the **left arm**, plug in the left adapter.
-   If calibrating the **right arm**, plug in the right adapter.

<br>

### 2. Run the Calibration Command

Use the follower-arm calibration command for each arm, but replace
the path with the actual USB serial device you previously recorded.

**Using your real device path:**

``` bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5A680135181
```

<br>


**Example output:**

During calibration, you should see live joint position values updating
as you move the arm.

    -------------------------------------------
    NAME            |    MIN |    POS |    MAX
    shoulder_pan    |   2047 |   2047 |   2047
    shoulder_lift   |   2047 |   2047 |   2047
    elbow_flex      |   2046 |   2046 |   2046
    wrist_flex      |   2047 |   2047 |   2047
    wrist_roll      |   2047 |   2047 |   2047
    gripper         |   2047 |   2047 |   2047

**Note:**
These numbers should change **as you physically move the arm** during
calibration.
<br>

</details>

<details>
<summary><b>🐧 Linux Instructions</b></summary>
   
We’ll assign fixed USB names to each arm so they remain consistent (`/dev/xle_right` and `/dev/xle_left`).

## 1. Get the Right Arm Serial Number

**Connect the following:**

-   Right bus servo adapter → Dock → Computer

Then, run:

``` bash
udevadm info -a -n /dev/ttyACM0 | grep 'ATTRS{serial}'
```

**Example output:**

``` bash
ATTRS{serial}=="A50285B1"
```

**Record the USB Serial Number**.


<br>

## 2. Get the Left Arm Serial Number

1.  Unplug the right arm adapter.
2.  Plug in the left arm adapter.
3.  Run:

``` bash
udevadm info -a -n /dev/ttyACM0 | grep 'ATTRS{serial}'
```

**Example output:**

``` bash
ATTRS{serial}=="A50285B1"
```

**Record the USB Serial Number**.

## 3. Create a New udev Rules File
   ```bash
   sudo nano /etc/udev/rules.d/99-so100-robot.rules
   ```

   Paste this (replace serials with yours):
   ```
   # Right Arm
   SUBSYSTEM=="tty", ATTRS{serial}=="YOUR_SERIAL_FOR_ARM_1", SYMLINK+="xle_right"

   # Left Arm
   SUBSYSTEM=="tty", ATTRS{serial}=="YOUR_SERIAL_FOR_ARM_2", SYMLINK+="xle_left"
   ```

 ## 4. Apply the Rules

   ``` bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

   After this, your arms will appear as:
   
   -   `/dev/xle_right`
   -   `/dev/xle_left`

   Grant permission to the ports:
   ``` bash
   sudo chmod 666 /dev/xle_left /dev/xle_right
   ```

 ## 5. **Make and copy calibration files:**
 We have calibrated the arms for you (Linux only). Please find the calibration files (left_arm.json and right_arm.json) and move them to your cache directory.
   ```bash
   mkdir -p ~/.cache/huggingface/lerobot/calibration/robots/so100_follower
   cp calibration/left_arm.json calibration/right_arm.json ~/.cache/huggingface/lerobot/calibration/robots/so100_follower
   ```

<br>
 </details>  

<br>
 
---

## 5. Optional Components Setup
These steps are optional and should be followed only if your project requires these tools.

<details>
<summary><b>🦾 Wrist Cameras Setup</b></summary>

<br>

### Setup Overview

1. **Find Camera Indices**
   ```bash
   v4l2-ctl --list-devices
   ```
   Example output:
   ```
   USB Camera (usb-0000:00:1a.0-1.2):
       /dev/video0
       /dev/video1
   ```
   The numbers (0, 1, etc.) are your camera indices.

2. **Install OpenCV**
   ```bash
   pip install opencv-python
   ```

3. **Update Example Script**
   Edit `examples/9_dual_wrist_camera.py` and set:
   ```python
   CAMERA_INDEX_1 = 0
   CAMERA_INDEX_2 = 1
   ```
   to match your detected camera indices.

</details>

<details>
<summary><b> 📷 RealSense Camera Setup</b></summary>

<br>

### 1. Install the SDK
```bash
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev
```

### 2. Install the Python Package
```bash
pip install pyrealsense2
```

### 3. Verify the Camera
```bash
realsense-viewer
```
✅ You should see both **color** and **depth** video streams.  
   -  If not, check your USB connection or reinstall the SDK.

</details>

<br>

---

### 6. Connect and Power Everything

1. Plug the **USB hub** into your laptop.  
2. Connect all four devices:
   - Two **wrist cameras**
   - Two **arm control boards**
3. _Optional:_ Plug the **RealSense camera** directly into your laptop.  
4. Turn on the power and verify that all indicator lights are active.

**⚠️ IMPORTANT:** If motors become unresponsive after a failure, unplug and reconnect their **motor power cables** to reset them.

<br>

---

### 7. Check Connected Devices

Run:
```bash
lerobot-find-port
```

<br>

### Example output:

**Mac:**
```bash
'/dev/tty.usbmodem5A680135181'
'/dev/tty.usbmodem5A680127941'
```

**Linux:**
```
right /dev/xle_right
left  /dev/xle_left
```


These are the right and left arm serial numbers. They should correspond with earlier calibration steps.

<br>

---

### 8. Run sample code

Navigate to the example folder and run a script:
```bash
cd examples
python3 0_so100_keyboard_joint_control.py
```

Provide the correct device path name as prompted, from earlier.


Additional compatible example scripts are available in the `examples` folder. More scripts can be found in `examples/provided_examples`, but these have not yet been tested for full compatibility with XLeRobot.

<br>

---

## 💡 Credits

This project builds on top of [🤗 LeRobot](https://github.com/huggingface/lerobot) by Hugging Face Robotics and [XLeRobot](https://github.com/Vector-Wangel/XLeRobot).

Additional development, hardware integration, and testing have been contributed by [@Minko82](https://github.com/Minko82), [@nanasci](https://github.com/nanasci), [@AlinaSkowronek](https://github.com/Alinaskowronek), [@JustinCosta10](https://github.com/justincosta10), and [@Hhy903](https://github.com/Hhy903). 
<br>

---

## 📜 License

Apache License 2.0 — see the [LICENSE](./LICENSE) file for details.
