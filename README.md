# Awesome Flipping Robot 🥞🤖

An interactive pancake-flipping robot project combining **computer vision, robotics simulation, and real-world camera input**.

The project uses a **Luxonis OAK-1** camera to detect pancakes from a top-down view and **MuJoCo** to simulate the robot's flipping environment.

## ✨ Features

* 🥞 Real-time pancake detection using OpenCV
* 📷 Support for:

  * Built-in laptop webcams
  * External USB webcams
  * **Luxonis OAK-1**
* 🎨 Interactive HSV colour calibration
* 🔎 Pancake detection using:

  * HSV colour segmentation
  * Contour analysis
  * Circularity
  * Solidity
  * Aspect ratio
  * Shape scoring
* 🧹 Morphological image processing to clean the detection mask
* 📊 Real-time detection information and FPS
* ⏯️ Pause and resume detection
* 💾 Screenshot capture
* 🤖 MuJoCo robot simulation
* 🔌 Separate vision and simulation processes

---

## 🏗️ Project Structure

```text
Awesome-Flipping-Robot/
│
├── demo.py
├── oak_test.py
├── requirements.txt
├── README.md
│
├── vision/
│   └── pancake_detection.py
│
└── sim/
    └── view_sim.py
```

### Main Files

| File                          | Purpose                                          |
| ----------------------------- | ------------------------------------------------ |
| `demo.py`                     | Launches the vision system and MuJoCo simulation |
| `oak_test.py`                 | Standalone OAK-1 camera test                     |
| `vision/pancake_detection.py` | Pancake detection and computer vision pipeline   |
| `sim/view_sim.py`             | MuJoCo simulation                                |
| `requirements.txt`            | Python dependencies                              |

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Awesome-Flipping-Robot
```

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses:

* Python
* NumPy
* OpenCV
* DepthAI
* MuJoCo

The `depthai` package is required when using a **Luxonis OAK-1**.

---

# 📷 Camera Setup

The pancake detector supports both standard OpenCV cameras and the OAK-1.

## Standard Webcam

A built-in laptop camera or normal USB webcam can be accessed through an OpenCV camera index.

For example:

```bash
python3 vision/pancake_detection.py --source 0
```

If you have multiple cameras, try:

```bash
python3 vision/pancake_detection.py --source 1
```

You can also let the program automatically find the first available camera:

```bash
python3 vision/pancake_detection.py --source auto
```

### List Available OpenCV Cameras

```bash
python3 vision/pancake_detection.py --list-cameras
```

---

# 🟢 Luxonis OAK-1

The OAK-1 uses **DepthAI** rather than OpenCV's normal camera-index system.

Therefore, you should **not** expect the OAK-1 to appear as camera `1` or `2`.

Connect the OAK-1 through USB and run:

```bash
python3 vision/pancake_detection.py --source oak
```

The program should display:

```text
Using OAK-1 via DepthAI
```

The camera provides a BGR image to the existing OpenCV pancake detection pipeline.

## Test the OAK-1 Independently

Before troubleshooting the pancake detector, you can test the camera itself:

```bash
python3 oak_test.py
```

If the OAK-1 feed appears, the camera and DepthAI installation are working.

---

# 🥞 Pancake Detection

The main computer vision pipeline is located at:

```text
vision/pancake_detection.py
```

The detector processes each frame through several stages:

```text
Camera
   ↓
BGR Frame
   ↓
Resize
   ↓
HSV Conversion
   ↓
Colour Thresholding
   ↓
Morphological Cleanup
   ↓
Contour Detection
   ↓
Shape Analysis
   ↓
Pancake Candidate Ranking
   ↓
Pancake Detection
```

The detector considers characteristics such as:

* Area
* Circularity
* Solidity
* Aspect ratio
* Border proximity
* Shape consistency
* Detection continuity across frames

---

# 🎨 HSV Calibration

When the detector starts, you can calibrate the pancake colour by clicking on the middle of the pancake.

The HSV controls can also be adjusted manually.

The interface includes controls for:

* Hue
* Saturation
* Value
* Minimum contour area
* Morphological cleanup
* Minimum shape percentage

This allows the detector to be adapted to different pancakes, lighting conditions, and cooking surfaces.

---

# ⌨️ Controls

| Key     | Action                              |
| ------- | ----------------------------------- |
| `Q`     | Quit                                |
| `ESC`   | Quit                                |
| `Space` | Pause / resume                      |
| `M`     | Show / hide detection mask          |
| `C`     | Show / hide candidates              |
| `S`     | Save screenshot                     |
| `P`     | Print current detection information |

### Mouse

Click on the **middle of the pancake** to automatically calibrate the HSV colour range.

---

# 🤖 Running the Full Demo

The main entry point is:

```bash
python3 demo.py
```

By default, the demo launches the available camera and the MuJoCo simulation.

### Use a Specific Webcam

```bash
python3 demo.py --source 0
```

### Use the OAK-1

```bash
python3 demo.py --source oak
```

### Run Only the Vision System

```bash
python3 demo.py --no-sim
```

### Run Only the Simulation

```bash
python3 demo.py --no-vision
```

---

# 🧩 System Architecture

The project separates the computer vision system from the robot simulation.

```text
                    ┌─────────────────────┐
                    │       Camera        │
                    │                     │
                    │  Webcam / OAK-1     │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │  Pancake Detection  │
                    │                     │
                    │  OpenCV + DepthAI   │
                    └──────────┬──────────┘
                               │
                               │ Detection
                               ↓
                    ┌─────────────────────┐
                    │     Robot Logic     │
                    │      / Future       │
                    │     Integration     │
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌─────────────────────┐
                    │       MuJoCo        │
                    │     Simulation      │
                    └─────────────────────┘
```

The vision system and simulation are launched as separate processes so that the camera pipeline and simulation can operate independently.

---

# 🛠️ Troubleshooting

## `ModuleNotFoundError: No module named 'depthai'`

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Or install DepthAI directly:

```bash
pip install depthai==3.10.0
```

---

## `Video file not found: oak`

This means the detector is treating `oak` as a video filename.

Make sure you are using the updated version of `pancake_detection.py` with OAK-1 / DepthAI support.

Run:

```bash
python3 vision/pancake_detection.py --source oak
```

---

## OAK-1 Is Not Detected

Check:

1. The OAK-1 is connected through USB.
2. Your USB cable supports data transfer.
3. The virtual environment is activated.
4. DepthAI is installed.
5. The standalone OAK-1 test works.

You can check whether DepthAI detects the device with:

```bash
python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

Then test the camera:

```bash
python3 oak_test.py
```

---

## OpenCV Cannot Find My Webcam

List available OpenCV cameras:

```bash
python3 vision/pancake_detection.py --list-cameras
```

You can then try a specific camera:

```bash
python3 vision/pancake_detection.py --source 0
```

or:

```bash
python3 vision/pancake_detection.py --source 1
```

---

# 📦 Dependencies

The main Python dependencies are listed in:

```text
requirements.txt
```

Current dependencies:

```text
numpy
opencv-python
depthai==3.10.0
mujoco
```

---

# 💻 Platform Notes

The project is currently developed and tested on **macOS with Apple Silicon**.

The computer vision pipeline uses OpenCV, while the OAK-1 camera is accessed through DepthAI.

MuJoCo may require platform-specific setup depending on your operating system and Python environment.

---

# 🔮 Future Development

* [ ] Connect pancake detection to robot control
* [ ] Estimate pancake position relative to the robot
* [ ] Detect pancake flipping events
* [ ] Control the simulated robot using vision data
* [ ] Transfer the control system from simulation to hardware
* [ ] Improve detection under different lighting conditions
* [ ] Add real-time robot feedback
* [ ] Support additional OAK camera capabilities
* [ ] Integrate the complete flipping pipeline

---

# 📄 License

Add your project's license here.

---

# 🙌 Acknowledgements

Built using:

* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)
* [DepthAI](https://github.com/luxonis/depthai-python)
* [Luxonis OAK-1](https://www.luxonis.com/)
* [MuJoCo](https://mujoco.org/)
