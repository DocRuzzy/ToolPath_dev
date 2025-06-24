# 3D Printing Toolpath (G-code) Editor

This project is a Python-based GUI tool for editing 3D printer G-code files. It uses PyQt5 for the interface.

## Purpose

This tool is designed to address a common issue in multi-material 3D printing where distinct material channels are created. Specifically, it aims to fix and optimize toolpaths in the layers immediately above and below a designated "channel" (printed with a specific extruder). The primary goal is to ensure that the filament extruded in these adjacent layers forms a continuous path, especially extending *past* the start and end points of the channel segments. This prevents artifacts like gaps or material "leaking" into the channel, which can occur if the nozzle lifts or changes direction precisely at the channel's edge.

This is particularly problematic at 3-way and 4-way junctions of channels. Such junctions often necessitate stops and starts in filament deposition in the layers forming the "floor" and "ceiling" of the channel. Without careful toolpath control, these areas can suffer from holes, gaps, or excessive filament that squeezes into and blocks the channel.

## Features

- Open a GUI to select a G-code file for editing.
- Quality checks to ensure a file is selected before editing or saving.
- Option to save the edited G-code file.
- Placeholder for a future 3D G-code viewer.
- **Automatic Channel-Aware Toolpath Optimization (Planned):**
    - User specifies the extruder used for printing "channels" (e.g., T2).
    - The program will automatically identify all layers utilizing this channel extruder.
    - It will then trace the paths of these channels within their respective layers.
    - Based on the channel geometry, the tool will automatically edit the toolpaths in the layers immediately above and below the channel.
    - These edits will extend the extrusion moves in the adjacent layers slightly beyond the channel's start and end points, promoting better sealing and structural integrity, especially at junctions.

## Getting Started

### 1. Install dependencies
```
pip install pyqt5
```

### 2. Run the application
```
python main.py
```

---

## Roadmap
- [ ] Add 3D G-code viewer
- [ ] Add advanced editing features