import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox, QStatusBar, QLabel,
    QListWidget, QAbstractItemView, QHBoxLayout, QDialog, QPushButton, QSlider, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import os
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np

viewer_open_count = 0

class GCodeEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Printing Toolpath (G-code) Editor")
        self.setGeometry(100, 100, 650, 350)
        self.gcode_file_path = None
        self.cleaned_gcode = None # Will be deprecated, use cleaned_gcode_with_extruders
        self.cleaned_gcode_with_extruders = [] # Stores tuples of (line, extruder_command_str)
        self.layer_indices = []
        self.layer_labels = []
        self.selected_layers = set()
        self.viewer_dialog = None  # Persistent 3D viewer dialog
        self.layer_edits = {}  # Store edits per layer index
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setAlignment(Qt.AlignCenter)

        self.info_label = QLabel("No G-code file selected.")
        self.info_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        self.info_label.setFont(font)
        main_layout.addWidget(self.info_label)

        btn_layout = QHBoxLayout()
        self.open_button = QPushButton("Select G-code File")
        self.open_button.setMinimumHeight(32)
        self.open_button.setFont(QFont('Arial', 11))
        self.open_button.clicked.connect(self.open_gcode_file)
        btn_layout.addWidget(self.open_button)

        self.save_button = QPushButton("Save As")
        self.save_button.setMinimumHeight(32)
        self.save_button.setFont(QFont('Arial', 11))
        self.save_button.clicked.connect(self.save_gcode_file)
        self.save_button.setEnabled(False)
        btn_layout.addWidget(self.save_button)

        self.layer_button = QPushButton("Select Layers")
        self.layer_button.setMinimumHeight(32)
        self.layer_button.setFont(QFont('Arial', 11))
        self.layer_button.clicked.connect(self.show_layer_selector)
        self.layer_button.setEnabled(False)
        btn_layout.addWidget(self.layer_button)

        self.view_layer_button = QPushButton("View Layer 3D")
        self.view_layer_button.setMinimumHeight(32)
        self.view_layer_button.setFont(QFont('Arial', 11))
        self.view_layer_button.clicked.connect(self.view_selected_layer)
        self.view_layer_button.setEnabled(False)
        btn_layout.addWidget(self.view_layer_button)

        main_layout.addLayout(btn_layout)

        # Placeholder for 3D G-code viewer
        # TODO: Add 3D viewer widget here in the future

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # --- UI Elements for Automatic Channel Processing ---
        auto_process_layout = QHBoxLayout()
        self.channel_extruder_label = QLabel("Channel Extruder (e.g., T1):")
        auto_process_layout.addWidget(self.channel_extruder_label)

        self.channel_extruder_input = QLineEdit()
        self.channel_extruder_input.setPlaceholderText("T1")
        self.channel_extruder_input.setFixedWidth(50)
        auto_process_layout.addWidget(self.channel_extruder_input)

        self.process_channel_button = QPushButton("Auto-Process Channels")
        self.process_channel_button.setFont(QFont('Arial', 10))
        self.process_channel_button.clicked.connect(self.trigger_auto_process_channels)
        self.process_channel_button.setEnabled(False) # Enable when G-code is loaded
        auto_process_layout.addWidget(self.process_channel_button)

        auto_process_layout.addStretch() # Push elements to the left

        main_layout.addLayout(auto_process_layout)
        # --- End UI Elements for Automatic Channel Processing ---


    def assign_extruders_to_lines(self, lines):
        processed_lines = []
        current_extruder_command = None # Stores the string like "T0", "T1"
        for line in lines:
            stripped_line = line.strip()
            # Check for tool change command (e.g., T0, T1).
            # It should not be part of a comment or other commands like G10.
            # A simple check: starts with T, followed by digits, and is the only command on the line (before comments).
            command_part = stripped_line.split(';')[0].strip() # Get command before any comment
            words = command_part.split()
            if len(words) > 0 and words[0].startswith('T'):
                tool_cmd = words[0]
                if len(tool_cmd) > 1 and tool_cmd[1:].isdigit():
                    # Check if it's a standalone tool command, not something like "G1 X10 T1" (which is not standard)
                    # Standard tool changes are usually on their own line: "T0"
                    # Or sometimes with M6: "M6 T0" - but slicers often just use "T0"
                    if len(words) == 1: # e.g. "T0"
                        current_extruder_command = tool_cmd
                    # Add more sophisticated parsing if M6 T<n> or other forms are common and need support
            processed_lines.append((line, current_extruder_command))
        return processed_lines

    def open_gcode_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select G-code File", "", "G-code Files (*.gcode *.nc *.txt);;All Files (*)")
        if file_path:
            self.gcode_file_path = file_path
            filename = os.path.basename(file_path)
            self.info_label.setText(f"Selected: {filename}")
            self.save_button.setEnabled(True)
            self.status_bar.showMessage(f"Selected: {filename}")
            try:
                with open(file_path, 'r') as file:
                    raw_lines = file.readlines()
                temp_cleaned_lines = self.remove_all_thumbnails(raw_lines)
                self.cleaned_gcode_with_extruders = self.assign_extruders_to_lines(temp_cleaned_lines)
                # For compatibility, self.cleaned_gcode can store just the lines
                self.cleaned_gcode = [item[0] for item in self.cleaned_gcode_with_extruders]
                self.parse_layers(self.cleaned_gcode) # parse_layers uses self.cleaned_gcode for now
                self.layer_button.setEnabled(True if self.layer_indices else False)
                self.process_channel_button.setEnabled(True if self.cleaned_gcode_with_extruders else False) # Enable auto-process button
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read file: {e}")
                self.cleaned_gcode = None
                self.cleaned_gcode_with_extruders = []
                self.layer_indices = []
                self.layer_labels = []
                self.layer_button.setEnabled(False)
                self.process_channel_button.setEnabled(False) # Disable on error
        else:
            self.status_bar.showMessage("No file selected.")

    def parse_layers(self, lines):
        self.layer_indices = []
        self.layer_labels = []
        for i, line in enumerate(lines):
            if line.strip() == ';LAYER_CHANGE':
                label = f"Layer {len(self.layer_indices)}"
                self.layer_indices.append(i)
                self.layer_labels.append(label)

    def show_layer_selector(self):
        if not self.layer_labels:
            QMessageBox.warning(self, "Warning", "No layers found in the G-code file.")
            return
        dlg = LayerSelectorDialog(self.layer_labels, self.selected_layers, self)
        if dlg.exec_():
            self.selected_layers = dlg.get_selected_layers()
            if not self.selected_layers:
                QMessageBox.warning(self, "Warning", "No layers selected.")
                self.view_layer_button.setEnabled(False)
            elif len(self.selected_layers) == 1:
                self.view_layer_button.setEnabled(True)
                self.status_bar.showMessage(f"Selected layer: {list(self.selected_layers)[0]}")
            else:
                self.view_layer_button.setEnabled(False)
                self.status_bar.showMessage(f"Selected layers: {sorted(self.selected_layers)}")

    def remove_all_thumbnails(self, lines):
        cleaned = []
        skip = False
        for line in lines:
            if 'thumbnail_QOI begin' in line or 'thumbnail begin' in line:
                skip = True
                continue
            if 'thumbnail_QOI end' in line or 'thumbnail end' in line:
                skip = False
                continue
            if not skip:
                cleaned.append(line)
        return cleaned

    def moves_to_gcode(self, moves):
        # Convert moves (list of dicts) back to G-code lines
        gcode_lines = []
        last_e_val_in_gcode = None # Tracks the E value actually written in the last G-code line that had an E.

        for m in moves:
            x_str = f"X{m['x']:.3f}" if 'x' in m and m['x'] is not None else ''
            y_str = f"Y{m['y']:.3f}" if 'y' in m and m['y'] is not None else ''
            z_str = f"Z{m['z']:.3f}" if 'z' in m and m['z'] is not None else ''
            e_str = '' # Default to no E parameter

            if m.get('type') == 'travel':
                # For travel moves, no E parameter is written.
                # The extruder holds its position from the last extrusion.
                # The m['e'] in the move dictionary should still reflect this logical extruder position.
                pass # e_str remains empty, so no E parameter will be added to the G-code line.
            else: # Non-travel (presumably extrusion)
                if 'e' in m and m['e'] is not None:
                    current_move_e_val = m['e']
                    # Write E only if it's different from the last written E value, 
                    # or if no E has been written yet in this sequence of moves.
                    if last_e_val_in_gcode is None or abs(current_move_e_val - last_e_val_in_gcode) > 1e-5: # Use 1e-5 for float comparison
                        e_str = f"E{current_move_e_val:.5f}"
                        # Update last_e_val_in_gcode because we are deciding to write this E value.
                        last_e_val_in_gcode = current_move_e_val
                    # If E is same as last written, e_str remains empty (no redundant E parameter)
                # If a non-travel move has no 'e' in its dictionary (e.g., a G1 Z-only move), 
                # e_str remains empty, which is correct.

            # Add type comments before the G-code line
            # These comments help identify the purpose of the following G-code move.
            if m.get('type') == 'external_perimeter':
                gcode_lines.append(';TYPE:External perimeter\n')
            elif m.get('type') == 'perimeter':
                gcode_lines.append(';TYPE:Perimeter\n')
            # Not adding ;TYPE:Travel as it's usually implicit by lack of E or specific G0 command (though we use G1 for all moves).
            # Other types like 'None' will not have a comment.

            # Construct the G-code line.
            # All moves are currently written as G1, as per the original code's behavior.
            gline_parts = ["G1"]
            if x_str: gline_parts.append(x_str)
            if y_str: gline_parts.append(y_str)
            if z_str: gline_parts.append(z_str)
            if e_str: gline_parts.append(e_str)
            
            gline = " ".join(gline_parts)

            # Only add the line if it's more than just "G1" (i.e., it has parameters).
            # A "G1" line by itself is invalid.
            if len(gline_parts) > 1:
                gcode_lines.append(gline + '\n')
            # If the line was just "G1" but it was supposed to be an E-only move where E didn't change,
            # it's correctly omitted. If it was a travel move with no X,Y,Z change, it's also omitted.

        return gcode_lines

    def save_gcode_file(self):
        if not self.gcode_file_path or self.cleaned_gcode is None:
            QMessageBox.warning(self, "Warning", "No G-code file loaded or cleaned G-code is missing.")
            return
        if not self.selected_layers:
            QMessageBox.warning(self, "Warning", "No layers selected for editing.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save G-code File As", "", "G-code Files (*.gcode *.nc *.txt);;All Files (*)")
        if file_path:
            try:
                # Build new G-code with edits, inserting at correct positions
                new_gcode = []
                layer_ptrs = self.layer_indices + [len(self.cleaned_gcode_with_extruders)]
                for i, (start, end) in enumerate(zip(layer_ptrs[:-1], layer_ptrs[1:])):
                    # Always add ;LAYER_CHANGE at the start of each saved layer
                    # Access the line part of the tuple for comparison and appending
                    if self.cleaned_gcode_with_extruders[start][0].strip() != ';LAYER_CHANGE':
                        new_gcode.append(';LAYER_CHANGE\n')
                    else:
                        new_gcode.append(self.cleaned_gcode_with_extruders[start][0])

                    if i in self.layer_edits:
                        # Extract only the G-code lines (first element of tuples) for parsing moves
                        # The extruder info per line is implicitly handled if parse_moves (called by viewer) stored it in move dicts.
                        # For now, the edit application logic relies on move dicts from layer_edits.
                        original_layer_lines_with_extruders = self.cleaned_gcode_with_extruders[start+1:end]
                        orig_lines = [item[0] for item in original_layer_lines_with_extruders] # Get just the text lines

                        # 1. Parse all moves and collect non-move lines with their indices FROM TEXT LINES
                        moves = []
                        non_move_lines = []  # (idx, line)
                        gcode_to_move_idx = {}  # Map G-code line idx -> move idx
                        move_counter = 0
                        for idx, line in enumerate(orig_lines):
                            l = line.strip()
                            if l.startswith(('G0', 'G1', 'G01', 'G2', 'G3')):
                                # Parse move
                                parts = l.split()
                                move = {'type': None, 'x': None, 'y': None, 'z': None, 'e': None}
                                for part in parts:
                                    if part.startswith('X'):
                                        move['x'] = float(part[1:])
                                    elif part.startswith('Y'):
                                        move['y'] = float(part[1:])
                                    elif part.startswith('Z'):
                                        move['z'] = float(part[1:])
                                    elif part.startswith('E'):
                                        move['e'] = float(part[1:])
                                # Try to infer type from previous comment
                                if idx > 0 and orig_lines[idx-1].strip().startswith(';TYPE:'):
                                    move['type'] = orig_lines[idx-1].strip()[6:].lower()
                                moves.append(move)
                                gcode_to_move_idx[idx] = move_counter
                                move_counter += 1
                            else:
                                non_move_lines.append((idx, line))
                        # 2. Apply all edits to the move list (using move dicts from the 3D viewer)
                        edits = self.layer_edits[i]
                        offset = 0
                        for gcode_idx, edit_moves in sorted(edits, key=lambda x: x[0]):
                            # Map G-code line index to move index
                            move_idx = gcode_to_move_idx.get(gcode_idx, None)
                            if move_idx is None:
                                continue  # Skip if not a move line
                            move_idx += offset
                            for m in edit_moves:
                                m['type'] = 'travel'
                            # Replace the move at move_idx with the edited moves
                            moves[move_idx:move_idx+1] = edit_moves
                            # Patch the next move to be a travel move
                            next_idx = move_idx + len(edit_moves)
                            if next_idx < len(moves):
                                patch = dict(edit_moves[-1])
                                patch['x'] = moves[next_idx]['x']
                                patch['y'] = moves[next_idx]['y']
                                patch['z'] = moves[next_idx]['z']
                                patch['type'] = 'travel'
                                patch['e'] = edit_moves[-1]['e']
                                moves[next_idx] = patch
                            offset += len(edit_moves) - 1
                        # 3. Regenerate G-code for the layer from the move list
                        gcode_lines = self.moves_to_gcode(moves)
                        # 4. Re-insert all non-move lines at their original indices
                        for idx, line in non_move_lines:
                            gcode_lines.insert(idx, line)
                        new_gcode.extend(gcode_lines)
                    else:
                        # Append original lines (text part only) if no edits for this layer
                        original_lines_for_layer = [item[0] for item in self.cleaned_gcode_with_extruders[start+1:end]]
                        new_gcode.extend(original_lines_for_layer)
                with open(file_path, 'w') as file:
                    file.writelines(new_gcode)
                self.status_bar.showMessage(f"Saved: {file_path}")
                QMessageBox.information(self, "Saved", f"G-code saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
        else:
            self.status_bar.showMessage("Save operation cancelled.")

    def view_selected_layer(self):
        if len(self.selected_layers) != 1:
            QMessageBox.warning(self, "Warning", "Please select exactly one layer to view.")
            return
        idx = list(self.selected_layers)[0]
        start = self.layer_indices[idx]
        # Use self.cleaned_gcode_with_extruders to get the end index
        end = self.layer_indices[idx+1] if idx+1 < len(self.layer_indices) else len(self.cleaned_gcode_with_extruders)

        # Pass the list of (line, extruder_command) tuples for the selected layer
        layer_lines_with_extruders = self.cleaned_gcode_with_extruders[start:end]

        moves_override = self.layer_edits.get(idx, None)
        if self.viewer_dialog is None:
            # Pass layer_lines_with_extruders to the viewer's constructor
            self.viewer_dialog = Layer3DViewerDialog(layer_lines_with_extruders, mainwin=self, layer_idx=idx, moves_override=moves_override)
        else:
            # Pass layer_lines_with_extruders to the viewer's set_layer method
            self.viewer_dialog.set_layer(layer_lines_with_extruders, layer_idx=idx, moves_override=moves_override)
        self.viewer_dialog.show()
        self.viewer_dialog.raise_()
        self.viewer_dialog.activateWindow()

    def save_layer_edits(self, layer_idx, moves):
        self.layer_edits[layer_idx] = moves

    def move_index_to_gcode_line(self, move_idx):
        """
        Map a move index (from self.moves) to the corresponding G-code line index in self.layer_lines.
        Returns the G-code line index where the move at move_idx starts.
        """
        move_count = 0
        for i, line in enumerate(self.layer_lines):
            line = line.strip()
            if not line or line.startswith('M'):
                continue
            if line.startswith('G0') or line.startswith('G1') or line.startswith('G01') or line.startswith('G2') or line.startswith('G3'):
                if move_count == move_idx:
                    return i
                move_count += 1
        # If not found, return end of layer
        return len(self.layer_lines)

    def trigger_auto_process_channels(self):
        if not self.cleaned_gcode_with_extruders:
            QMessageBox.warning(self, "Warning", "No G-code file loaded.")
            return

        channel_extruder_cmd = self.channel_extruder_input.text().strip()
        if not channel_extruder_cmd:
            QMessageBox.warning(self, "Input Required", "Please enter the channel extruder command (e.g., T1).")
            return

        # Validate format like T<number>
        if not (channel_extruder_cmd.startswith('T') and channel_extruder_cmd[1:].isdigit()):
            QMessageBox.warning(self, "Invalid Format", "Channel extruder should be in the format T<number> (e.g., T0, T1, T2).")
            return

        QMessageBox.information(self, "Auto-Process", f"Starting auto-processing for channel extruder: {channel_extruder_cmd}")

        channel_segments = self.identify_channel_paths(channel_extruder_cmd)
        if not channel_segments:
            QMessageBox.information(self, "Auto-Process", f"No channel paths found for extruder {channel_extruder_cmd}.")
            return
        else:
            QMessageBox.information(self, "Auto-Process", f"Found {len(channel_segments)} channel segments for {channel_extruder_cmd}. Next step would be extension (pending).")

        # Placeholder for next step:
        self.extend_toolpaths_around_channel(channel_segments, channel_extruder_cmd)

    def _parse_layer_moves(self, layer_target_idx):
        """
        Parses moves for a specific layer index (from GCodeEditor's layer_indices).
        Returns a list of tuples: (move_dict, original_line_idx_in_layer_lines).
        move_dict includes 'x', 'y', 'z', 'e', 'type', 'extruder'.
        'original_line_idx_in_layer_lines' is the index relative to the start of that layer's G-code lines
        (excluding the initial ';LAYER_CHANGE' line itself).
        """
        if layer_target_idx < 0 or layer_target_idx >= len(self.layer_indices):
            return []

        layer_start_gcode_idx = self.layer_indices[layer_target_idx]
        layer_end_gcode_idx = self.layer_indices[layer_target_idx + 1] if layer_target_idx + 1 < len(self.layer_indices) else len(self.cleaned_gcode_with_extruders)

        # Get (line_text, extruder_command) tuples for the layer, skipping the initial ';LAYER_CHANGE'
        # The gcode_idx for layer_edits should be relative to these layer_lines.
        layer_lines_with_extruders_tuples = self.cleaned_gcode_with_extruders[layer_start_gcode_idx + 1 : layer_end_gcode_idx]

        moves_with_original_indices = []
        x = y = z = e = None
        last_x = last_y = last_z = last_e = None # State for current layer's parsing
        current_type = None # For things like ;TYPE:Perimeter

        # Need to track the active extruder within this layer based on T commands if they exist IN THIS LAYER's lines
        # However, self.assign_extruders_to_lines should have already given us extruder_for_line correctly.

        for idx_in_layer, (line_text, extruder_for_line) in enumerate(layer_lines_with_extruders_tuples):
            line = line_text.strip()

            # Similar parsing logic to Layer3DViewerDialog.parse_moves
            if line.startswith(';TYPE:External perimeter'):
                current_type = 'external_perimeter'; continue
            elif line.startswith(';TYPE:Perimeter'):
                current_type = 'perimeter'; continue
            elif line.startswith(';TYPE:'):
                current_type = None; continue

            if not line or line.startswith('M') or (line.startswith('T') and line[1:].isdigit()):
                # Update last_x,y,z,e if a T command implies a new tool head position (rarely specified in G-code)
                # For now, assume T commands don't change XYZ position.
                continue

            if line.startswith(('G0', 'G1', 'G01')):
                parts = line.split()
                move_x, move_y, move_z, move_e = None, None, None, None # Params for this specific line
                has_xy_move = False
                has_e_move = False

                for part in parts:
                    if part.startswith('X'): move_x = float(part[1:])
                    elif part.startswith('Y'): move_y = float(part[1:])
                    elif part.startswith('Z'): move_z = float(part[1:])
                    elif part.startswith('E'):
                        try: move_e = float(part[1:])
                        except ValueError: pass

                # Update coordinates: if not present in current G-code, use last known
                if move_x is not None: x = move_x; has_xy_move = True
                if move_y is not None: y = move_y; has_xy_move = True
                if move_z is not None: z = move_z
                if move_e is not None: e = move_e; has_e_move = True

                if has_xy_move: # Only consider it a "move" for our list if X or Y changes
                    is_travel = False
                    if line.startswith('G0'): is_travel = True
                    # For G1, travel if no E or E is not changing significantly
                    elif not has_e_move or (last_e is not None and abs(e - last_e) < 1e-5):
                        is_travel = True

                    # Ensure z is not None (e.g. from previous moves or layer start Z)
                    current_z_for_move = z if z is not None else (last_z if last_z is not None else 0)
                    current_e_for_move = e if e is not None else (last_e if last_e is not None else 0)

                    move_dict = {
                        'x': x, 'y': y, 'z': current_z_for_move, 'e': current_e_for_move,
                        'type': 'travel' if is_travel else current_type,
                        'extruder': extruder_for_line,
                        'original_line_in_layer': idx_in_layer # Store its original index within the layer's lines
                    }
                    moves_with_original_indices.append(move_dict)

                    # Update last known state for next iteration
                    if x is not None: last_x = x
                    if y is not None: last_y = y
                    if z is not None: last_z = z # z here is the 'current z for move' effectively
                    if e is not None: last_e = e # e here is the 'current e for move'
                elif move_z is not None: # Z-only move
                    last_z = z # Update Z, but don't add to moves list unless it's an XY move

            # G2/G3 Arc moves - simplified for now, treat like G1 for XY endpoint
            elif line.startswith(('G2', 'G3')):
                parts = line.split()
                move_x, move_y, move_z, move_e = None, None, None, None
                has_xy_move = False; has_e_move = False
                for part in parts:
                    if part.startswith('X'): move_x = float(part[1:])
                    elif part.startswith('Y'): move_y = float(part[1:])
                    elif part.startswith('Z'): move_z = float(part[1:])
                    elif part.startswith('E'):
                        try: move_e = float(part[1:])
                        except ValueError: pass

                if move_x is not None: x = move_x; has_xy_move = True
                if move_y is not None: y = move_y; has_xy_move = True
                if move_z is not None: z = move_z
                if move_e is not None: e = move_e; has_e_move = True

                if has_xy_move:
                    is_travel = not has_e_move or (last_e is not None and abs(e - last_e) < 1e-5)
                    current_z_for_move = z if z is not None else (last_z if last_z is not None else 0)
                    current_e_for_move = e if e is not None else (last_e if last_e is not None else 0)
                    move_dict = {
                        'x': x, 'y': y, 'z': current_z_for_move, 'e': current_e_for_move,
                        'type': 'travel' if is_travel else current_type, # G2/G3 are usually not 'perimeter' typed by slicer comments
                        'extruder': extruder_for_line,
                        'original_line_in_layer': idx_in_layer
                    }
                    moves_with_original_indices.append(move_dict)
                    if x is not None: last_x = x
                    if y is not None: last_y = y
                    if z is not None: last_z = z
                    if e is not None: last_e = e

        return moves_with_original_indices


    def extend_toolpaths_around_channel(self, channel_segments, channel_extruder_cmd_text):
        """
        Extends toolpaths in layers above and below the identified channel segments.
        Modifications are prepared for self.layer_edits.
        """
        EXTENSION_DISTANCE = 0.5  # mm
        PROXIMITY_THRESHOLD = 1.0 # mm, how close a path end needs to be to a channel end

        if not channel_segments:
            return

        # Group channel segments by layer index for easier processing if needed, though iterating through is fine

        for segment_idx, chan_segment in enumerate(channel_segments):
            layer_channel_idx = chan_segment['layer_index']

            # Define layers to check: one below, one above
            layers_to_process = []
            if layer_channel_idx > 0:
                layers_to_process.append(layer_channel_idx - 1)  # Layer below
            if layer_channel_idx < len(self.layer_indices) - 1:
                layers_to_process.append(layer_channel_idx + 1)  # Layer above

            for adj_layer_idx in layers_to_process:
                if adj_layer_idx < 0 or adj_layer_idx >= len(self.layer_indices): # Should be caught by above but double check
                    continue

                adj_layer_moves = self._parse_layer_moves(adj_layer_idx)
                if not adj_layer_moves:
                    continue

                # Store edits for this adjacent layer: {gcode_line_idx_in_layer: [new_move_dicts]}
                # This will be merged into self.layer_edits later
                current_layer_modifications = {}

                # Iterate through moves in the adjacent layer
                # We need the previous move to know the start of the current segment being checked
                prev_adj_move = None
                for i in range(len(adj_layer_moves)):
                    current_adj_move_dict = adj_layer_moves[i]

                    # Skip moves by the channel extruder itself in adjacent layers (unlikely but possible)
                    if current_adj_move_dict['extruder'] == channel_extruder_cmd_text:
                        if prev_adj_move: prev_adj_move = current_adj_move_dict # update prev before skipping
                        else: prev_adj_move = current_adj_move_dict
                        continue

                    # We need a start point for the current adjacent move segment
                    if prev_adj_move is None:
                        # Try to find a G1 X Y Z E line before this to establish a start point,
                        # or use (0,0, Z_of_current_move) if it's the very first move.
                        # For simplicity, if it's the first move, we can't easily determine its incoming direction
                        # to extend "backwards". So, we'll focus on extending the *end* of moves.
                        # If we need to extend starts, we'd need more context or make assumptions.
                        # Let's prime prev_adj_move here. If this is the first move, it has no "previous".
                        # The first move in adj_layer_moves *is* the first G1/G0 with XY.
                        # Its 'start' is effectively the state *before* it.
                        # This parsing logic needs to be more robust to get start point of first move.
                        # For now, this means we can only reliably extend the *end* of `prev_adj_move`
                        # when `current_adj_move_dict` is being processed.

                        # Let's adjust: a "segment" is from prev_adj_move to current_adj_move_dict
                        # So, we process when we have at least one point.
                        # The `current_adj_move_dict` represents the *end point* of a path segment.
                        # The `prev_adj_move` (if exists) is the *start point*.

                        # We are interested in the segment ending at current_adj_move_dict['x'], ['y']
                        # and starting at prev_adj_move['x'], ['y'] (if prev_adj_move exists)

                        # Simpler: consider each move in adj_layer_moves. Its (x,y,z) are its end point.
                        # The start point is from the state *before* this move command.
                        # Our _parse_layer_moves gives move_dict which has its end coords.
                        # To get start coords of a move, we need the end_coords of the *previous* move in the list.

                        # Let current_adj_path_segment_end be (current_adj_move_dict['x'], current_adj_move_dict['y'])
                        # Let current_adj_path_segment_start be (prev_adj_move['x'], prev_adj_move['y']) if prev_adj_move else some default/origin

                        if i == 0: # First move in the layer
                            # Cannot easily determine its start point from previous G-code line in this simplified parsing.
                            # So, we can't extend the "start" of the first path segment.
                            # We *can* extend the *end* of this first path segment.
                            prev_adj_move = current_adj_move_dict # current becomes prev for next iteration.
                            continue # Process its end on the NEXT iteration, or handle specially if it's also the last.

                    # Now, prev_adj_move is the start of the segment, current_adj_move_dict is the end.
                    # Segment is from (prev_adj_move['x'], prev_adj_move['y']) to (current_adj_move_dict['x'], current_adj_move_dict['y'])

                    # Check proximity of current_adj_move_dict (end of adj path) to chan_segment start/end
                    adj_path_end_pt = np.array([current_adj_move_dict['x'], current_adj_move_dict['y']])
                    chan_start_pt = np.array(chan_segment['start_xy'])
                    chan_end_pt = np.array(chan_segment['end_xy'])

                    # Is the adjacent path's END near the CHANNEL's START?
                    dist_adj_end_to_chan_start = np.linalg.norm(adj_path_end_pt - chan_start_pt)
                    # Is the adjacent path's END near the CHANNEL's END?
                    dist_adj_end_to_chan_end = np.linalg.norm(adj_path_end_pt - chan_end_pt)

                    target_channel_pt = None
                    if dist_adj_end_to_chan_start < PROXIMITY_THRESHOLD:
                        target_channel_pt = chan_start_pt
                    elif dist_adj_end_to_chan_end < PROXIMITY_THRESHOLD:
                        target_channel_pt = chan_end_pt

                    if target_channel_pt is not None:
                        # The segment ending at current_adj_move_dict is close to a channel end.
                        # We want to extend this segment.
                        adj_path_start_pt = np.array([prev_adj_move['x'], prev_adj_move['y']])
                        original_adj_segment_vec = adj_path_end_pt - adj_path_start_pt
                        original_adj_segment_len = np.linalg.norm(original_adj_segment_vec)

                        if original_adj_segment_len > 1e-5: # Avoid division by zero for zero-length segments
                            adj_segment_unit_vec = original_adj_segment_vec / original_adj_segment_len

                            # New end point for the adjacent path segment
                            new_adj_path_end_pt = adj_path_end_pt + adj_segment_unit_vec * EXTENSION_DISTANCE

                            # Create a new move dictionary for the extended move.
                            # This new move replaces `current_adj_move_dict`.
                            modified_move = dict(current_adj_move_dict) # Copy
                            modified_move['x'] = new_adj_path_end_pt[0]
                            modified_move['y'] = new_adj_path_end_pt[1]

                            # Recalculate E value
                            original_extrusion_amount_for_segment = current_adj_move_dict['e'] - prev_adj_move['e']
                            if abs(original_extrusion_amount_for_segment) > 1e-5 and not current_adj_move_dict.get('type') == 'travel':
                                new_segment_len = np.linalg.norm(new_adj_path_end_pt - adj_path_start_pt)
                                # if original_adj_segment_len was also checked > 1e-5
                                new_extrusion_amount = original_extrusion_amount_for_segment * (new_segment_len / original_adj_segment_len)
                                modified_move['e'] = prev_adj_move['e'] + new_extrusion_amount
                            elif current_adj_move_dict.get('type') == 'travel':
                                modified_move['e'] = prev_adj_move['e'] # Maintain E for travel
                            # else: E was not changing, or it's the first extrusion, complex case. Keep E as is for now.

                            # Store this modification
                            # Key is the original G-code line index *within the layer's lines*
                            original_gcode_idx_in_layer = current_adj_move_dict['original_line_in_layer']

                            if adj_layer_idx not in self.layer_edits:
                                self.layer_edits[adj_layer_idx] = []

                            # The format for layer_edits is a list of (gcode_idx, [list_of_new_move_dicts])
                            # We are replacing one move with one modified move.
                            # Need to ensure we don't add duplicate gcode_idx entries if multiple channel segments affect the same adj move.
                            # This simple approach might overwrite if that happens. A robust solution would merge edits.
                            # For now, let's assume one modification per original line for simplicity of this first pass.

                            # Remove existing edit for this line if present, then add new one.
                            self.layer_edits[adj_layer_idx] = [
                                edit for edit in self.layer_edits[adj_layer_idx]
                                if edit[0] != original_gcode_idx_in_layer
                            ]
                            self.layer_edits[adj_layer_idx].append((original_gcode_idx_in_layer, [modified_move]))

                            # Debugging print removed.
                            # print(f"Layer {adj_layer_idx}: Modifying line {original_gcode_idx_in_layer} (orig end {adj_path_end_pt}) to new end {new_adj_path_end_pt.round(3)}")


                    prev_adj_move = current_adj_move_dict # Current becomes previous for next iteration

        if self.layer_edits:
             QMessageBox.information(self, "Auto-Process", f"Toolpath extension logic applied. {len(self.layer_edits)} layers have modifications prepared. Save to apply.")
        else:
            QMessageBox.information(self, "Auto-Process", "No toolpaths were identified for extension.")


    def identify_channel_paths(self, target_extruder_cmd):
        """
        Identifies G1 move segments made by the target_extruder_cmd.
        Returns a list of dictionaries, each representing a channel segment.
        Segment: {'layer_index': int, 'start_xy': (float, float), 'end_xy': (float, float), 'z': float, 'extruder': str}
        """
        if not self.cleaned_gcode_with_extruders:
            return []

        channel_segments = []
        current_layer_index = -1
        last_x, last_y, last_z = None, None, None

        for line_idx, (line_text, line_extruder_cmd) in enumerate(self.cleaned_gcode_with_extruders):
            line = line_text.strip()

            if line == ';LAYER_CHANGE':
                current_layer_index += 1
                # Reset last known coordinates at layer change, as they are layer-specific
                last_x, last_y, last_z = None, None, None
                # PrusaSlicer and other slicers might issue a G1 Z move after LAYER_CHANGE
                # So, we need to parse G1 immediately following to catch the Z.
                # However, the first actual XY move will establish the real start for that layer.

            if line.startswith(('G0', 'G1')): # Considering G0 as potential travel, G1 as extrusion/travel
                parts = line.split()
                current_x, current_y, current_z, current_e = None, None, None, None

                for part in parts:
                    if part.startswith('X'):
                        current_x = float(part[1:])
                    elif part.startswith('Y'):
                        current_y = float(part[1:])
                    elif part.startswith('Z'):
                        current_z = float(part[1:])
                    # E is not directly used for path definition but good to parse
                    elif part.startswith('E'):
                        try:
                            current_e = float(part[1:])
                        except ValueError:
                            pass # Ignore if E is not a float

                # Update Z if present, otherwise carry over last known Z. Crucial for first Z of a layer.
                if current_z is not None:
                    last_z = current_z

                # An XY movement is required to define a segment
                if current_x is not None and current_y is not None:
                    if line_extruder_cmd == target_extruder_cmd:
                        # This move is by the target extruder.
                        # Requires a previous point to form a segment.
                        if last_x is not None and last_y is not None and last_z is not None: # Ensure Z is known
                            # Check if it's an extrusion move (E value changed or is present and positive)
                            # For channel paths, we are typically interested in extrusion moves.
                            # However, the prompt implies any path by the channel extruder.
                            # Let's assume any G1 XY move by the channel extruder is part of the channel.
                            # A more sophisticated check for actual extrusion (E value increasing) could be added if needed.
                            is_extrusion_move = False # Default
                            if 'E' in line and current_e is not None: # E parameter is present
                                # A simple check: if E is present, assume extrusion for the channel path.
                                # Slicers might use G1 for travel without E, or with E not changing.
                                # For defining the channel itself, we primarily care about where material is laid.
                                # This needs careful thought: if channel extruder makes a G1 travel move, is it part of "channel"?
                                # For now, let's be inclusive if it's the target extruder.
                                is_extrusion_move = True # Simplified: if it's G1 and target extruder, count it.

                            # For now, any G1/G0 XY move by the channel extruder is considered.
                            # We need a start (last_x, last_y) and end (current_x, current_y)
                            segment = {
                                'layer_index': current_layer_index,
                                'start_xy': (last_x, last_y),
                                'end_xy': (current_x, current_y),
                                'z': last_z, # Z of the segment start
                                'extruder': line_extruder_cmd
                            }
                            channel_segments.append(segment)

                    # Update last known X, Y for the next segment calculation, regardless of extruder
                    last_x, last_y = current_x, current_y
                elif current_z is not None and last_x is None and last_y is None:
                    # This handles cases like G1 Z<val> at the start of a layer before any XY moves.
                    # We just update last_z. last_x, last_y remain None.
                    pass


        return channel_segments


class LayerSelectorDialog(QDialog):
    def __init__(self, layer_labels, selected_layers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Layers")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        for idx, label in enumerate(layer_labels):
            self.list_widget.addItem(label)
            if idx in selected_layers:
                self.list_widget.item(idx).setSelected(True)
        layout.addWidget(self.list_widget)
        btn_box = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)
    def get_selected_layers(self):
        return set([i.row() for i in self.list_widget.selectedIndexes()])

class Layer3DViewerDialog(QDialog):
    def __init__(self, layer_lines_with_extruders, parent=None, mainwin=None, layer_idx=None, moves_override=None):
        super().__init__(parent)
        self.mainwin = mainwin
        self.layer_idx = layer_idx if layer_idx is not None else -1
        global viewer_open_count
        viewer_open_count += 1
        if viewer_open_count == 1:
            self.extruder_head_style = 'sphere'
        else:
            self.extruder_head_style = 'square'
        self.setWindowTitle("3D Layer Viewer")
        self.setMinimumSize(900, 700)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)
        self.layer_lines_with_extruders = layer_lines_with_extruders # Store the new structure
        self.moves = self.parse_moves(self.layer_lines_with_extruders) # Pass it to parse_moves
        self.edit_sessions = []  # List of dicts: {'origin_idx', 'current_idx', 'color', 'move_stack', 'origin_coords'}
        self.session_colors = [
            (1,0,0,1),      # Red
            (1,0.5,0,1),    # Orange
            (1,1,0,1),      # Yellow
            (0,1,0,1),      # Green
            (0,0,1,1),      # Blue
            (0.5,0,1,1),    # Purple
            (1,0,1,1),      # Magenta
            (0,1,1,1),      # Cyan
        ]
        self.session_color_idx = 0
        self.manual_origin_idx = None
        self.manual_current_idx = None
        self.manual_move_stack = []
        self.editor_active = False
        self.current_index = len(self.moves) if self.moves else 0
        layout = QVBoxLayout(self)
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('w')
        layout.addWidget(self.gl_widget, stretch=1)
        # Add Toolpath Editor button and D-pad
        self.editor_button = QPushButton("Enable Toolpath Editor")
        self.editor_button.setFixedWidth(270)  # Even wider for text
        self.editor_button.setFixedHeight(38)
        self.editor_button.clicked.connect(self.toggle_editor)
        # Place in a horizontal layout, right-aligned
        top_bar = QHBoxLayout()
        top_bar.addStretch(1)
        top_bar.addWidget(self.editor_button)
        layout.addLayout(top_bar)
        # D-Pad controls (hidden by default)
        self.dpad_widget = QWidget()
        dpad_layout = QGridLayout()
        dpad_layout.setContentsMargins(0,0,0,0)
        dpad_layout.setSpacing(2)
        self.dpad_buttons = {}
        directions = {
            (0,1): ("▲", "up"),
            (0,2): ("↗", "up_right"),
            (1,0): ("◀", "left"),
            (1,1): ("●", None),
            (1,2): ("▶", "right"),
            (2,0): ("↙", "down_left"),
            (2,1): ("▼", "down"),
            (2,2): ("↘", "down_right"),
            (0,0): ("↖", "up_left"),
        }
        for (row, col), (label, direction) in directions.items():
            btn = QPushButton(label)
            btn.setFixedSize(32,32)
            if direction:
                btn.clicked.connect(lambda checked, d=direction: self.move_extruder(d))
            self.dpad_buttons[direction] = btn
            dpad_layout.addWidget(btn, row, col)
        self.dpad_widget.setLayout(dpad_layout)
        self.dpad_widget.setVisible(False)
        # Place D-pad in a horizontal layout, right-aligned
        dpad_bar = QHBoxLayout()
        dpad_bar.addStretch(1)
        dpad_bar.addWidget(self.dpad_widget)
        layout.addLayout(dpad_bar)
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(len(self.moves) if self.moves else 1)
        self.slider.setValue(self.current_index)
        self.slider.valueChanged.connect(self.update_plot)
        slider_layout.addWidget(self.slider)
        self.status_label = QLabel()
        slider_layout.addWidget(self.status_label)
        self.add_slider_arrows(slider_layout)
        layout.addLayout(slider_layout)
        self.update_plot()
        # Widen Minimize/Maximize and Save Edits buttons
        self.minmax_button = QPushButton("Minimize/Maximize")
        self.minmax_button.setFixedWidth(220)
        self.minmax_button.clicked.connect(self.toggle_minmax)
        minmax_bar = QHBoxLayout()
        minmax_bar.addStretch(1)
        minmax_bar.addWidget(self.minmax_button)
        layout.insertLayout(0, minmax_bar)
        self.save_edits_button = QPushButton("Save Edits (in viewer)")
        self.save_edits_button.setFixedWidth(260)
        self.save_edits_button.clicked.connect(self.save_edits)
        save_bar = QHBoxLayout()
        save_bar.addStretch(1)
        save_bar.addWidget(self.save_edits_button)
        layout.insertLayout(1, save_bar)
        # Set initial state to minimized (25%) and initialize toggle state
        screen = QApplication.primaryScreen().availableGeometry()
        w, h = int(screen.width()*0.25), int(screen.height()*0.25)
        self.resize(w, h)
        self.move((screen.width()-w)//2, (screen.height()-h)//2)
        self._is_custom_maximized = False
        self._has_been_shown = False
        self.edit_inserts = []  # Track all (insert_idx, moves) for this layer

    def move_extruder(self, direction):
        if not self.dpad_widget.isVisible():
            return
        if not self.moves:
            return
        idx = self.slider.value()
        dir_vectors = {
            "up":      np.array([-2.0, 0]),
            "down":    np.array([2.0, 0]),
            "left":    np.array([0, -2.0]),
            "right":   np.array([0, 2.0]),
            "up_right":   np.array([-2.0, 2.0]) / 1.414,
            "up_left":    np.array([-2.0, -2.0]) / 1.414,
            "down_right": np.array([2.0, 2.0]) / 1.414,
            "down_left":  np.array([2.0, -2.0]) / 1.414,
        }
        move_vec = dir_vectors[direction]
        session = self.edit_sessions[-1]
        # If moving back cancels last move, pop it
        if session['move_stack']:
            last_vec = session['move_stack'][-1]
            if np.allclose(move_vec, -last_vec):
                self.moves.pop(idx-1)
                self.current_index -= 1
                self.slider.setMaximum(len(self.moves))
                self.slider.setValue(idx-1)
                session['move_stack'].pop()
                session['current_idx'] = self.slider.value()-1
                if not session['move_stack']:
                    session['origin_idx'] = None
                    session['current_idx'] = None
                self.manual_origin_idx = session['origin_idx']
                self.manual_current_idx = session['current_idx']
                self.manual_move_stack = session['move_stack']
                self.update_plot()
                return
        # Otherwise, add new lifted travel move (sloped, no drop)
        last_move = self.moves[idx-1].copy()
        orig_x, orig_y, orig_z = last_move['x'], last_move['y'], last_move['z']
        dx, dy = move_vec[0], move_vec[1]
        lift_height = 1.5
        xy_dist = np.linalg.norm([dx, dy])
        slope_angle_rad = np.deg2rad(30)
        slope_dz = np.tan(slope_angle_rad) * xy_dist
        actual_lift = min(lift_height, slope_dz)
        # Move in XY and Z (sloped lift)
        lifted_move = last_move.copy()
        lifted_move['x'] += dx
        lifted_move['y'] += dy
        lifted_move['z'] = orig_z + actual_lift
        # Always set travel move type and do not extrude
        lifted_move['type'] = 'travel'
        lifted_move['e'] = last_move['e']  # No extrusion for travel
        self.moves.insert(idx, lifted_move)
        session['move_stack'].append(move_vec)
        session['current_idx'] = idx
        # Record the edit as (insert_idx, [move_dicts])
        insert_idx = self.slider.value() - 1 if self.slider.value() > 0 else 0
        # Map move index to G-code line index for precise insertion
        move_idx = self.slider.value() - 1 if self.slider.value() > 0 else 0
        gcode_line_idx = self.move_index_to_gcode_line(move_idx)
        moves_to_insert = [self.moves[move_idx]]
        # --- Multi-move edit support ---
        # If this is the first move in a new edit session, start a new edit_inserts entry
        if not self.edit_inserts or self.edit_inserts[-1][0] != self.move_index_to_gcode_line(idx-1):
            gcode_line_idx = self.move_index_to_gcode_line(idx-1)
            self.edit_inserts.append((gcode_line_idx, [lifted_move]))
        else:
            # Append to the last edit's move list
            self.edit_inserts[-1][1].append(lifted_move)
        self.manual_origin_idx = session['origin_idx']
        self.manual_current_idx = session['current_idx']
        self.manual_move_stack = session['move_stack']
        self.current_index += 1
        self.slider.setMaximum(len(self.moves))
        self.slider.setValue(idx+1)
        # If back at origin, reset indices and stack for this session
        origin = np.array([
            self.moves[session['origin_idx']]['x'],
            self.moves[session['origin_idx']]['y'],
            self.moves[session['origin_idx']]['z']
        ])
        current = np.array([
            lifted_move['x'], lifted_move['y'], lifted_move['z']
        ])
        if np.allclose(origin, current):
            session['origin_idx'] = None
            session['current_idx'] = None
            session['move_stack'] = []
            self.manual_origin_idx = None
            self.manual_current_idx = None
            self.manual_move_stack = []
        self.update_plot()

    def toggle_editor(self):
        if not self.dpad_widget.isVisible():
            self.dpad_widget.setVisible(True)
            self.editor_button.setText("Stop Editing")
            self.editor_button.setStyleSheet("background-color: red; color: white;")
            self.editor_active = True
            # Start a new edit session with a fixed color and its own origin
            idx = self.slider.value()-1
            # Assign color based on the number of sessions already created (fixed per session)
            color = self.session_colors[len(self.edit_sessions) % len(self.session_colors)]
            self.manual_origin_idx = idx
            self.manual_current_idx = idx
            self.manual_move_stack = []
            self.edit_sessions.append({
                'origin_idx': idx,
                'current_idx': idx,
                'color': color,
                'move_stack': [],
                'origin_coords': np.array([
                    self.moves[idx]['x'],
                    self.moves[idx]['y'],
                    self.moves[idx]['z']
                ])
            })
        else:
            self.dpad_widget.setVisible(False)
            self.editor_button.setText("Enable Toolpath Editor")
            self.editor_button.setStyleSheet("")
            self.editor_active = False
            #self.end_editing_and_patch_path()
        self.update_plot()

    def update_plot(self):
        self.gl_widget.clear()
        idx = self.slider.value()
        if not self.moves or idx < 2:
            self.status_label.setText("Not enough moves to display.")
            return
        pts = np.array([[m['x'], m['y'], m['z']] for m in self.moves[:idx]])
        # Add axes/grid for reference
        grid = gl.GLGridItem()
        grid.setSize(x=200, y=200)
        grid.setSpacing(x=10, y=10)
        self.gl_widget.addItem(grid)
        grid.translate((pts[:,0].min()+pts[:,0].max())/2, (pts[:,1].min()+pts[:,1].max())/2, 0)
        # Draw toolpath segments by type
        for i in range(1, idx):
            seg = np.array([pts[i-1], pts[i]])
            move_type = self.moves[i]['type']
            # Default: gray for unspecified
            color = (0.5,0.5,0.5,1)
            width = 3
            antialias = True
            if move_type == 'external_perimeter':
                color = (0.5,0,0.5,1)  # Purple
            elif move_type == 'perimeter':
                color = (0,0,1,1)  # Blue
            elif move_type == 'travel':
                color = (0,1,0,1)  # Green
                width = 2
                antialias = False
                # Dotted line: use many short segments
                dots = np.linspace(0, 1, 10)
                for j in range(0, len(dots)-1, 2):
                    dot_seg = np.vstack([
                        seg[0] + (seg[1] - seg[0]) * dots[j],
                        seg[0] + (seg[1] - seg[0]) * dots[j+1]
                    ])
                    dot_line = gl.GLLinePlotItem(pos=dot_seg, color=color, width=width, antialias=antialias, mode='lines')
                    dot_line.setGLOptions('translucent')
                    self.gl_widget.addItem(dot_line)
                continue  # Skip normal line for travel
            plt = gl.GLLinePlotItem(pos=seg, color=color, width=width, antialias=antialias, mode='lines')
            self.gl_widget.addItem(plt)
        # Draw all edit session net moves as colored lines (only if editor is active)
        if self.editor_active:
            for session in self.edit_sessions:
                # Only draw if session has a net move
                if session['origin_idx'] is not None and session['current_idx'] is not None:
                    # Only draw if both indices are within the current slider range
                    if session['origin_idx'] < len(self.moves) and session['current_idx'] < len(self.moves):
                        origin = session['origin_coords']
                        current = np.array([
                            self.moves[session['current_idx']]['x'],
                            self.moves[session['current_idx']]['y'],
                            self.moves[session['current_idx']]['z']
                        ])
                        if not np.allclose(origin, current):
                            seg = np.array([origin, current])
                            color = session['color']
                            line = gl.GLLinePlotItem(pos=seg, color=color, width=7, antialias=True, mode='lines')
                            self.gl_widget.addItem(line)
        # Draw extruder position: use sphere for first open, square for subsequent opens
        last = pts[idx-1]
        if self.extruder_head_style == 'sphere':
            try:
                md = gl.MeshData.sphere(rows=20, cols=20, radius=0.625)
                sphere = gl.GLMeshItem(meshdata=md, color=(1,0,0,1), smooth=True, shader='shaded', drawEdges=False)
                sphere.translate(last[0], last[1], last[2])
                sphere.setGLOptions('opaque')
                self.gl_widget.addItem(sphere)
            except Exception as e:
                scatter = gl.GLScatterPlotItem(pos=np.array([last]), color=(1,0,0,1), size=20, pxMode=True)
                scatter.setGLOptions('opaque')
                self.gl_widget.addItem(scatter)
        else:
            scatter = gl.GLScatterPlotItem(pos=np.array([last]), color=(1,0,0,1), size=20, pxMode=True)
            scatter.setGLOptions('opaque')
            self.gl_widget.addItem(scatter)
        self.status_label.setText(f"Showing {idx} moves / {len(self.moves)}")
        # Set camera to fit the data
        x_range = np.ptp(pts[:,0])
        y_range = np.ptp(pts[:,1])
        max_range = max(x_range, y_range, 100)
        center_coords = [(pts[:,0].min()+pts[:,0].max())/2, (pts[:,1].min()+pts[:,1].max())/2, (pts[:,2].min()+pts[:,2].max())/2]
        center = pg.Vector(center_coords[0], center_coords[1], center_coords[2])
        self.gl_widget.setCameraPosition(pos=center, distance=max_range, elevation=90, azimuth=0)
        self.gl_widget.setBackgroundColor('w')

    def add_slider_arrows(self, layout):
        arrow_back = QPushButton('◀')
        arrow_back.setFixedWidth(32)
        arrow_forward = QPushButton('▶')
        arrow_forward.setFixedWidth(32)
        arrow_back.clicked.connect(self.slider_back)
        arrow_forward.clicked.connect(self.slider_forward)
        layout.insertWidget(0, arrow_back)
        layout.addWidget(arrow_forward)
        self.arrow_back = arrow_back
        self.arrow_forward = arrow_forward

    def slider_back(self):
        val = self.slider.value()
        if val > self.slider.minimum():
            self.slider.setValue(val-1)

    def slider_forward(self):
        val = self.slider.value()
        if val < self.slider.maximum():
            self.slider.setValue(val+1)

    def set_layer(self, layer_lines_with_extruders, layer_idx=None, moves_override=None): # Changed parameter name
        self.layer_lines_with_extruders = layer_lines_with_extruders # Store the new structure
        if layer_idx is not None:
            self.layer_idx = layer_idx
        elif not hasattr(self, 'layer_idx'):
            self.layer_idx = -1 # Should ideally always have a layer_idx if layer is set

        if moves_override is not None:
            # moves_override are already parsed move dictionaries, so just use them.
            self.moves = [dict(m) for m in moves_override] # Ensure they are copied
        else:
            # Parse from the new structure
            self.moves = self.parse_moves(self.layer_lines_with_extruders)
        self.current_index = len(self.moves) if self.moves else 0
        self.slider.setMaximum(len(self.moves) if self.moves else 1)
        self.slider.setValue(self.current_index)
        self.edit_sessions = []
        self.manual_origin_idx = None
        self.manual_current_idx = None
        self.manual_move_stack = []
        self.session_color_idx = 0
        self.update_plot()

    def parse_moves(self, lines_with_extruders): # Parameter changed
        moves = []
        x = y = z = e = None
        last_x = last_y = last_z = last_e = None
        current_type = None
        # current_extruder will be set by the T command from lines_with_extruders
        # However, GCodeEditor.assign_extruders_to_lines already provides this per line.

        for line_item in lines_with_extruders: # Iterate through tuples
            line_text, extruder_for_line = line_item # Unpack the tuple
            line = line_text.strip()

            if not line or line.startswith('M'): # Skip M codes for move parsing, T codes are handled by extruder_for_line
                # Tool change commands (T0, T1 etc.) are not moves themselves.
                # Their effect (active extruder) is already captured in extruder_for_line.
                if line.startswith('T') and line[1:].isdigit(): # Example: T0
                    # This is just confirming we see it, but extruder_for_line is the source of truth for the move.
                    pass
                else:
                    continue # Skip other M codes or empty/comment lines for move parsing

            if line.startswith(';TYPE:External perimeter'):
                current_type = 'external_perimeter'
                continue
            elif line.startswith(';TYPE:Perimeter'):
                current_type = 'perimeter'
                continue
            elif line.startswith(';TYPE:'): # Clear type if a generic ;TYPE: comment is encountered
                current_type = None
                continue

            # G0, G1, G01 are motion commands
            if line.startswith('G0') or line.startswith('G1') or line.startswith('G01'):
                parts = line.split()
                for part in parts:
                    if part.startswith('X'):
                        x = float(part[1:])
                    elif part.startswith('Y'):
                        y = float(part[1:])
                    elif part.startswith('Z'):
                        z = float(part[1:])
                    elif part.startswith('E'):
                        try:
                            e = float(part[1:])
                        except ValueError:
                            pass
                is_travel = False
                if last_e is not None and (e is None or e == last_e): # Basic travel detection
                    is_travel = True

                # More robust travel detection: G0 is often explicitly travel
                if line.startswith('G0') and not any(p.startswith('E') for p in parts if p and p[0] == 'E' and len(p) > 1): # G0 without E is travel
                    is_travel = True

                # If G1 has no E, it's also travel (unless it's a Z-only move, which can be travel or part of print)
                # The existing logic e is None or e == last_e handles this well for G1.

                if x is not None and y is not None: # A move must have at least X and Y
                    move_dict = {
                        'x': x, 'y': y,
                        'z': z if z is not None else (last_z if last_z is not None else 0),
                        'e': e if e is not None else (last_e if last_e is not None else 0),
                        'type': 'travel' if is_travel else current_type,
                        'extruder': extruder_for_line # Store the active extruder command
                    }
                    moves.append(move_dict)
                    last_x, last_y, last_z, last_e = x, y, z if z is not None else last_z, e if e is not None else last_e
            elif line.startswith('G2') or line.startswith('G3'): # Arc moves
                parts = line.split()
                for part in parts:
                    if part.startswith('X'):
                        x = float(part[1:])
                    elif part.startswith('Y'):
                        y = float(part[1:])
                    elif part.startswith('Z'):
                        z = float(part[1:])
                    elif part.startswith('E'):
                        try:
                            e = float(part[1:])
                        except ValueError:
                            pass
                is_travel = False
                if last_e is not None and (e is None or e == last_e): # Basic travel detection
                    is_travel = True
                # Arc moves (G2/G3) usually imply extrusion if E is present and changing.
                # If E is not present or not changing, it could be a travel arc.

                if x is not None and y is not None: # A move must have at least X and Y
                    move_dict = {
                        'x': x, 'y': y,
                        'z': z if z is not None else (last_z if last_z is not None else 0),
                        'e': e if e is not None else (last_e if last_e is not None else 0),
                        'type': 'travel' if is_travel else current_type, # Assuming current_type applies unless it's travel
                        'extruder': extruder_for_line # Store the active extruder command
                    }
                    moves.append(move_dict)
                    last_x, last_y, last_z, last_e = x, y, z if z is not None else last_z, e if e is not None else last_e
        return moves

    def showEvent(self, event):
        # Always minimize to 25% the first time the dialog is shown and set state
        if not hasattr(self, '_has_been_shown') or not self._has_been_shown:
            screen = QApplication.primaryScreen().availableGeometry()
            w_min, h_min = int(screen.width()*0.25), int(screen.height()*0.25)
            self.resize(w_min, h_min)
            self.move((screen.width()-w_min)//2, (screen.height()-h_min)//2)
            self._is_custom_maximized = False
            self._has_been_shown = True
        super().showEvent(event)

    def toggle_minmax(self):
        screen = QApplication.primaryScreen().availableGeometry()
        w_min, h_min = int(screen.width()*0.25), int(screen.height()*0.25)
        w_max, h_max = int(screen.width()*0.95), int(screen.height()*0.95)
        cur_w, cur_h = self.width(), self.height()
        # Decide based on current size, not just state variable
        if abs(cur_w - w_min) < abs(cur_w - w_max):
            # Currently minimized, so maximize
            self.resize(w_max, h_max)
            self.move((screen.width()-w_max)//2, (screen.height()-h_max)//2)
            self._is_custom_maximized = True
        else:
            # Currently maximized or other, so minimize
            self.resize(w_min, h_min)
            self.move((screen.width()-w_min)//2, (screen.height()-h_min)//2)
            self._is_custom_maximized = False

    def closeEvent(self, event):
        # Hide instead of destroy, to persist OpenGL context
        self.hide()
        event.ignore()

    def save_edits(self):
        # Save all current edit insertions for this layer back to the main window
        if not hasattr(self, 'edit_inserts'):
            self.edit_inserts = []
        if self.mainwin and hasattr(self.mainwin, 'save_layer_edits'):
            self.mainwin.save_layer_edits(self.layer_idx, list(self.edit_inserts))
            QMessageBox.information(self, "Edits Saved", f"Edits for layer {self.layer_idx} saved.")
        else:
            QMessageBox.warning(self, "Warning", "Could not save edits: main window not found.")

    # Map a move index (from self.moves) to the corresponding G-code line index in self.layer_lines.
    def move_index_to_gcode_line(self, move_idx):
        move_count = 0
        # Iterate over self.layer_lines_with_extruders, but use only the line text for logic
        for i, line_item in enumerate(self.layer_lines_with_extruders):
            line_text, _ = line_item # Unpack, we only need the text part here
            line = line_text.strip()
            if not line or line.startswith('M'): # T-codes are M-like in structure for this check
                if line.startswith('T') and line[1:].isdigit(): # Explicitly skip T-codes as non-moves
                    pass # Or continue, but the M check might catch it if it's not G
                else:
                    continue

            if line.startswith('G0') or line.startswith('G1') or line.startswith('G01') or line.startswith('G2') or line.startswith('G3'):
                if move_count == move_idx:
                    return i # Return the index in the layer_lines_with_extruders list
                move_count += 1
        return len(self.layer_lines_with_extruders) # If not found, return end of layer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GCodeEditor()
    window.show()
    sys.exit(app.exec_())
