from gui.debugger.ui.watched import WatchedPanel
from ...qt.QtWidgets import *
from ...qt.QtCore import Qt, QThread
from ...qt import QtSignal
from ...qt.QtGui import QColor
from ...utils.config import CONFIG
import socket
import json
import time
from datetime import datetime

from .controls import DebugControls
from .variables import VariablesPanel
from .callstack import CallStackPanel
from .watched import WatchedPanel

# PROTOCOL 
#
# ui → debugger
# {
#   type: command
#   name: <contiue | step_line | step_into | step_out | quit | update_watchlist>
#   payload: {} 
# }
#
# debugger → ui
# {
#   type: state_update
#   name: state_update | quit
#   payload: {
#     'line': ...
#     'locals': ...
#     'call_stack': ...
#     'watch_list': ...
#   }
# }
class PersistentSocketThread(QThread):
    connected_ok = QtSignal()
    error = QtSignal(str)
    closed = QtSignal()

    line_received = QtSignal(int)
    vars_received = QtSignal(dict)
    stack_received = QtSignal(list)
    watch_list_received = QtSignal(dict)

    send_command_signal = QtSignal(bytes)

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.socket = None
        self.running = True
        self.connected = False

        self.send_queue = []
        self.send_command_signal.connect(self.enqueue_command)

    def enqueue_command(self, cmd: bytes):
        self.send_queue.append(cmd)

    def connect_socket(self, retries=5, delay=0.5):
        attempt = 0
        while attempt < retries and self.running:
            try:
                self.socket = socket.create_connection((self.host, self.port), timeout=3)
                self.socket.settimeout(0.25)
                self.connected = True
                self.connected_ok.emit()
                return
            except Exception as e:
                attempt += 1
                if attempt < retries:
                    time.sleep(delay)
                else:
                    self.error.emit(f"Connection failed after {retries} attempts: {e}")
                    self.connected = False
                    return

    def run(self):
        self.connect_socket(retries=CONFIG['debugger/connection_retires'],delay=CONFIG['debugger/connection_retry_delay'])
        if not self.connected:
            return

        while self.running:
            if self.send_queue:
                cmd = self.send_queue.pop(0)
                try:
                    self.socket.sendall(cmd)
                except Exception as e:
                    self.error.emit(f"Send error: {e}")
                    break

            try:
                data = self.recv_json(self.socket)
                if data:
                    try:
                        data = json.loads(data)
                        if data.get("name") == "state_update" and data.get("type") == "state_update":
                            payload = data.get("payload", {})
                            self.line_received.emit(payload.get("line", -1))
                            self.vars_received.emit(payload.get("locals", {}))
                            self.stack_received.emit(payload.get("call_stack", {}))
                            self.watch_list_received.emit(payload.get("watch_list", {}))
                        elif data.get("name") == "quit" and data.get("type") == "state_update":
                            self.running = False
                            break
                    except Exception as e:
                        self.error.emit(f"JSON decode error: {e}")

            except socket.timeout:
                pass
            except Exception as e:
                self.error.emit(f"Socket error: {e}")
                break

            time.sleep(0.01)

        try:
            if self.socket:
                self.socket.close()
        except:
            pass

        self.closed.emit()

    def recv_json(self, sock):
        if not hasattr(self, "_buffer"):
            self._buffer = ""

        try:
            chunk = sock.recv(4096).decode("utf-8")
            if not chunk:
                return None
            self._buffer += chunk

            if "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                return line

        except socket.timeout:
            raise

        return None

    def stop(self):
        self.running = False
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except:
                pass

class DebuggerPanel(QDockWidget):
    current_line_signal = QtSignal(int)

    ask_breakpoints = QtSignal()
    received_breakpoints = QtSignal(set)

    def __init__(self, window_parent):
        super().__init__("Debugger", window_parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)

        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("Host:"))
        self.host_label = QLabel("127.0.0.1")
        config_layout.addWidget(self.host_label)

        config_layout.addWidget(QLabel("Port:"))
        self.port_label = QLabel(str(CONFIG['launcher_debug/port']))
        config_layout.addWidget(self.port_label)

        self.controls_widget = DebugControls()
        self.variables_widget = VariablesPanel()
        self.call_stack_widget = CallStackPanel()
        self.watched_widget = WatchedPanel()

        self.reconnect_button = QPushButton("Reconnect")
        self.reconnect_button.setToolTip("Reconnect to debugger backend.")
        self.reconnect_button.setEnabled(False) 
        self.reconnect_button.setVisible(False) 
        self.reconnect_button.clicked.connect(self.connect_debugger)

        vars_section = self.CollapsibleSection("Variables", self.variables_widget)
        stack_section = self.CollapsibleSection("Call Stack", self.call_stack_widget)
        watch_section = self.CollapsibleSection("Watch Expressions", self.watched_widget)

        self.sections = [vars_section, stack_section, watch_section]

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)

        self.splitter.addWidget(vars_section)
        self.splitter.addWidget(stack_section)
        self.splitter.addWidget(watch_section)

        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #ccc;
            }
            QSplitter::handle:hover {
                background-color: #aaa;
            }
        """)

        layout.addLayout(config_layout)
        layout.addWidget(self.controls_widget)
        layout.addWidget(self.reconnect_button)
        layout.addWidget(self.splitter)
        layout.setContentsMargins(4, 4, 4, 4)

        self.setWidget(container)
        self.setVisible(False)

        if window_parent and hasattr(window_parent, "addDockWidget"):
            window_parent.addDockWidget(Qt.RightDockWidgetArea, self)

        self._connect_signals()

        # Thread ref
        self.socket_thread = None
        self.breakpoints = set()

    def _connect_signals(self):
        self.controls_widget.continue_clicked.connect(self.send_continue)
        self.controls_widget.step_line_clicked.connect(self.send_step_line)
        self.controls_widget.step_into_clicked.connect(self.send_step_into)
        self.controls_widget.step_out_clicked.connect(self.send_step_out)
        self.controls_widget.stop_clicked.connect(self.stop_debugger)
        self.watched_widget.update_watched_expressions.connect(self.send_watched_expressions)

    class CollapsibleSection(QWidget):
        toggled = QtSignal()

        def __init__(self, title: str, content: QWidget):
            super().__init__()

            self._saved_size = None

            self.toggle_btn = QPushButton(f"▼ {title}")
            self.toggle_btn.setCheckable(True)
            self.toggle_btn.setChecked(True)
            self.toggle_btn.setFlat(True)

            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 4px 6px;
                    border: none;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """)

            self.content = content

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self.toggle_btn)
            layout.addWidget(self.content)

            self.toggle_btn.toggled.connect(self.on_toggle)

        def on_toggle(self, checked):
            title = self.toggle_btn.text()[2:]
            self.toggle_btn.setText(("▼ " if checked else "▶ ") + title)

            splitter = self.parentWidget()

            if not isinstance(splitter, QSplitter):
                self.content.setVisible(checked)
                return

            widgets = [splitter.widget(i) for i in range(splitter.count())]
            sizes = splitter.sizes()

            size_map = dict(zip(widgets, sizes))

            collapsed_height = self.toggle_btn.sizeHint().height()

            expanded = []
            collapsed = []

            for w in widgets:
                if isinstance(w, DebuggerPanel.CollapsibleSection):
                    if w.toggle_btn.isChecked():
                        expanded.append(w)
                    else:
                        collapsed.append(w)

            total_size = sum(sizes)

            collapsed_total = len(collapsed) * collapsed_height
            remaining = max(0, total_size - collapsed_total)

            expanded_sizes = [size_map[w] for w in expanded]
            expanded_total = sum(expanded_sizes)

            new_sizes = []

            for w in widgets:
                if w in collapsed:
                    new_sizes.append(collapsed_height)
                else:
                    if expanded_total > 0:
                        proportion = size_map[w] / expanded_total
                        new_sizes.append(int(proportion * remaining))
                    else:
                        new_sizes.append(remaining // max(1, len(expanded)))

            MIN_EXPANDED = 80

            for i, w in enumerate(widgets):
                if w in expanded:
                    new_sizes[i] = max(new_sizes[i], MIN_EXPANDED)

            splitter.setSizes(new_sizes)

            self.content.setVisible(checked)
            self.toggled.emit()

    def add_panel_message(self, panel, text, msg_type="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        item = QTreeWidgetItem([timestamp, text])
        
        color_map = {
            "error": QColor("red"),
            "warn": QColor("orange"),
            "info": QColor("blue"),
            "debug": QColor("gray")
        }
        color = color_map.get(msg_type, QColor("black"))
        item.setForeground(1, color)
        
        panel.addTopLevelItem(item)
        panel.scrollToItem(item)

    def get_breakpoint_lines(self):
        self.request_breakpoints()
        return self.breakpoints

    def request_breakpoints(self):
        self.ask_breakpoints.emit()

    def recieve_breakpoints(self, breakpoints):
        self.breakpoints = breakpoints

    def toggle_visibility(self):
        self.setVisible(not self.isVisible())

    def connect_debugger(self):
        host = "127.0.0.1"
        port = CONFIG['launcher_debug/port']

        self.add_panel_message(self.variables_widget.variables_panel, "Connecting...", "info")

        self.socket_thread = PersistentSocketThread(host, port)
        self.socket_thread.connected_ok.connect(self.on_connected)
        self.socket_thread.vars_received.connect(self.variables_widget.update_vars)
        self.socket_thread.line_received.connect(self.update_current_showed_line)
        self.socket_thread.error.connect(self.show_error)
        self.socket_thread.closed.connect(self.on_closed)
        self.socket_thread.stack_received.connect(self.call_stack_widget.update_call_stack)
        self.socket_thread.watch_list_received.connect(self.watched_widget.update_watch_list)
        self.socket_thread.start()

    def send_continue(self):
        cmd = {
                'type': 'command',
                'name': 'continue',
                'payload': {}
            }
        self.send_cmd((json.dumps(cmd)+"\n").encode('utf-8'))
    
    def send_step_line(self):
        cmd = {
                'type': 'command',
                'name': 'step_line',
                'payload': {}
            }
        self.send_cmd((json.dumps(cmd)+"\n").encode('utf-8'))

    def send_step_into(self):
        cmd = {
                'type': 'command',
                'name': 'step_into',
                'payload': {}
            }
        self.send_cmd((json.dumps(cmd)+"\n").encode('utf-8'))

    def send_step_out(self):
        cmd = {
                'type': 'command',
                'name': 'step_out',
                'payload': {}
            }
        self.send_cmd((json.dumps(cmd)+"\n").encode('utf-8'))

    def send_watched_expressions(self, expressions):
        cmd = {
                'type': 'command',
                'name': 'update_watchlist',
                'payload': expressions
            }
        self.send_cmd((json.dumps(cmd)+"\n").encode('utf-8'))

    def update_current_showed_line(self, line):
        self.current_line_signal.emit(line)

    def send_cmd(self, cmd: bytes):
        if self.socket_thread and self.socket_thread.connected:
            self.socket_thread.send_command_signal.emit(cmd)
        else:
            self.add_panel_message(self.variables_widget.variables_panel, "Not connected", "warn")

    def stop_debugger(self):
        if self.socket_thread:
            self.send_cmd(b"STOP\n")
            self.socket_thread.stop()

    def on_connected(self):
        self.variables_widget.variables_panel.clear()
        self.add_panel_message(self.variables_widget.variables_panel, "Connected to debugger.", "info")
        self.reconnect_button.setVisible(False)
        self.reconnect_button.setEnabled(False)

        self.controls_widget.setEnabled(True)
        self.variables_widget.setEnabled(True)
        self.call_stack_widget.setEnabled(True)
        self.watched_widget.setEnabled(True)

        for btn in [
            self.controls_widget.continue_button,
            self.controls_widget.step_line_button,
            self.controls_widget.step_into_button,
            self.controls_widget.step_out_button,
            self.controls_widget.stop_button
        ]:
            btn.setEnabled(True)

        self.send_continue()

    def show_error(self, msg):
        self.add_panel_message(self.variables_widget.variables_panel, msg, "error")

        if self.socket_thread and not self.socket_thread.connected:
                self.reconnect_button.setVisible(True)
                self.reconnect_button.setEnabled(True)

    def on_closed(self):
        self.variables_widget.variables_panel.clear()
        self.add_panel_message(self.variables_widget.variables_panel, "Debugger connection closed.", "info")

        self.reconnect_button.setVisible(True)
        self.reconnect_button.setEnabled(True)

        for btn in [
            self.controls_widget.continue_button,
            self.controls_widget.step_line_button,
            self.controls_widget.step_into_button,
            self.controls_widget.step_out_button,
            self.controls_widget.stop_button
        ]:
            btn.setEnabled(False)

        self.controls_widget.setEnabled(False)
        self.variables_widget.setEnabled(False)
        self.call_stack_widget.setEnabled(False)
        self.watched_widget.setEnabled(False)

        self.current_line_signal.emit(-1)
        if self.socket_thread:
            self.socket_thread.stop()
            self.socket_thread.wait()
            self.socket_thread.deleteLater()
            self.socket_thread = None

