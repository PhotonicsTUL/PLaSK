import json
import sys
import threading
import socket 
import os

import plask
from adapter import DebuggerAdapter

def run_server(adapter, code, HOST, PORT):
    dbg_thread = None
    conn = None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        s.bind((HOST, PORT) if PORT is not None else (HOST, 0))
        s.listen()

        PORT = s.getsockname()[1]
        print(f"[DEBUGGER]: Started socket on: {HOST}:{PORT}", flush=True)

        try:
            conn, addr = s.accept()
            print(f"[DEBUGGER]: Connected by {addr}", flush=True)

            def emit_state(state_json):
                try:
                    conn.sendall(state_json.encode("utf-8") + b"\n")
                except OSError:
                    pass

            adapter.emit_state = emit_state

            def run_dbg():
                adapter.run(code)

            dbg_thread = threading.Thread(target=run_dbg, daemon=True)
            dbg_thread.start()

            buffer = b""

            while True:
                if dbg_thread and not dbg_thread.is_alive():
                    print("[DEBUGGER]: Program finished, exiting.", flush=True)
                    break

                try:
                    data = conn.recv(4096)
                except ConnectionResetError:
                    print("[DEBUGGER]: Connection reset by client.", flush=True)
                    break
                except OSError as e:
                    print(f"[DEBUGGER]: Socket error: {e}", flush=True)
                    break

                if not data:
                    print("[DEBUGGER]: Client disconnected.", flush=True)
                    break

                buffer += data

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    try:
                        cmd = json.loads(line.decode("utf-8"))
                    except Exception as e:
                        print(f"[DEBUGGER]: Invalid command: {e}", flush=True)
                        continue

                    adapter.handle_command(cmd)

        finally:
            if dbg_thread and dbg_thread.is_alive():
                adapter.quit()
                dbg_thread.join(timeout=2)

            if conn:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                conn.close()

    print("[DEBUGGER]: Successfully exited")

if __name__ == "__main__":
    PORT = None
    WORK_DIR = None

    # Parse command-line flags
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        try:
            PORT = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Error: --port must be followed by a valid number", flush=True)
            sys.exit(1)
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    if "--work_dir" in sys.argv:
        idx = sys.argv.index("--work_dir")
        try:
            WORK_DIR = str(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Error: --work_dir must be followed by a string path", flush=True)
            sys.exit(1)
        sys.argv.pop(idx)
        sys.argv.pop(idx)

    if len(sys.argv) < 2:
        print("Usage: python debugger.py <file.xpl> [breakpoints] [--port <port>]", flush=True)
        sys.exit(1)

    script_path = sys.argv[1]
    breakpoints = sys.argv[2] if len(sys.argv) >= 3 else ""

    manager = plask.Manager()
    manager.load(script_path)
    first_line = manager._scriptline

    adapter = DebuggerAdapter(line_offset=first_line)

    # Parse breakpoints
    for bp in breakpoints.split(","):
        if bp.strip():
            try:
                bp_file, bp_line = bp.split(":")
                adapter.debugger.set_break(bp_file.strip(), int(bp_line))
            except ValueError:
                print(f"Invalid breakpoint format: {bp}", flush=True)

    if WORK_DIR is not None:
        os.chdir(WORK_DIR)

    code_str = "import plask\n" + ("\n" * (first_line - 2)) + manager.script
    code = compile(code_str, script_path, "exec")
    print("[DEBUGGER]: Loading and compilation finished", flush=True)

    HOST = "127.0.0.1"

    run_server(adapter, code, HOST, PORT)
