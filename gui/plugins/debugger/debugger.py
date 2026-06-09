import json
import sys
import threading
import socket
import os

import plask

from plask_debugger.adapter import DebuggerAdapter

running = True

def run_server(adapter, host, port):
    conn = None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        s.bind((host, port) if port is not None else (host, 0))
        s.listen()

        port = s.getsockname()[1]
        plask.print_log('debug', f"[DEBUGGER]: Started socket on: {host}:{port}")

        try:
            conn, addr = s.accept()
            plask.print_log('debug', f"[DEBUGGER]: Connected by {addr}")

            def emit_state(state_json):
                try:
                    conn.sendall(state_json.encode("utf-8") + b"\n")
                except OSError:
                    pass

            adapter.emit_state = emit_state

            buffer = b""

            while True:
                if not running:
                    plask.print_log('info', "[DEBUGGER]: Program finished, exiting")
                    break

                try:
                    data = conn.recv(4096)
                except ConnectionResetError:
                    plask.print_log('critical_error', "[DEBUGGER]: Connection reset by client")
                    break
                except OSError as e:
                    plask.print_log('critical_error', f"[DEBUGGER]: Socket error: {e}")
                    break

                if not data:
                    plask.print_log('info', "[DEBUGGER]: Client disconnected.")
                    break

                buffer += data

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    try:
                        cmd = json.loads(line.decode("utf-8"))
                    except Exception as e:
                        plask.print_log('warning', f"[DEBUGGER]: Invalid command: {e}")
                        continue

                    adapter.handle_command(cmd)

        finally:
            if conn:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                conn.close()


if __name__ == "__main__":
    PORT = None
    breakpoints = ""

    sys._original_argv = sys.argv.copy()

    defines = {}

    while len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        if sys.argv[1] == "--port":
            if len(sys.argv) < 3:
                print("Error: --port must be followed by a valid number", file=sys.stderr, flush=True)
                sys.exit(1)
            try:
                PORT = int(sys.argv[2])
            except ValueError:
                print("Error: --port must be followed by a valid number", file=sys.stderr, flush=True)
                sys.exit(1)
            sys.argv.pop(1)
            sys.argv.pop(1)
        elif sys.argv[1] == "--breakpoints":
            if len(sys.argv) < 3:
                print("Error: --breakpoints must be followed by a comma-separated list of breakpoints", file=sys.stderr, flush=True)
                sys.exit(1)
            breakpoints = sys.argv[2]
            sys.argv.pop(1)
            sys.argv.pop(1)
        elif sys.argv[1].startswith("-D"):
            define = sys.argv[1][2:]
            if "=" in define:
                key, value = define.split("=", 1)
                try:
                    defines[key] = eval(value)
                except ValueError:
                    defines[key] = value
            else:
                defines[define] = True
            sys.argv.pop(1)
        elif sys.argv[1] == "--":
            sys.argv.pop(1)
            break
        else:
            print(f"Unknown option: {sys.argv[1]}", file=sys.stderr, flush=True)
            sys.exit(1)

    if len(sys.argv) < 2:
        print(
            "Usage: python debugger.py [--breakpoints <bp1,bp2,...>] [--port <port>] [-Ddef=val...] [--] <file.xpl> [args...]",
            file=sys.stderr,
            flush=True
        )
        sys.exit(1)

    script_path = sys.argv[1]
    sys.argv.pop(0) # Remove the debugger script from arguments, leaving only the target script and its args

    sys.path[0] = os.path.dirname(script_path) # Ensure the script's directory is in the path for imports, not the debugger's one

    manager = plask.Manager()
    if script_path.endswith('.py'):
        code_str = open(script_path, "r", encoding="UTF-8").read()
        first_line = 0
    else:
        manager.load(script_path)
        first_line = manager._scriptline
        code_str = "\n" * (first_line - 1) + manager.script

    env = manager._globals
    manager.export(env)
    env.update(defines)

    adapter = DebuggerAdapter(line_offset=first_line)

    # Parse breakpoints
    for bp in breakpoints.split(","):
        if bp.strip():
            try:
                bp_file, bp_line = bp.split(":")
                adapter.debugger.set_break(bp_file.strip(), int(bp_line))
            except ValueError:
                plask.print_log('warning', f"Invalid breakpoint format: {bp}")

    code = compile(code_str, script_path, "exec")
    plask.print_log('debug', "[DEBUGGER]: Loading and compilation finished")

    HOST = "127.0.0.1"

    # Start server in another thread and run the debugger
    running = True
    server_thread = threading.Thread(target=run_server, args=(adapter, HOST, PORT), daemon=True)
    server_thread.start()

    try:
        adapter.run(code, env=env)
    finally:
        adapter.quit()
        running = False
        server_thread.join(timeout=2)
        if server_thread.is_alive():
            plask.print_log('warning', "[DEBUGGER]: Warning: server thread did not exit cleanly")
            server_thread.kill()

    plask.print_log('info', "[DEBUGGER]: Successfully exited")
