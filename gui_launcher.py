import os
import sys
import threading
import subprocess
import time
import glob
import queue
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
try:
    import pystray
    from pystray import MenuItem as Item
except Exception:
    pystray = None

# Determine project root robustly. When running as a PyInstaller bundle (frozen),
# __file__ points inside the bundle; we instead derive the source root from the
# location of the executable (which is usually at <project>\tools\dist\...).
if getattr(sys, 'frozen', False):
    try:
        exe_path = Path(sys.executable).resolve()
        # exe_path example: D:/SRTP/SRTP/tools/dist/SRTP-GUI.exe
        # navigate up to project root: exe.parent (dist) -> parent.parent (tools) -> parent.parent.parent (project)
        ROOT = exe_path.parent.parent.parent
    except Exception:
        # If __file__ is in project root (e.g., D:/SRTP/SRTP/gui_launcher.py),
        # use parent (D:/SRTP/SRTP). If in tools/ (e.g., D:/SRTP/SRTP/tools/gui_launcher.py),
        # use parents[1] (D:/SRTP/SRTP).
        file_path = Path(__file__).resolve()
        if file_path.parent.name == 'tools':
            ROOT = file_path.parents[1]  # tools/ -> project root
        else:
            ROOT = file_path.parent  # project root -> project root
else:
    # When running as script: gui_launcher.py is in project root
    # If __file__ is D:/SRTP/SRTP/gui_launcher.py, parent is D:/SRTP/SRTP
    # If __file__ is D:/SRTP/SRTP/tools/gui_launcher.py, parents[1] is D:/SRTP/SRTP
    file_path = Path(__file__).resolve()
    if file_path.parent.name == 'tools':
        ROOT = file_path.parents[1]  # tools/ -> project root
    else:
        ROOT = file_path.parent  # project root -> project root

# Mapping: task -> model -> method -> script relative path
SCRIPT_MAP = {
    "节点分类": {
        "GCN": {"GraphPrompt": "node_classify/GCN_node_GraphPrompt_dynamic.py",
                "ProG": "node_classify/GCN_node_ProG_dynamic.py"},
        "GAT": {"GraphPrompt": "node_classify/GAT_node_GraphPrompt_dynamic.py",
                "ProG": "node_classify/GAT_node_ProG_dynamic.py"},
        "GAE": {"GraphPrompt": "node_classify/GAE_node_GraphPrompt_dynamic.py",
                "ProG": "node_classify/GAE_node_ProG_dynamic.py"},
    },
    "图分类": {
        "GCN": {"GraphPrompt": "graph_classify/GCN_graph_GraphPrompt_attack.py",
                "ProG": "graph_classify/GCN_graph_ProG_attack.py"},
        "GAT": {"GraphPrompt": "graph_classify/GAT_graph_GraphPrompt_attack.py",
                "ProG": "graph_classify/GAT_graph_ProG_attack.py"},
        "GAE": {"GraphPrompt": "graph_classify/GAE_graph_GraphPrompt_attack.py",
                "ProG": "graph_classify/GAE_graph_ProG_attack.py"},
    },
    "跨类别": {
        "GCN": {"GraphPrompt": "cross_category/test3_GCN_GraphPrompt.py",
                "ProG": "cross_category/test4_GCN_ProG.py"},
        "GAT": {"GraphPrompt": "cross_category/test1_GAT_GraphPrompt.py",
                "ProG": "cross_category/test2_GAT_ProG.py"},
        "GAE": {"GraphPrompt": "cross_category/test5_GAE_GraphPrompt.py",
                "ProG": "cross_category/test6_GAE_ProG.py"},
    },
    "跨数据集": {
        "GCN": {"GraphPrompt": "cross_database/test3_GCN_GraphPrompt.py",
                "ProG": "cross_database/test4_GCN_ProG.py"},
        "GAT": {"GraphPrompt": "cross_database/test1_GAT_GraphPrompt.py",
                "ProG": "cross_database/test2_GAT_ProG.py"},
        "GAE": {"GraphPrompt": "cross_database/test5_GAE_GraphPrompt.py",
                "ProG": "cross_database/test6_GAE_ProG.py"},
    },
    "跨分布": {
        "GCN": {"GraphPrompt": "cross_distribution/test2_GCN_GraphPrompt.py",
                "ProG": "cross_distribution/test5_GCN_ProG.py"},
        "GAT": {"GraphPrompt": "cross_distribution/test1_GAT_GraphPrompt.py",
                "ProG": "cross_distribution/test4_GAT_ProG.py"},
        "GAE": {"GraphPrompt": "cross_distribution/test3_GAE_GraphPrompt.py",
                "ProG": "cross_distribution/test6_GAE_ProG.py"},
    },
    "带权图攻击": {
        "GCN": {"weighted": "^pnode_classify/weighted_GCN.py"},
        "GAT": {"weighted": "^pnode_classify/weighted_GAT.py"},
        "GAE": {"weighted": "^pnode_classify/weighted_GAE.py"},
    },
}

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("基于图提示学习的隐蔽式对抗注入攻击测试软件V1.0")
        self.geometry("1100x720")
        # ttk style
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        default_font = ('Segoe UI', 10)
        self.option_add("*Font", default_font)
        # Button accent style: cyan with black text
        try:
            style.configure('Accent.TButton', background='#00FFFF', foreground='black', borderwidth=1, focusthickness=3)
            style.map('Accent.TButton',
                      background=[('active', '#00e5e5'), ('pressed', '#00cccc')],
                      foreground=[('disabled', '#a0a0a0')])
        except Exception:
            # some themes may not accept background changes; ignore failures
            pass

        self.task_var = tk.StringVar(value=list(SCRIPT_MAP.keys())[0])
        self.model_var = tk.StringVar(value="GCN")
        self.method_var = tk.StringVar(value="GraphPrompt")

        # Set window icon from build_icon.ico (for Windows titlebar and taskbar)
        icon_ico_path = None
        icon_png_path = None
        tray_icon_path = None
        self._app_icon = None
        
        # First priority: build_icon.ico (Windows native .ico format)
        icon_ico_path = ROOT / 'build_icon.ico'
        if not icon_ico_path.exists():
            icon_ico_path = None
        
        # Fallback: icon.png in project root
        icon_png_path = ROOT / 'icon.png'
        if not icon_png_path.exists():
            icon_png_path = None
        # Set window icon (titlebar and taskbar on Windows)
        # Store icon path for later setting (after window is fully created)
        self._icon_path_for_windows = None
        if icon_ico_path and sys.platform.startswith('win'):
            self._icon_path_for_windows = str(icon_ico_path.resolve())
            # Schedule icon setting after window is fully initialized
            self.after(100, self._set_windows_icon)
        
        # Create PhotoImage for UI display and iconphoto (works on all platforms)
        icon_to_use = icon_ico_path if icon_ico_path else icon_png_path
        if icon_to_use:
            try:
                img = Image.open(str(icon_to_use))
                # Convert to RGBA for better compatibility
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                # Resize for UI display
                img.thumbnail((32, 32), Image.Resampling.LANCZOS)
                self._app_icon = ImageTk.PhotoImage(img)
                
                # Use iconphoto as additional method (works on Linux/Mac, and as fallback on Windows)
                try:
                    self.iconphoto(False, self._app_icon)
                except Exception as e:
                    print(f"[GUI] iconphoto设置失败: {e}")
            except Exception as e:
                print(f"[GUI] 加载图标文件失败: {e}")
        
        # Set system tray icon
        tray_icon_path = icon_ico_path if icon_ico_path else icon_png_path
        if tray_icon_path and pystray is not None:
            try:
                self._start_tray(tray_icon_path)
            except Exception as e:
                print(f"[GUI] 设置系统托盘图标失败: {e}")

        self._create_widgets()
        self.process = None
        self.img_label = None
        # queue for thread-safe communication from worker thread to main thread
        self._q = queue.Queue()
        # start polling queue
        self.after(100, self._process_queue)
    
    def _set_windows_icon(self):
        """Set Windows window icon after window is fully created."""
        if not hasattr(self, '_icon_path_for_windows') or not self._icon_path_for_windows:
            return
        
        if not sys.platform.startswith('win'):
            return
        
        icon_path_str = self._icon_path_for_windows
        # Ensure absolute path
        icon_path = Path(icon_path_str).resolve()
        icon_path_str = str(icon_path)
        # Convert to Windows native path format (backslashes)
        icon_path_str = icon_path_str.replace('/', '\\')
        
        try:
            # Update window first to ensure it's fully rendered
            self.update_idletasks()
            
            # Method 1: Standard iconbitmap() - primary method for Windows
            try:
                self.iconbitmap(icon_path_str)
                return  # Success
            except tk.TclError as e:
                print(f"[GUI] iconbitmap失败: {e}")
            except Exception as e:
                print(f"[GUI] iconbitmap异常: {e}")
            
            # Method 2: Try wm_iconbitmap (some tkinter versions)
            try:
                self.wm_iconbitmap(icon_path_str)
                return  # Success
            except Exception as e:
                print(f"[GUI] wm_iconbitmap失败: {e}")
            
            # Method 3: Try with default parameter
            try:
                self.iconbitmap(default=icon_path_str)
                return  # Success
            except Exception as e1:
                print(f"[GUI] iconbitmap所有标准方法都失败: {e1}")
            
            # Method 4: Try Windows API approach as last resort
            try:
                import ctypes
                # Load icon using Windows API
                LR_LOADFROMFILE = 0x0010
                IMAGE_ICON = 1
                NULL = 0
                
                hicon = ctypes.windll.user32.LoadImageW(
                    NULL,
                    icon_path_str,
                    IMAGE_ICON,
                    0, 0,
                    LR_LOADFROMFILE
                )
                
                if hicon:
                    # Get window handle
                    try:
                        window_id = self.winfo_id()
                        WM_SETICON = 0x0080
                        ICON_SMALL = 0
                        ICON_BIG = 1
                        ctypes.windll.user32.SendMessageW(window_id, WM_SETICON, ICON_SMALL, hicon)
                        ctypes.windll.user32.SendMessageW(window_id, WM_SETICON, ICON_BIG, hicon)
                        print(f"[GUI] ✓ 窗口图标已设置 (Windows API): {icon_path_str}")
                    except Exception as e3:
                        print(f"[GUI] Windows API SendMessage失败: {e3}")
                else:
                    print(f"[GUI] Windows API无法加载图标文件")
            except ImportError:
                print(f"[GUI] 无法导入ctypes")
            except Exception as e2:
                print(f"[GUI] Windows API设置图标失败: {e2}")
                
        except Exception as e:
            print(f"[GUI] 设置Windows图标时发生错误: {e}")
            import traceback
            traceback.print_exc()

    def _create_widgets(self):
        frame = ttk.Frame(self, padding=(4, 4))
        frame.pack(fill=tk.X, padx=4, pady=2)

        # If app icon was loaded in __init__, show it in the top menu bar (left side)
        icon_present = hasattr(self, '_app_icon') and getattr(self, '_app_icon') is not None
        if icon_present:
            try:
                icon_label = ttk.Label(frame, image=self._app_icon)
                icon_label.grid(row=0, column=0, padx=(4,8), sticky='w')
            except Exception:
                icon_present = False

        # Controls (menu starts after icon if present)
        label_col = 1 if icon_present else 0

        ttk.Label(frame, text="任务:").grid(row=0, column=label_col, sticky=tk.W)
        task_cb = ttk.Combobox(frame, textvariable=self.task_var, values=list(SCRIPT_MAP.keys()), state='readonly', width=14)
        task_cb.grid(row=0, column=label_col + 1, padx=4, sticky='w')
        task_cb.bind("<<ComboboxSelected>>", lambda e: self._on_task_change())

        ttk.Label(frame, text="模型:").grid(row=0, column=label_col + 2, sticky=tk.W)
        model_cb = ttk.Combobox(frame, textvariable=self.model_var, values=["GCN", "GAT", "GAE"], state='readonly', width=6)
        model_cb.grid(row=0, column=label_col + 3, padx=4, sticky='w')
        model_cb.bind("<<ComboboxSelected>>", lambda e: self._on_task_change())

        ttk.Label(frame, text="方法:").grid(row=0, column=label_col + 4, sticky=tk.W)
        self.method_cb = ttk.Combobox(frame, textvariable=self.method_var, values=["GraphPrompt", "ProG"], state='readonly', width=10)
        self.method_cb.grid(row=0, column=label_col + 5, padx=4, sticky='w')

        run_btn = ttk.Button(frame, text="运行并显示结果", command=self.run_selected, width=14, style='Accent.TButton', cursor='hand2')
        run_btn.grid(row=0, column=label_col + 6, padx=(8, 4))

        # Log area
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        # Left: logs (scrolled)
        left = ttk.Frame(bottom)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_text = ScrolledText(left, wrap=tk.NONE, height=28)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=(0, 6))

        # no bottom status bar (system tray used instead)
        self._on_task_change()

    # logo loading removed (no longer used)

    def _on_task_change(self):
        task = self.task_var.get()
        if task == "带权图攻击":
            # weighted only needs model; disable method
            self.method_cb.set("weighted")
            self.method_cb.config(values=["weighted"], state='disabled')
        else:
            self.method_cb.config(values=["GraphPrompt", "ProG"], state='readonly')
            # ensure current method valid
            if self.method_var.get() not in ["GraphPrompt", "ProG"]:
                self.method_var.set("GraphPrompt")

    def run_selected(self):
        task = self.task_var.get()
        model = self.model_var.get()
        method = self.method_var.get()

        # locate script
        script_rel = None
        try:
            script_rel = SCRIPT_MAP[task][model][method]
        except Exception:
            messagebox.showerror("错误", "未找到对应的脚本映射。请检查选择或映射配置。")
            return

        # Try several candidate roots to find the target script. This helps when
        # running the bundled exe (onefile) where paths can differ.
        candidate_roots = [ROOT, Path.cwd(), Path(sys.argv[0]).resolve().parent]
        script_path = None
        for r in candidate_roots:
            p = (r / script_rel).resolve()
            if p.exists():
                script_path = p
                break

        if script_path is None:
            messagebox.showerror("错误", f"脚本不存在: {ROOT / script_rel}\n候选路径: {candidate_roots}")
            return

        # snapshot PNGs before (check both outputs directory and project-wide)
        before = set(Path(p).resolve() for p in glob.glob(str(OUTPUT_DIR / "*.png"), recursive=False))
        before |= set(Path(p).resolve() for p in glob.glob(str(OUTPUT_DIR / "*.jpg"), recursive=False))
        before |= set(Path(p).resolve() for p in glob.glob(str(ROOT / "**" / "*.png"), recursive=True))
        before |= set(Path(p).resolve() for p in glob.glob(str(ROOT / "**" / "*.jpg"), recursive=True))

        # clear log
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, f"运行: {script_path}\n\n")
        self._update_status(f'运行: {script_path.name}')

        def target():
            # choose interpreter: when running as a frozen exe, prefer the project's
            # virtualenv python. Fall back to system python, then to current interpreter.
            python_exec = None
            try:
                # 1) Prefer ROOT/.venv on Windows
                venv_py = None
                if sys.platform.startswith('win'):
                    cand = ROOT / '.venv' / 'Scripts' / 'python.exe'
                    if cand.exists():
                        venv_py = cand
                else:
                    cand = ROOT / '.venv' / 'bin' / 'python'
                    if cand.exists():
                        venv_py = cand
                if venv_py and venv_py.exists():
                    python_exec = str(venv_py)
                # 2) Try VIRTUAL_ENV env var if provided
                if not python_exec:
                    ve = os.environ.get('VIRTUAL_ENV')
                    if ve:
                        if sys.platform.startswith('win'):
                            cand2 = Path(ve) / 'Scripts' / 'python.exe'
                        else:
                            cand2 = Path(ve) / 'bin' / 'python'
                        if cand2.exists():
                            python_exec = str(cand2)
                # 3) If running frozen and still not found, try system python
                if not python_exec and getattr(sys, 'frozen', False):
                    python_exec = shutil.which('python') or shutil.which('py')
                # 4) Last resort: current interpreter
                if not python_exec:
                    python_exec = sys.executable
            except Exception:
                python_exec = sys.executable

            # Inject matplotlib hook before running script to auto-save figures
            hook_path = ROOT / "tools" / "matplotlib_hook.py"
            wrapper_script = None
            if hook_path.exists():
                # Create a wrapper script that imports the hook first, then runs the target script
                import tempfile
                # Escape backslashes for Windows paths in Python string
                tools_path = str(ROOT / "tools").replace('\\', '\\\\')
                script_path_str = str(script_path).replace('\\', '\\\\')
                
                wrapper_code = f'''import sys
import os
import runpy
# Add tools directory to path
sys.path.insert(0, r"{tools_path}")

# Import matplotlib hook to intercept plt.show() and save figures
try:
    import matplotlib_hook
    print("[GUI] Matplotlib hook已加载，plt.show()将自动保存图片到outputs目录")
except Exception as e:
    print(f"[GUI] 加载matplotlib hook失败: {{e}}")

# Now run the actual script using runpy to preserve __name__ and execution context
script_path = r"{script_path_str}"
runpy.run_path(script_path, run_name="__main__")
'''
                # Write wrapper to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                    f.write(wrapper_code)
                    wrapper_script = f.name
                
                cmd = [python_exec, '-u', wrapper_script]
                self._q.put(("LINE", f"使用解释器: {python_exec}\n"))
                self._q.put(("LINE", f"[GUI] 已注入matplotlib hook，图片将自动保存\n"))
            else:
                # Fallback: run script directly if hook not found
                cmd = [python_exec, '-u', str(script_path)]
                self._q.put(("LINE", f"使用解释器: {python_exec}\n"))
                self._q.put(("LINE", f"[GUI] 警告: matplotlib hook未找到，将直接运行脚本\n"))

            # Ensure child process uses UTF-8 to avoid GBK decode issues on Windows consoles
            child_env = os.environ.copy()
            child_env.setdefault('PYTHONIOENCODING', 'utf-8')
            # Force non-interactive matplotlib backend
            child_env.setdefault('MPLBACKEND', 'Agg')

            # run subprocess with robust cross-version settings (avoid 'text'/encoding args)
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=child_env,
            )
            self.process = proc

            # stream output -> push to queue instead of writing GUI from thread
            try:
                assert proc.stdout is not None
                while True:
                    raw = proc.stdout.readline()
                    if not raw:
                        break
                    try:
                        line = raw.decode('utf-8', errors='replace')
                    except Exception:
                        try:
                            line = raw.decode('gbk', errors='replace')
                        except Exception:
                            line = raw.decode(errors='replace')
                    # put each line into queue for main thread to consume
                    self._q.put(("LINE", line))
            except Exception as e:
                self._q.put(("LINE", f"读取子进程输出时出错: {e}\n"))

            proc.wait()
            
            # Clean up wrapper script if it was created
            if wrapper_script is not None:
                try:
                    os.unlink(wrapper_script)
                except Exception:
                    pass
            
            self._q.put(("LINE", f"\n子进程退出: {proc.returncode}\n"))
            self._q.put(("STATUS", '就绪'))

            # Check for new images in outputs directory (where matplotlib_hook saves them)
            # Also check entire project for any new images
            time.sleep(0.5)  # Small delay for files to be written
            after = set(Path(p).resolve() for p in glob.glob(str(OUTPUT_DIR / "*.png"), recursive=False))
            after |= set(Path(p).resolve() for p in glob.glob(str(OUTPUT_DIR / "*.jpg"), recursive=False))
            # Also check project-wide for any new images
            after |= set(Path(p).resolve() for p in glob.glob(str(ROOT / "**" / "*.png"), recursive=True))
            after |= set(Path(p).resolve() for p in glob.glob(str(ROOT / "**" / "*.jpg"), recursive=True))
            new = list(after - before)
            if new:
                # pick newest by mtime
                new_sorted = sorted(new, key=lambda p: p.stat().st_mtime, reverse=True)
                # Show all new images, starting with the newest
                for img_path in new_sorted[:5]:  # Limit to 5 newest images
                    self._q.put(("LINE", f"\n找到新图片: {img_path}\n"))
                    # send image path to main thread to show
                    self._q.put(("IMAGE", str(img_path)))
                if len(new_sorted) > 5:
                    self._q.put(("LINE", f"\n(共找到{len(new_sorted)}张新图片，已显示最新的5张)\n"))
            else:
                self._q.put(("LINE", "\n未检测到新图片文件。matplotlib图片应该通过plt.show()自动保存到outputs目录。\n"))

        t = threading.Thread(target=target, daemon=True)
        t.start()

    def _process_queue(self):
        # poll queue and handle messages from worker thread
        try:
            while True:
                item = self._q.get_nowait()
                if not item:
                    continue
                kind, payload = item
                if kind == "LINE":
                    # insert text into Text widget (main thread)
                    self.log_text.insert(tk.END, payload)
                    self.log_text.see(tk.END)
                elif kind == "IMAGE":
                    # payload is a path string
                    try:
                        self._show_image(Path(payload))
                    except Exception as e:
                        self.log_text.insert(tk.END, f"加载图片失败: {e}\n")
                elif kind == "STATUS":
                    self._update_status(payload)
                self._q.task_done()
        except queue.Empty:
            pass
        # schedule next poll
        self.after(100, self._process_queue)

    def _show_image(self, path: Path):
        # Open image using system default viewer instead of internal canvas
        try:
            p = str(path)
            if sys.platform.startswith('win'):
                os.startfile(p)
            else:
                import subprocess as _sub
                if sys.platform == 'darwin':
                    _sub.Popen(['open', p])
                else:
                    _sub.Popen(['xdg-open', p])
            self.log_text.insert(tk.END, f"已用系统查看器打开图片: {path}\n")
            self.log_text.see(tk.END)
            self._update_status('已打开图片')
        except Exception as e:
            self.log_text.insert(tk.END, f"打开图片失败: {e}\n")

    def _update_status(self, text: str):
        # status_var may not exist if status bar removed; ignore in that case
        try:
            if hasattr(self, 'status_var'):
                self.status_var.set(text)
        except Exception:
            pass

    # ---- system tray support ----
    def _start_tray(self, icon_path: Path):
        # prepare PIL image for tray
        try:
            tray_img = Image.open(str(icon_path))
        except Exception:
            return

        def on_show(icon, item):
            try:
                # show and raise window
                self.after(0, lambda: (self.deiconify(), self.lift(), self.focus_force()))
            except Exception:
                pass

        def on_quit(icon, item):
            try:
                icon.stop()
            except Exception:
                pass
            try:
                # close GUI
                self.after(0, self.destroy)
            except Exception:
                pass

        menu = (Item('显示窗口', on_show), Item('退出', on_quit))

        def tray_thread():
            try:
                icon = pystray.Icon('srtp_gui', tray_img, 'SRTP 项目 GUI', menu)
                self._tray_icon = icon
                icon.run()
            except Exception:
                pass

        t = threading.Thread(target=tray_thread, daemon=True)
        t.start()


if __name__ == '__main__':
    # ensure Pillow present
    try:
        from PIL import Image, ImageTk
    except Exception:
        print("缺少依赖 Pillow。请先运行: pip install -r requirements.txt")
        sys.exit(1)

    app = App()
    app.mainloop()
