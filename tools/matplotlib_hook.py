"""
Matplotlib hook to automatically save figures when plt.show() is called.
This allows scripts to use plt.show() without modification, and the GUI can display the saved images.
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Determine output directory first (before importing matplotlib)
if getattr(sys, 'frozen', False):
    # When running as PyInstaller bundle
    exe_path = Path(sys.executable).resolve()
    OUTPUT_DIR = exe_path.parent.parent.parent / "outputs"
else:
    # When running as script
    script_dir = Path(__file__).resolve().parent
    OUTPUT_DIR = script_dir.parent / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Force non-interactive backend BEFORE any matplotlib imports
# This must happen before any script imports matplotlib
import matplotlib
matplotlib.use('Agg')

# Now import pyplot
import matplotlib.pyplot as plt

_figure_count = 0

def _save_and_show(*args, **kwargs):
    """Replacement for plt.show() that saves the figure before showing."""
    global _figure_count
    
    try:
        # Get current figure
        fig = plt.gcf()
        
        # Only save if figure has content
        if fig and fig.get_axes():
            _figure_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plot_{timestamp}_{_figure_count:03d}.png"
            filepath = OUTPUT_DIR / filename
            
            try:
                # Save figure
                fig.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
                print(f"[Matplotlib Hook] 图片已保存: {filepath}")
                sys.stdout.flush()  # Ensure output is immediately visible
            except Exception as e:
                print(f"[Matplotlib Hook] 保存图片失败: {e}")
                sys.stdout.flush()
            
            # Close figure to free memory (Agg backend doesn't show anyway)
            plt.close(fig)
        elif fig:
            # Figure exists but has no axes, just close it
            plt.close(fig)
    except Exception as e:
        # If anything goes wrong, try to close any open figures
        try:
            plt.close('all')
        except Exception:
            pass
        print(f"[Matplotlib Hook] 处理图片时出错: {e}")
        sys.stdout.flush()

# Replace plt.show with our hook
plt.show = _save_and_show

# Patch the matplotlib.pyplot module in sys.modules
# This ensures that any script importing pyplot (even after this hook loads)
# will get our patched show function due to Python's module caching
try:
    # Directly patch the module that's now in sys.modules
    pyplot_module = sys.modules.get('matplotlib.pyplot')
    if pyplot_module:
        pyplot_module.show = _save_and_show
        # Also ensure the module's __dict__ is patched for robustness
        if hasattr(pyplot_module, '__dict__'):
            pyplot_module.__dict__['show'] = _save_and_show
except Exception:
    pass

# Note: Since this hook is imported BEFORE the target script runs,
# and Python caches imported modules in sys.modules, when the target script
# does "import matplotlib.pyplot as plt", it will get the already-imported
# and already-patched module, so plt.show() will call our _save_and_show function.

