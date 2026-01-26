# launch_demo.py - Wrapper with license agreement
"""
PATTERN HUNTER - DEMO VERSION LAUNCHER
=======================================
This launcher ensures users agree to portfolio license terms
before running the demonstration GUI.
"""

import sys
import os
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

def show_license_agreement():
    """Show license agreement before launching demo"""
    app = QApplication(sys.argv)
    
    license_text = """
    PATTERN HUNTER - TECHNICAL PORTFOLIO DEMO
    ==========================================
    
    LICENSE AGREEMENT
    
    This software demonstration is provided as part of a technical
    portfolio for employment evaluation purposes ONLY.
    
    PERMITTED USES:
    ✅ Review of GUI design and architecture
    ✅ Compilation and execution for skill assessment
    ✅ Technical discussion in interviews
    
    PROHIBITED USES:
    ❌ Commercial use of any kind
    ❌ Modification or redistribution
    ❌ Reverse engineering of algorithms
    ❌ Production deployment
    
    PROPRIETARY INFORMATION:
    The following are trade secrets and NOT included in this demo:
    • CUDA kernel optimizations (memory coalescing, warp operations)
    • Advanced GPU memory management patterns
    • Performance tuning parameters
    • Multi-GPU scaling algorithms
    • Custom search algorithms
    
    FULL VERSION AVAILABLE:
    • Under NDA for hiring processes
    • Commercial licensing for production use
    • Custom development services
    
    By clicking "AGREE", you acknowledge that this is demonstration
    code for portfolio purposes only, and agree to the restrictions above.
    """
    
    dialog = QDialog()
    dialog.setWindowTitle("Portfolio Demo License Agreement")
    dialog.setFixedSize(600, 500)
    
    layout = QVBoxLayout()
    
    # Title
    title = QLabel("⚠️ PORTFOLIO DEMO - LICENSE REQUIRED")
    title.setStyleSheet("font-weight: bold; color: #FF9800; font-size: 16px;")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(title)
    
    # License text
    text_edit = QTextEdit()
    text_edit.setPlainText(license_text)
    text_edit.setReadOnly(True)
    text_edit.setFontFamily("Monospace")
    text_edit.setFontPointSize(10)
    layout.addWidget(text_edit)
    
    # Buttons
    button_box = QDialogButtonBox()
    agree_btn = button_box.addButton("✅ I AGREE - LAUNCH DEMO", QDialogButtonBox.ButtonRole.AcceptRole)
    decline_btn = button_box.addButton("❌ DECLINE - EXIT", QDialogButtonBox.ButtonRole.RejectRole)
    
    agree_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
    decline_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
    
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)
    
    layout.addWidget(button_box)
    dialog.setLayout(layout)
    
    result = dialog.exec()
    
    if result == QDialog.DialogCode.Accepted:
        return True
    else:
        return False

if __name__ == "__main__":
    if show_license_agreement():
        # Import and run the actual demo
        from gui_demo import main
        main()
    else:
        print("License declined. Exiting.")
        sys.exit(0)
