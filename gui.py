import sys
from PySide6.QtWidgets import QApplication, QWidget, QLabel

# 1. Every app needs one (and only one) QApplication instance
# sys.argv allows you to pass in command-line arguments
app = QApplication(sys.argv)

# 2. Create a basic window (a QWidget)
window = QWidget()
window.setWindowTitle("SnapSearch")
window.setGeometry(100, 100, 300, 200) # x, y, width, height

# 3. Add a label widget to the window
label = QLabel("Hello, PySide6!", parent=window)
label.move(100, 80) # Position the label inside the window

# 4. Show the window
window.show()

# 5. Start the application's main event loop
# This waits for user interaction (clicks, keypresses, etc.)
sys.exit(app.exec())