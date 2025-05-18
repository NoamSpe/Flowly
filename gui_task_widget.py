from PyQt5.QtWidgets import QWidget, QHBoxLayout, QCheckBox, QLabel, QPushButton, QSizePolicy, QSpacerItem
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPalette, QColor

class TaskItemWidget(QWidget):
    status_changed = pyqtSignal(int, bool) # task_id, is_done
    edit_requested = pyqtSignal(int)       # task_id
    delete_requested = pyqtSignal(int)     # task_id

    def __init__(self, task_id, desc, date, time, category, status, parent=None):
        super().__init__(parent)
        self.task_id = task_id
        self.is_done = (status == 'done')
        layout = QHBoxLayout(self); layout.setContentsMargins(5, 5, 5, 5)
        self.checkbox = QCheckBox(); self.checkbox.setChecked(self.is_done)
        self.checkbox.stateChanged.connect(self.on_checkbox_change)
        self.desc_label = QLabel(desc); self.desc_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        datetime_str = ""
        if date and date != 'None': datetime_str += f"{date} "
        if time and time != 'None': datetime_str += f"{time}"
        self.datetime_label = QLabel(datetime_str.strip()); self.datetime_label.setStyleSheet("color: gray; font-size: 9pt;")
        cat_label = QLabel(f"Cat: {category}"); cat_label.setStyleSheet("color: #888;")
        self.edit_button = QPushButton("Edit"); self.edit_button.setFixedSize(40, 25)
        self.edit_button.clicked.connect(lambda: self.edit_requested.emit(self.task_id))
        self.delete_button = QPushButton("Del"); self.delete_button.setFixedSize(40, 25); self.delete_button.setStyleSheet("color: red;")
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self.task_id))
        layout.addWidget(self.checkbox); layout.addWidget(self.desc_label, 1); layout.addWidget(self.datetime_label)
        layout.addSpacerItem(QSpacerItem(10, 0)); layout.addWidget(cat_label); layout.addSpacerItem(QSpacerItem(10, 0))
        layout.addWidget(self.edit_button); layout.addWidget(self.delete_button)
        self.setAttribute(Qt.WA_StyledBackground)  # Enable custom styling
        self.setStyleSheet("""TaskItemWidget {background: #ffffff;}""")

        self.update_appearance()

    def on_checkbox_change(self, state):
        self.is_done = (state == Qt.Checked)
        self.update_appearance()
        self.status_changed.emit(self.task_id, self.is_done)

    def update_appearance(self):
        font = self.desc_label.font(); font.setStrikeOut(self.is_done); self.desc_label.setFont(font)
        palette = self.desc_label.palette(); palette.setColor(self.desc_label.foregroundRole(), Qt.gray if self.is_done else Qt.black)
        self.desc_label.setPalette(palette); self.datetime_label.setVisible(not self.is_done)