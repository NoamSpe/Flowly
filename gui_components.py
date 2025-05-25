# ui_components.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
                             QPushButton, QLabel, QListWidget, QDialog, QFormLayout,
                             QComboBox, QDialogButtonBox, QCheckBox, QSpacerItem,
                             QSizePolicy, QRadioButton, QCalendarWidget,
                             QInputDialog, QMessageBox, QMenu)
from PyQt5.QtCore import Qt, QDate, QLocale, QPoint
from PyQt5.QtGui import QPixmap, QFont, QTextCharFormat, QBrush, QColor

from app_config import LOGO_TEXT_TRANSPARENT_PATH, CALENDAR_LOCALE, TASK_CATEGORIES

def setup_main_ui(app):
    """Sets up the main window UI elements for the FlowlyApp instance."""
    app.layout = QVBoxLayout(app)

    # Logo
    app.logo_label = QLabel(app)
    logo_pixmap = QPixmap(LOGO_TEXT_TRANSPARENT_PATH)
    logo_width = 150
    scaled_pixmap = logo_pixmap.scaledToWidth(logo_width, Qt.SmoothTransformation)
    app.logo_label.setPixmap(scaled_pixmap)
    app.layout.addWidget(app.logo_label, alignment=Qt.AlignCenter)

    content_layout = QHBoxLayout()

    # --- Calendar Pane ---
    cal_pane_layout = QVBoxLayout()
    cal_title_label = QLabel("Filter by Date:")
    cal_title_label.setAlignment(Qt.AlignCenter)
    app.calendar_widget = QCalendarWidget()
    app.calendar_widget.setGridVisible(True)
    app.calendar_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding) # Calendar takes vertical space
    app.calendar_widget.setVerticalHeaderFormat(QCalendarWidget.NoVerticalHeader)
    app.calendar_widget.setLocale(QLocale(CALENDAR_LOCALE))
    app.calendar_widget.clicked[QDate].connect(app.handle_calendar_date_clicked)
    
    cal_pane_layout.addWidget(cal_title_label)
    cal_pane_layout.addWidget(app.calendar_widget, 1)

    app.clearDateFilter_btn = QPushButton("Show All Dates")
    app.clearDateFilter_btn.clicked.connect(app.clear_calendar_selection)
    cal_pane_layout.addWidget(app.clearDateFilter_btn)

    uni_format = QTextCharFormat()
    uni_format.setForeground(QBrush(QColor('#000000')))
    for day_int in range(1, 8): # Qt.Monday is 1, Qt.Sunday is 7
        app.calendar_widget.setWeekdayTextFormat(Qt.DayOfWeek(day_int), uni_format)
    
    # content_layout.addLayout(cal_pane_layout, 0) # Calendar pane, less stretch factor

    # --- Task Area Pane ---
    task_area_layout = QVBoxLayout()

    # Input Area
    input_layout = QHBoxLayout()
    app.text_field = QLineEdit()
    app.text_field.setPlaceholderText("Enter new task...")
    app.text_field.returnPressed.connect(app.send_task)
    app.sendTask_btn = QPushButton("Add Task")
    app.sendTask_btn.clicked.connect(app.send_task)
    app.sendTask_btn.setObjectName("SendTaskBtn")
    app.record_btn = QPushButton("Record")
    app.record_btn.clicked.connect(app.record_task)
    input_layout.addWidget(app.text_field, 1)
    input_layout.addWidget(app.sendTask_btn)
    input_layout.addWidget(app.record_btn)
    task_area_layout.addLayout(input_layout)

    # List Controls
    list_controls_layout = QHBoxLayout()

    # Status filter
    status_filter_layout = QVBoxLayout()
    status_filter_label = QLabel("Filter by Status:")
    status_filter_layout.addWidget(status_filter_label)
    status_radios_layout = QHBoxLayout()
    app.Rstatus_all = QRadioButton("All")
    app.Rstatus_pending = QRadioButton("Pending")
    app.Rstatus_done = QRadioButton("Done")
    app.Rstatus_pending.setChecked(True)

    status_radios_layout.addWidget(app.Rstatus_all)
    status_radios_layout.addWidget(app.Rstatus_pending)
    status_radios_layout.addWidget(app.Rstatus_done)
    status_filter_layout.addLayout(status_radios_layout)

    app.Rstatus_all.toggled.connect(app.update_filters_display)
    app.Rstatus_pending.toggled.connect(app.update_filters_display)
    # app_instance.Rstatus_done.toggled.connect(app_instance.update_filters_display) # Handled by the other radio buttons

    list_controls_layout.addLayout(status_filter_layout)
    list_controls_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

    # Category filter
    app.category_checkboxes = {}
    category_filter_layout = QVBoxLayout()
    category_filter_label = QLabel("Filter by Category:")
    category_filter_layout.addWidget(category_filter_label)
    category_checkboxes_layout = QHBoxLayout()
    for cat in TASK_CATEGORIES: # from app_config
        cb = QCheckBox(cat)
        app.category_checkboxes[cat] = cb
        cb.stateChanged.connect(app.update_filters_display)
        category_checkboxes_layout.addWidget(cb)
    category_filter_layout.addLayout(category_checkboxes_layout)
    list_controls_layout.addLayout(category_filter_layout)

    list_controls_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

    # Sort Control
    sort_layout_container = QVBoxLayout() # Use QVBoxLayout to align with other filters
    sort_controls_layout = QHBoxLayout()
    sort_label = QLabel("Sort by:")
    app.sort_combo = QComboBox()
    app.sort_combo.addItems(["Due Date", "Urgency"])
    app.sort_combo.currentIndexChanged.connect(app.change_sort_mode)
    sort_controls_layout.addWidget(sort_label)
    sort_controls_layout.addWidget(app.sort_combo)
    sort_layout_container.addLayout(sort_controls_layout)

    list_controls_layout.addLayout(sort_layout_container)
    task_area_layout.addLayout(list_controls_layout)

    list_label_layout = QVBoxLayout()

    # Task List Label
    app.task_list_label = QLabel("Showing Tasks")
    app.task_list_label.setStyleSheet("font-weight: bold; padding: 5px 0px;")
    list_label_layout.addWidget(app.task_list_label)

    # Task List
    app.task_list = QListWidget()
    app.task_list.setSelectionMode(QListWidget.NoSelection)
    app.task_list.setFocusPolicy(Qt.NoFocus)
    app.task_list.setContextMenuPolicy(Qt.CustomContextMenu)
    app.task_list.customContextMenuRequested.connect(app.show_task_context_menu)
    list_label_layout.addWidget(app.task_list, 1)

    cal_list_layout = QHBoxLayout()
    cal_list_layout.addLayout(cal_pane_layout)
    cal_list_layout.addLayout(list_label_layout, 1)

    task_area_layout.addLayout(cal_list_layout, 1)

    # Button Layout
    button_layout = QHBoxLayout()
    app.refreshList_btn = QPushButton("Refresh List")
    app.refreshList_btn.clicked.connect(app.request_get_tasks)
    app.logout_btn = QPushButton("Logout")
    app.logout_btn.clicked.connect(app.handle_logout)
    button_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
    button_layout.addWidget(app.refreshList_btn)
    button_layout.addWidget(app.logout_btn)
    task_area_layout.addLayout(button_layout)

    content_layout.addLayout(task_area_layout, 1)

    app.layout.addLayout(content_layout)

    # Status Bar
    app.status_label = QLabel("Status: Initializing...")
    app.status_label.setStyleSheet("color: gray;")
    app.layout.addWidget(app.status_label)


def create_login_dialog(parent_widget):
    dialog = QDialog(parent_widget)
    dialog.setWindowTitle("Login / Sign Up")
    layout = QFormLayout(dialog)

    username_edit = QLineEdit(dialog)
    password_edit = QLineEdit(dialog)
    password_edit.setEchoMode(QLineEdit.Password)

    layout.addRow(QLabel("Username:"), username_edit)
    layout.addRow(QLabel("Password:"), password_edit)

    buttonBox = QDialogButtonBox(dialog)
    login_btn = buttonBox.addButton("Login", QDialogButtonBox.AcceptRole)
    login_btn.setObjectName("LoginBtn")
    signup_btn = buttonBox.addButton("Sign Up", QDialogButtonBox.ActionRole)
    cancel_btn = buttonBox.addButton(QDialogButtonBox.Cancel)
    layout.addRow(buttonBox)

    return dialog, username_edit, password_edit, login_btn, signup_btn, cancel_btn


def create_edit_task_dialog(parent_widget, task_data):
    task_id = task_data.get('TaskID')
    dialog = QDialog(parent_widget)
    dialog.setWindowTitle(f"Edit Task {task_id}")
    
    layout = QVBoxLayout(dialog)
    form_layout = QFormLayout()

    desc_edit = QLineEdit(task_data.get('TaskDesc', ''))
    date_edit = QLineEdit(task_data.get('Date', '') if task_data.get('Date') not in [None, 'None'] else '')
    date_edit.setPlaceholderText("YYYY-MM-DD")
    time_edit = QLineEdit(task_data.get('Time', '') if task_data.get('Time') not in [None, 'None'] else '')
    time_edit.setPlaceholderText("HH:MM:SS")
    
    category_combo = QComboBox()
    category_combo.addItems(TASK_CATEGORIES + ["Other"]) # Allow set categories or "Other"
    current_category = task_data.get('Category', 'Personal') # Default to personal or first in list
    if current_category not in TASK_CATEGORIES and current_category:
        category_combo.addItem(current_category)
    category_combo.setCurrentText(current_category)

    status_combo = QComboBox()
    status_combo.addItems(["pending", "done"])
    status_combo.setCurrentText(task_data.get('Status', 'pending'))

    form_layout.addRow("Description:", desc_edit)
    form_layout.addRow("Date:", date_edit)
    form_layout.addRow("Time:", time_edit)
    form_layout.addRow("Category:", category_combo)
    form_layout.addRow("Status:", status_combo)
    layout.addLayout(form_layout)

    btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
    layout.addWidget(btn_box)
    
    return dialog, {
        "desc": desc_edit, "date": date_edit, "time": time_edit,
        "category": category_combo, "status": status_combo
    }, btn_box


def update_calendar_display_highlights(calendar_widget, tasks_cache, highlighted_dates_set):
    if not calendar_widget: return

    default_format = QTextCharFormat() 
    today_qdate = QDate.currentDate()

    dates_to_clear = highlighted_dates_set.copy()
    dates_to_clear.add(today_qdate) 
    for old_highlight_qdate in dates_to_clear:
        calendar_widget.setDateTextFormat(old_highlight_qdate, default_format)
    highlighted_dates_set.clear()

    date_task_info = {} 
    for task_data in tasks_cache:
        try:
            task_status = task_data[5] 
            date_str = task_data[2]    
        except IndexError:
            continue # skip malformed task data

        if task_status == 'pending' and date_str and date_str != 'None':
            q_task_date = QDate.fromString(date_str, 'yyyy-MM-dd')
            if q_task_date.isValid():
                current_priority_str = date_task_info.get(q_task_date)
                is_overdue = q_task_date < today_qdate
                task_type_str = 'overdue' if is_overdue else 'upcoming'
                
                if current_priority_str == 'overdue':
                    continue 
                elif current_priority_str == 'upcoming' and task_type_str == 'overdue':
                    date_task_info[q_task_date] = 'overdue'
                elif not current_priority_str:
                    date_task_info[q_task_date] = task_type_str
    
    for q_date, task_type in date_task_info.items():
        fmt = QTextCharFormat() 
        if task_type == 'overdue':
            fmt.setForeground(QBrush(QColor(255, 0, 0))) 
        elif task_type == 'upcoming':
            fmt.setForeground(QBrush(QColor('#ff914d'))) 
        calendar_widget.setDateTextFormat(q_date, fmt)
        highlighted_dates_set.add(q_date)
    
    today_fmt = calendar_widget.dateTextFormat(today_qdate) 
    today_fmt.setBackground(QBrush(QColor('#ffede1'))) 
    calendar_widget.setDateTextFormat(today_qdate, today_fmt)
    highlighted_dates_set.add(today_qdate)