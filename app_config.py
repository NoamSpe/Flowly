# app_config.py

# Urgency sorting factors
CATEGORY_FACTORS = {
    "Work": 0.8,
    "School": 0.6,
    "Personal": 0.5,
    "Household": 0.3,
    "Health": 0.7,
    "_default_": 0.5  # Default for unknown categories
}
DTIME_WEIGHT = 0.6
CATEGORY_WEIGHT = 0.4
TIME_URGENCY_K = 0.01
DF_TIME = 0.1

# Network
SERVER_HOST = '10.100.102.96' # Replace with your server's IP address
SERVER_PORT = 4320

# Assets
LOGO_SQUARE_PATH = 'assets/FlowlyLogo-Square.png'
LOGO_TEXT_TRANSPARENT_PATH = 'assets/FlowlyLogo-TextTransparent.png'
STYLESHEET_PATH = "style.qss"

# Other UI
APP_TITLE_BASE = "Flowly - Task Manager"
CALENDAR_LOCALE = 'en_US'
TASK_CATEGORIES = ["Work", "School", "Personal", "Household", "Health"]