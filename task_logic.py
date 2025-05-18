# task_logic.py
import datetime
import math
from app_config import CATEGORY_FACTORS, DTIME_WEIGHT, CATEGORY_WEIGHT, TIME_URGENCY_K, DF_TIME

def calculate_urgency(task_data):
    """Calculates dynamic urgency score for a single task."""
    try:
        # TaskID, TaskDesc, DateStr, TimeStr, Category, Status
        _, _, date_str, time_str, category, _ = task_data
    except (ValueError, IndexError): # Handle cases where task_data might not have 6 elements
        print(f"WARN: Could not unpack task data for urgency calculation: {task_data}")
        return 0

    now = datetime.datetime.now()
    due_datetime = None

    if date_str and date_str != 'None':
        try:
            parsed_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            parsed_time = datetime.time.min
            if time_str and time_str != 'None':
                try:
                    time_str_clean = time_str.split('.')[0]
                    parsed_time = datetime.datetime.strptime(time_str_clean, '%H:%M:%S').time()
                except ValueError:
                    print(f"WARN: Invalid time format '{time_str}', using 00:00:00")
            due_datetime = datetime.datetime.combine(parsed_date, parsed_time)
        except ValueError:
            print(f"WARN: Invalid date format '{date_str}', cannot determine due date.")

    time_factor = DF_TIME
    if due_datetime:
        delta = due_datetime - now
        delta_hours = delta.total_seconds() / 3600.0
        if delta_hours <= 0: # Overdue or due now
            time_factor = 1.0
        else:
            # Ensure time_factor doesn't become excessively small for very distant tasks
            time_factor = max(0.0, math.exp(-TIME_URGENCY_K * delta_hours))


    category_factor = CATEGORY_FACTORS.get(category, CATEGORY_FACTORS["_default_"]) if category else CATEGORY_FACTORS["_default_"]
    urgency_score = (DTIME_WEIGHT * time_factor) + (CATEGORY_WEIGHT * category_factor)
    return urgency_score

def sort_tasks_by_mode(tasks, mode):
    """Sorts tasks by the given mode ('urgency' or 'due_date')."""
    if not tasks:
        return []
    
    sorted_tasks = []
    if mode == 'urgency':
        tasks_with_scores = [(calculate_urgency(task), task) for task in tasks]
        # Primary sort by score (desc), secondary by due date (asc) for tasks with same score
        tasks_with_scores.sort(key=lambda item: (-item[0], get_datetime_from_task(item[1]) or datetime.datetime.max))
        sorted_tasks = [item[1] for item in tasks_with_scores]
    elif mode == 'due_date':
        sorted_tasks = sorted(tasks, key=lambda task: get_datetime_from_task(task) or datetime.datetime.max)
    else:
        sorted_tasks = tasks # Should not happen with combobox
    return sorted_tasks

def get_datetime_from_task(task_data):
    """Helper to get a datetime object from task data for sorting, or None."""
    try:
        date_str = task_data[2]
        time_str = task_data[3]
    except IndexError:
        return None

    if date_str and date_str != 'None':
        try:
            parsed_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            parsed_time = datetime.time.min
            if time_str and time_str != 'None':
                try:
                    time_str_clean = time_str.split('.')[0]
                    parsed_time = datetime.datetime.strptime(time_str_clean, '%H:%M:%S').time()
                except ValueError: pass # Keep min time if time is invalid
            return datetime.datetime.combine(parsed_date, parsed_time)
        except ValueError:
            return None # Invalid date format
    return None # No date

def filter_and_sort_tasks(tasks_cache, selected_calendar_date, status_filter_state, 
                           category_checkboxes_state, current_sort_mode_ui):
    """
    Applies filters and then sorts tasks.
    status_filter_state: {'all': bool, 'pending': bool, 'done': bool}
    category_checkboxes_state: {category_name: bool}
    current_sort_mode_ui: 'due_date' or 'urgency' (from combobox)
    """
    from PyQt5.QtCore import QDate # Local import

    tasks_to_process = list(tasks_cache)

    # 1. Calendar date filter
    if selected_calendar_date:
        date_filter_tasks = []
        for task_data in tasks_to_process:
            date_str = task_data[2] if len(task_data) > 2 else None
            if date_str and date_str != 'None':
                q_task_date = QDate.fromString(date_str, 'yyyy-MM-dd')
                if q_task_date.isValid() and q_task_date == selected_calendar_date:
                    date_filter_tasks.append(task_data)
        tasks_to_process = date_filter_tasks

    # 2. Status filter
    selected_status_str = 'all'
    if status_filter_state.get('pending'): selected_status_str = 'pending'
    elif status_filter_state.get('done'): selected_status_str = 'done'
    
    filtered_by_status = []
    for task in tasks_to_process:
        task_status = task[5] if len(task) > 5 else None
        if selected_status_str == 'all' or task_status == selected_status_str:
            filtered_by_status.append(task)
    
    # 3. Category filter
    selected_categories = [cat for cat, checked in category_checkboxes_state.items() if checked]
    if not selected_categories:
        selected_categories = list(category_checkboxes_state.keys()) # If none checked, show all
        
    category_filtered_tasks = []
    for task in filtered_by_status:
        task_cat = task[4] if len(task) > 4 else None
        if task_cat in selected_categories:
            category_filtered_tasks.append(task)

    # 4. Sort tasks
    effective_sort_mode = current_sort_mode_ui
    
    if selected_status_str == 'all':
        pending_tasks = [task for task in category_filtered_tasks if (task[5] if len(task) > 5 else None) == 'pending']
        done_tasks = [task for task in category_filtered_tasks if (task[5] if len(task) > 5 else None) == 'done']
        
        sorted_pending = sort_tasks_by_mode(pending_tasks, effective_sort_mode)
        sorted_done = sort_tasks_by_mode(done_tasks, 'due_date') # Done tasks always by due_date
        
        final_tasks = (sorted_pending or []) + (sorted_done or [])
    elif selected_status_str == 'done':
        final_tasks = sort_tasks_by_mode(category_filtered_tasks, 'due_date')
    else: # 'pending'
        final_tasks = sort_tasks_by_mode(category_filtered_tasks, effective_sort_mode)
        
    return final_tasks