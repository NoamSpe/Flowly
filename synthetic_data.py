import pandas as pd
from faker import Faker as fk
import random

# Initialize Faker instance
fake = fk()
print(fake.date_this_month())

# Define the labels
LABELS = ['O', 'B-Task', 'I-Task', 'B-Date', 'I-Date', 'B-Time', 'I-Time']
NUM_CLASSES = len(LABELS)

# Task template and example sentence patterns
TASKS = [
    "Book a dentist appointment for next Friday at 3:00 PM",
    "Book flight tickets today at 3:00 PM.",
    "I need to send an update mail to the team at 9:00 am tomorrow.",
    "Pick up my dry cleaning in two days",
    "Call Mom on her birthday, which is on the 22nd at 5:00 PM.",
    "Buy tickets for the concert on May 5th at 8:00 PM.",
    "Submit the project report by 3pm on Thursday",
    "Complete the monthly performance review by the deadline on February 28th at 23:59.",
    "I need to study chapter 5 of the biology textbook by Monday.",
    "Book a table at Nando's for Friday at 3:00 pm",
    "Check mail confirmation today at 14:30",
    "I have to finish the presentation for the team meeting at 11am tomorrow",
    "Pick up Steve's medical prescriptions on March 5th at 2:00 PM.",
    "Call John on his anniversary, which is tomorrow next week",
    "Learn lyrics for the concert on May 12th at 21:30",
    "Submit preliminary idea report by 3:00 PM on Wednesday",
    "Go through the monthly expense summary by the deadline on Sep 13th at 14:21.",
    "I need to buy a ruler for the physics project by Tuesday.",
    "I need to get a hammer by Monday.",
    "Get a taxi for the tech conference on Friday by tomorrow at 15:00",
    "Mow the lawn tomorrow",
    "Prepare for the math quiz on Friday",
    "Order a new phone case by tomorrow at noon"
]

profession = ["dentist", "doctor", "nurse", "lawyer", 
    "architect", "photographer", "chef", "therapist", "plumber", 
    "receptionist", "manager"]
time_period = ["AM", "PM", "am", "pm"]
relative_date = ["today", "tomorrow"]
activity = ["flight", "concert", "party", "conference", "webinar", "movie", "trip", "vacation", 
    "shopping", "haircut", "wedding", "birthday","graduation", "reunion", "festival", "tournament", 
    "exhibition", "fair", "carnival", "parade", "rally", "protest", "demonstration", "gathering",
    "ceremony", "ritual", "observance", "celebration", "commemoration", "memorial", "tribute"]
verb_1 = ["book", "send", "watch", "call", "complete", "study", "buy", "finish", "submit", "analyze", 
          "check", "learn","design","test","clean","prepare","read", "arrange", "organize", "attend", 
          "prepare", "review", "schedule", "plan", "reserve", "confirm", "finalize", "coordinate", "manage", "mow"]
verb_2 = ["pick up", "go through", "go over","look into", "check out", "follow up","sort out", "deal with","wrap up"]
number_12 = [_ for _ in range(1,13)]
number = [_ for _ in range(1,13)] + ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
person_list = ["Mom", "Dad", "Dana", "Shani", "Ron", "Lia", "Tahel", "Noa", "Noam", "Gal", "Dror", "Daniel", "John"]
day_in_month = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "21st", "22nd", "23rd", "24th", "25th", "26th", "27th", "28th", "29th", "30th", "31st"]
month = ["Jan", "January", "Feb", "February", "Mar", "March", "Apr", "April", "May", "Jun", "June", 
         "Jul", "July", "Aug", "August", "Sep", "September", "Oct", "October", "Nov", "November", "Dec", "December"]
event_2 = ["tech conference", "team meeting", "physics project", "math quiz","board meeting", "client presentation", 
           "product launch", "training session", "networking event", "press conference", "trade show",  "panel discussion"]
obj_1 = ["lawn","garden","hammer","pen","ruler","book","phone","tablet","laptop","camera",
         "watch","clock","computer","printer","monitor","keyboard","mouse","radio","headphones"]
obj_2 = ["phone case", "car key", "ice cream", "coffee table", "paper tray", "tooth brush", "shower gel", "hand mirror", "gym bag", "photo album"]

TASK_TEMPLATES = [
    "{verb_1} a {profession} appointment for next {weekday} at {hour} {time_period}",
    "{verb_1} {activity} tickets {relative_date} at {hour} {time_period}.",
    "I need to {verb_1} an update mail to the team at {hour} {time_period} {relative_date}",
    "{verb_2} my dry cleaning in {number} days",
    "{verb_1} {person} on her {activity} which is on the {day_in_month} at {number} {time_period}.",
    "{verb_1} tickets for the {activity} on {month} {day_in_month} at {hour} {time_period}.", #hour
    "{verb_1} the project report by {number_12}{time_period} on {weekday}",
    "{verb_1} the monthly performance review by the deadline on {month} {day_in_month} at {hour}.", #hour
    "I need to {verb_1} chapter {number} of the biology textbook by {weekday}.", #day of week
    "{verb_1} a table at {person}'s for {weekday} at {hour} {time_period}", #day of week #hour
    "{verb_1} mail confirmation {relative_date} at {hour}", #hour
    "I have to {verb_1} the presentation for the {event_2} at {number} {time_period} {relative_date}",
    "{verb_2} {person}'s medical prescriptions on {month} {day_in_month} at {hour} {time_period}.", #hour
    "{verb_1} {person} for his {activity} which is {relative_date} next week",
    "{verb_1} lyrics for the {activity} on {month} {day_in_month} at 21:30", #hour
    "{verb_1} preliminary idea report by {hour} {time_period} on {weekday}", #hour #day of week
    "{verb_2} the monthly expense summary by the deadline on {month} {day_in_month} at {hour}", #hour
    "I need to {verb_1} a {obj_1} for the {event_2} by {weekday}.", #day of week
    "I need to {verb_1} a {obj_1} by {weekday}.", #day_in_month
    "{verb_1} a taxi for the {event_2} on {weekday} by {relative_date} at {hour}", #day of week #hour
    "{verb_1} the {obj_1} {relative_date}",
    "{verb_1} for the {event_2} on {weekday}", #day of week
    "{verb_1} a new {obj_2} by {relative_date} at noon", #hour
    "Remind me to {verb_1} the {event_2} report before {weekday} at {hour} {time_period}",
    "I have a {event_2} scheduled for {weekday} next week at {hour}",
    "Don't forget to {verb_1} the {activity} tickets by {relative_date} at {hour}",
    "{verb_1} {person}'s {activity} gift before {month} {day_in_month}",
    "make sure to {verb_2} the {activity} certification by {relative_date} at midnight",
]

LABELS = [
    "B-Task,O,B-Task,I-Task,O,B-Date,I-Date,O,B-Time,I-Time",
    "B-Task,I-Task,I-Task,B-Date,O,B-Time,I-Time",
    "O,O,O,B-Task,O,B-Task,I-Task,I-Task,O,B-Task,O,B-Time,I-Time,B-Date",
    "B-Task,I-Task,O,B-Task,I-Task,B-Date,I-Date,I-Date",
    "B-Task,I-Task,O,O,B-Task,O,O,B-Date,I-Date,I-Date,O,B-Time,I-Time",
    "B-Task,I-Task,I-Task,O,B-Task,O,B-Date,I-Date,O,B-Time,I-Time",
    "B-Task,O,B-Task,I-Task,O,B-Time,O,B-Date",
    "B-Task,O,B-Task,I-Task,I-Task,O,O,O,O,B-Date,I-Date,O,B-Time",
    "O,O,O,B-Task,I-Task,I-Task,O,O,O,O,O,B-Date",
    "B-Task,O,B-Task,I-Task,I-Task,O,B-Date,O,B-Time,I-Time",
    "B-Task,I-Task,I-Task,B-Date,O,B-Time",
    "O,O,O,B-Task,O,B-Task,I-Task,I-Task,I-Task,I-Task,O,B-Time,I-Time,B-Date",
    "B-Task,I-Task,I-Task,I-Task,I-Task,O,B-Date,I-Date,O,B-Time,I-Time",
    "B-Task,I-Task,O,O,B-Task,O,O,B-Date,I-Date,I-Date",
    "B-Task,I-Task,I-Task,O,B-Task,O,B-Date,I-Date,O,B-Time",
    "B-Task,I-Task,I-Task,I-Task,O,B-Time,I-Time,O,B-Date",
    "B-Task,I-Task,O,B-Task,I-Task,I-Task,O,O,O,O,B-Date,I-Date,O,B-Time",
    "O,O,O,B-Task,I-Task,I-Task,I-Task,O,B-Task,I-Task,O,B-Date",
    "O,O,O,B-Task,I-Task,I-Task,O,B-Date",
    "B-Task,O,B-Task,I-Task,O,B-Task,I-Task,O,O,O,B-Date,O,B-Time",
    "B-Task,O,B-Task,B-Date",
    "B-Task,I-Task,O,B-Task,I-Task,O,B-Date",
    "B-Task,I-Task,I-Task,I-Task,I-Task,O,B-Date,O,B-Time",
    "O,O,O,B-Task,O,B-Task,I-Task,I-Task,O,B-Date,O,B-Time,I-Time",
    "O,O,O,B-Task,I-Task,O,O,B-Date,I-Date,I-Date,O,B-Time",
    "O,O,O,B-Task,O,B-Task,I-Task,O,B-Date,O,B-Time",
    "B-Task,I-Task,I-Task,I-Task,O,B-Date,I-Date",  # Existing 27 labels
    "O,O,O,B-Task,I-Task,O,B-Task,I-Task,O,B-Date,O,B-Time"  # Added for template 28
]

data = []

for _ in range(5000):
    # Randomly choose an hour in the format hh:mm (12-hour format)
    time = f"{random.randint(1, 12)}:{random.choice([0, 15, 30, 45]):02}"
    person_choice = random.choice(person_list + [fake.first_name()])

    # Generate the task using random choices and the chosen time
    task_template = random.choice(TASK_TEMPLATES)
    task_index = TASK_TEMPLATES.index(task_template)

    task = task_template.format(
        verb_1=random.choice(verb_1),
        verb_2=random.choice(verb_2),
        profession=random.choice(profession),
        weekday=random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
        hour=time,
        time_period=random.choice(time_period),
        relative_date=random.choice(relative_date),
        activity=random.choice(activity),
        person=person_choice,
        day_in_month=random.choice(day_in_month),
        month=random.choice(month),
        event_2=random.choice(event_2),
        number=random.choice(number),
        number_12 = random.choice(number_12),
        obj_1 = random.choice(obj_1),
        obj_2 = random.choice(obj_2)
    )

    label = LABELS[task_index]
    data.append({"Task":task, "Label":label})

df = pd.DataFrame(data)
df.to_csv("NER_Data.csv", index=False)
print("data saved to 'NER_Data.csv'")