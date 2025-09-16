import datetime

def ProcessWeek(year, week_number):
    # Loop through ISO weekdays 1 (Monday) to 7 (Sunday)
    day_date = []
    for weekday in range(1, 8):
        # Construct date from ISO year, week, and weekday
        day_date.append(datetime.date.fromisocalendar(year, week_number, weekday))
    return day_date