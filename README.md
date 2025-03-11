📌# Investigating-No-Show-Appointments
This project analyzes factors influencing patient no-shows in medical appointments. Using data from over 100,000 appointments in Brazil, the analysis explores trends in age, medical conditions, and SMS reminders to understand their impact on attendance.

📊 Dataset Information

* Dataset Name: No-Show Appointments

Source: Kaggle - No-Show Appointments Dataset (https://www.kaggle.com/datasets/joniarroba/noshowappointments)

Size: 110,527 rows, 14 columns

Key Features:

PatientId, AppointmentID: Identifiers

Age, Gender, Neighbourhood: Demographics

Hypertension, Diabetes, Alcoholism, Handicap: Medical conditions

SMS_received: Whether an SMS reminder was sent

No-show: Target variable (Yes = Missed appointment, No = Attended)

📌 Key Questions Explored

Does age affect appointment attendance?

Do medical conditions (Hypertension, Diabetes, Alcoholism) impact attendance?

Does receiving an SMS reminder reduce no-shows?

🛠️ Tools & Libraries Used

Python: Core programming language

Pandas, NumPy: Data cleaning, statistical analysis

Matplotlib: Data visualization

Jupyter Notebook: Interactive analysis

🔍 Summary of Findings

Age: Attendance increases with age, peaking at 66-100 years.

Medical Conditions: No significant impact on no-show rates.

SMS Reminders: Unexpectedly, patients who received an SMS had a higher no-show rate (27.6%). This suggests reminders might be targeted at patients already prone to missing appointments.

🛑 Limitations & Future Work

Correlation, Not Causation: The analysis only identifies trends and does not establish causal relationships.

Missing Context: Factors like transportation, work conflicts, or appointment urgency are not included.

Future Research: A randomized control study on SMS effectiveness could provide better insights.

✍️ Author: Hillary | 🚀 Project Completed: 03/11/2025
