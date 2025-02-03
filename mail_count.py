import matplotlib.pyplot as plt
import numpy as np

email_counts=[[20,15,16,18,5,11],
              [12,11,18,16,3,14],
              [13,14,21,8,5,12],
              [11,12,22,3,2,18],
              [15,6,9,1,0,11],
              [12,13,14,9,12,7],
              [6,5,11,2,3,5]]

time_intervals=['8:00 AM-12:00 PM','12:00 PM-4:00 PM','4:00 PM-8:00 PM','8:00 PM-12:00 AM','12:00 AM-4:00 AM','4:00 AM-8:00 AM']
days=['Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday']

n_intervals=len(time_intervals)
n_days=len(days)

fig,ax=plt.subplots(figsize=(8,6))
bar_width=0.1
x=np.arange(n_intervals)

for i in range(n_days):
    ax.bar(x+i*bar_width,email_counts[i],width=bar_width,label=days[i])

ax.set_xlabel('Time intervals')
ax.set_ylabel('Number of Emails')
ax.set_xticks(x + bar_width*(n_days-1)/2)
ax.set_xticklabels(time_intervals,rotation=45)
ax.set_title('Comaprison of mails recieved in different time intervals across days')
ax.legend(title='Days of the week')

plt.tight_layout()
plt.show()

