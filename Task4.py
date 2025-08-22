# 1) Imports
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Helpful check so users know what version is running
try:
    import sklearn
    print('scikit-learn version:', sklearn.__version__) 
except Exception as e:
    print('scikit-learn not found. Please install it with pip or conda and restart the kernel.')
    raise
# 2) Tiny, embedded dataset (no downloads)
# 'label': 'spam' or 'ham' (not spam)
data_samples = [
    # spam examples
    ('spam', 'You won a prize! Claim now to receive your reward.'),
    ('spam', 'Limited time offer. Click the link to get cash back.'),
    ('spam', 'Congratulations! You are selected for a free gift card.'),
    ('spam', 'URGENT: Your account qualifies for a bonus. Reply YES.'),
    ('spam', 'Free entry in our weekly draw. Join today.'),
    ('spam', 'Get rich quickly with this simple method. Learn more.'),
    ('spam', 'Exclusive deal just for you. Don\'t miss it.'),
    ('spam', 'Winner! Text WIN to 90090 to collect.'),
    ('spam', 'Final notice: You have an unclaimed reward.'),
    ('spam', 'Hot sale is live. Grab your discount voucher.'),
    ('spam', 'Zero-cost trial: activate your premium access now.'),
    ('spam', 'Your number was drawn. Confirm to receive payment.'),
    ('spam', 'Double your money in a week. Read the guide.'),
    ('spam', 'Act now to unlock VIP benefits. Limited seats.'),
    ('spam', 'No fees, no risk. Start earning today.'),
    ('spam', 'Your loan is approved. Click to finalize.'),
    ('spam', 'Exclusive: pre-approved credit increase available.'),
    ('spam', 'You have a parcel waiting. Pay small fee to release.'),
    ('spam', 'Install this app to win rewards instantly.'),
    ('spam', 'Special bonus for first 100 users! Enroll now.'),

    # ham examples
    ('ham', 'Are we still meeting at 5 pm today?'),
    ('ham', 'Please call me when you reach home.'),
    ('ham', 'Happy birthday! Have a great day.'),
    ('ham', 'I will send the report by evening.'),
    ('ham', 'Let\'s have lunch at the new cafe tomorrow.'),
    ('ham', 'Did you get the notes from class?'),
    ('ham', 'I\'m running late, be there in 10 minutes.'),
    ('ham', 'Thanks for the update. Looks good to me.'),
    ('ham', 'Where are you right now?'),
    ('ham', 'The meeting is moved to Monday morning.'),
    ('ham', 'Don\'t forget to bring your ID card.'),
    ('ham', 'I\'ll call you after this session ends.'),
    ('ham', 'Great job on the presentation!'),
    ('ham', 'Let me know if you need any help.'),
    ('ham', 'See you at the event this weekend.'),
    ('ham', 'Can you share the file on email?'),
    ('ham', 'I\'m fine with either option you choose.'),
    ('ham', 'We can reschedule if you\'re busy today.'),
    ('ham', 'Don\'t worry, I\'ll handle it.'),
    ('ham', 'Please review the draft when you get time.'),
]

df = pd.DataFrame(data_samples, columns=['label', 'message'])
df.head()

# 3) Train-test split
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
len(X_train), len(X_test)

# 4) Vectorize text -> counts
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)
X_train_counts.shape, X_test_counts.shape


# 5) Train model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_counts, y_train)
print('Model trained.')

# 6) Evaluate
y_pred = model.predict(X_test_counts)
acc = accuracy_score(y_test, y_pred)
print('Accuracy:', round(acc, 3))
print('\nClassification Report:\n')
print(classification_report(y_test, y_pred, digits=3))

# 7) Confusion Matrix (matplotlib only)
cm = confusion_matrix(y_test, y_pred, labels=['ham','spam'])
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest')  # no explicit colors per instructions
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(['ham','spam'])
ax.set_yticklabels(['ham','spam'])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha='center', va='center')
fig.tight_layout()
plt.show()

# 8) Try your own messages
examples = [
    'Congratulations! You have been selected for a cash reward.',
    'Are we still on for dinner tonight?',
    'Final notice: claim your bonus now!',
    'Please call me when you are free.'
]
for msg in examples:
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    print(f'{msg} -> {pred}')
