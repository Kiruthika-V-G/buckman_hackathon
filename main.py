

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('sample.csv')



# Gender distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(x='Marital Status', data=df)
plt.title('Marital Status Distribution')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(y='Role', data=df, order=df['Role'].value_counts().index)
plt.title('Employment Roles Distribution')
plt.xlabel('Count')
plt.ylabel('Role')
plt.show()


df['Income Lower'] = df['Household Income'].str.extract(r'US\$\s*(\d+)')
df['Income Upper'] = df['Household Income'].str.extract(r'US\$\s*(\d+)').astype(float)
df.loc[df['Household Income'].str.contains('Above'), 'Income Lower'] = df['Income Upper']
df['Income Lower'] = df['Income Lower'].astype(float)

# Visualize income distribution
plt.figure(figsize=(10, 6))
plt.hist(df['Income Lower'], bins=20, color='skyblue', edgecolor='black')
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



# Percentage of household income invested
plt.figure(figsize=(10, 6))
sns.histplot(df['Percentage of Investment'], bins=20, kde=True, color='skyblue')
plt.title('Percentage of Household Income Invested', fontsize=16)
plt.xlabel('Percentage', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Sources of awareness about investments
plt.figure(figsize=(10, 6))
sns.countplot(y='Source of Awareness about Investment', data=df, order=df['Source of Awareness about Investment'].value_counts().index, palette='muted')
plt.title('Sources of Awareness about Investments', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Source', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Knowledge levels
plt.figure(figsize=(8, 6))
sns.countplot(x='Knowledge level about different investment product', data=df, palette='pastel')
plt.title('Knowledge Levels', fontsize=16)
plt.xlabel('Knowledge Level', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Influencers
plt.figure(figsize=(10, 6))
sns.countplot(y='Investment Influencer', data=df, order=df['Investment Influencer'].value_counts().index, palette='dark')
plt.title('Influencers Affecting Investment Decisions', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Influencer', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Risk levels
plt.figure(figsize=(8, 6))
sns.countplot(x='Risk Level', data=df, palette='bright')
plt.title('Risk Levels', fontsize=16)
plt.xlabel('Risk Level', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Reasons for investment
plt.figure(figsize=(10, 6))
sns.countplot(y='Reason for Investment', data=df, order=df['Reason for Investment'].value_counts().index, palette='colorblind')
plt.title('Reasons for Investment', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Reason', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()




print("First few rows of the dataset:")
print(df.head())


print("\nSummary statistics of numerical variables:")
print(df.describe())


df.drop(columns=['Sno'], inplace=True)

# Explore correlation matrix
numeric_columns = df.select_dtypes(include='number').columns
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


df = pd.read_csv('/content/drive/MyDrive/buckman/sample.csv')

df = df.dropna()
df = df.drop_duplicates()

def parse_percent(percent_str):
    if percent_str == "Don't Want to Reveal":
        return 10
    elif 'Above' in percent_str:
        upper_limit = int(percent_str.split('Above ')[1].rstrip('%'))
        return upper_limit + 1
    elif 'Upto' in percent_str:
        upper_limit = int(percent_str.split('Upto ')[1].rstrip('%'))
        return upper_limit
    else:
        percent_str = str(percent_str)
        percent_str = percent_str.rstrip('%')
        percent_str = percent_str.replace("%", '')
        if 'to' in percent_str:
            lower, upper = map(int, percent_str.split(' to '))
            return (lower + upper) / 2
        else:
            return int(percent_str)

# Apply the parse_percent function to the `Percentage of Investment` column
df['Percentage of Investment'] = df['Percentage of Investment'].apply(parse_percent)

# Convert all string columns to numerical columns using the LabelEncoder
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop(columns=['Return Earned'])
y = df['Return Earned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
# Train the random forest classifier on the training set
clf.fit(X_train_scaled, y_train)



y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # Specify average parameter
recall = recall_score(y_test, y_pred, average='macro')  # Specify average parameter
f1 = f1_score(y_test, y_pred, average='macro')  # Specify average parameter

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)




new_data = pd.read_csv('test.csv')
def parse_percent(percent_str):
    if percent_str == "Don't Want to Reveal":
        return 10
    elif 'Above' in percent_str:
        upper_limit = int(percent_str.split('Above ')[1].rstrip('%'))
        return upper_limit + 1
    elif 'Upto' in percent_str:
        upper_limit = int(percent_str.split('Upto ')[1].rstrip('%'))
        return upper_limit
    else:
        percent_str = str(percent_str)
        percent_str = percent_str.rstrip('%')
        percent_str = percent_str.replace("%", '')
        if 'to' in percent_str:
            lower, upper = map(int, percent_str.split(' to '))
            return (lower + upper) / 2
        else:
            return int(percent_str)

print(new_data['Percentage of Investment'])
new_data['Percentage of Investment'] = new_data['Percentage of Investment'].apply(parse_percent)
print(new_data['Percentage of Investment'])

scaler = StandardScaler()
new_data[['Percentage of Investment']] = scaler.fit_transform(new_data[['Percentage of Investment']])

le = LabelEncoder()
for col in new_data.columns:
    if new_data[col].dtype == 'object':
        new_data[col] = le.fit_transform(new_data[col])

X_new = new_data.drop(['Return Earned'], axis=1)
y_new = new_data['Return Earned']

# Predict using the trained model
y_pred_new = clf.predict(X_new)
print(y_pred_new)

accuracy_new = accuracy_score(y_new, y_pred_new)
precision_new = precision_score(y_new, y_pred_new, average='macro')
recall_new = recall_score(y_new, y_pred_new, average='macro')
f1_new = f1_score(y_new, y_pred_new, average='macro')

print('Accuracy on New Data:', accuracy_new)
print('Precision on New Data:', precision_new)
print('Recall on New Data:', recall_new)
print('F1-score on New Data:', f1_new)

invest_decision = ['Invest' if label == 1 else 'Not Invest' for label in y_pred_new]
new_data['Invest Decision'] = invest_decision
print(new_data[['Invest Decision']])
