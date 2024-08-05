import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler



file_name = 'diabetes.csv'
raw_data = pd.read_csv(file_name)

print("Raw data:")
print(raw_data)

X = raw_data.drop('Outcome', axis=1)
y = raw_data['Outcome']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

print("Data after imputation:")
print(pd.DataFrame(X_imputed, columns=X.columns))

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_imputed)

print("Data after standardization:")
print(pd.DataFrame(X_standardized, columns=X.columns))

normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X_standardized)

print("Data after normalization:")
print(pd.DataFrame(X_normalized, columns=X.columns))

df_preprocessed = pd.DataFrame(X_normalized, columns=X.columns)
df_preprocessed['Outcome'] = y.values

print("Preprocessed data:")
print(df_preprocessed)

df_preprocessed.to_csv('preprocessed_diabetes_data.csv', index=False)

print("Data processing complete. The processed data has been saved to 'preprocessed_diabetes_data.csv'.")
