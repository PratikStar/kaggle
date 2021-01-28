import pandas as pd

melbourne_file_path = './melb_data.csv'

melb_data = pd.read_csv(melbourne_file_path)

#print(melb_data.describe())

melb_data = melb_data.dropna(axis=0)

print(melb_data.columns)

y = melb_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

x = melb_data[melbourne_features]

print(x.describe())



from sklearn.tree import DecisionTreeRegressor
melbourne_model = DecisionTreeRegressor(random_state=1)


melbourne_model.fit(x, y)

print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")

print(melbourne_model.predict(x.head()))

#print(melb_data.head().Price)


from sklearn.metrics import mean_absolute_error

predicted_prices = melbourne_model.predict(x)

print(mean_absolute_error(y, predicted_prices))