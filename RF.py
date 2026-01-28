import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# 1. Connect and load tables
# -----------------------------** changed from geeks for geeks to work with sql db
conn = sqlite3.connect("plants.db")
pump_log = pd.read_sql_query("SELECT * FROM pump_log", conn)
sensors = pd.read_sql_query("SELECT * FROM sensors", conn)
conn.close()

# -----------------------------
# 2. Prepare timestamps
# ----------------------------- ** additions made by copilot to drop unnecessary values of timestamps and to get timestamps to work correctly
pump_log['timestamp'] = pd.to_datetime(pump_log['timestamp'])
sensors['timestamp'] = pd.to_datetime(sensors['timestamp'])

pump_log = pump_log.dropna(subset=['timestamp']).sort_values('timestamp')
sensors = sensors.dropna(subset=['timestamp']).sort_values('timestamp')

# -----------------------------
# 3. Merge sensor readings with pump events
# -----------------------------** adition made by copilot to merge data, works better going backwards because I want the last known value not the offset

pump_log = pd.merge_asof(pump_log, sensors, on='timestamp', direction='backward')

# -----------------------------
# 4. date time
# -----------------------------** addition by copilot to improve the model performance and get the time till next water in date time
pump_log['time_since_last_water'] = pump_log['timestamp'].diff().dt.total_seconds().fillna(0)
pump_log['time_until_next'] = (pump_log['timestamp'].shift(-1) - pump_log['timestamp']).dt.total_seconds()
pump_log = pump_log.dropna(subset=['time_until_next'])

# Add time-based features (helps accuracy)
pump_log['hour_of_day'] = pump_log['timestamp'].dt.hour
pump_log['day_of_week'] = pump_log['timestamp'].dt.dayofweek

# -----------------------------
# 5. Train/test split
# -----------------------------** changed by me from geeks for geeks-- values also chaged to match my preferences
X = pump_log[['moisture_1', 'moisture_2', 'moisture_3',
              'time_since_last_water', 'hour_of_day', 'day_of_week']]
y = pump_log['time_until_next']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=600)

# -----------------------------
# 6. Train Random Forest (tuned)
# ----------------------------- ** values changed by me from source datacamp and scikit learn
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=2,
    random_state=600
)
rf.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate
# ----------------------------- ** from geeks for geeks changed from mse to mae
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae/60:.2f} minutes")

# -----------------------------
# 8. Predict next watering time
# ----------------------------- ** print from geeks for geeks code to determine next watering in ideal time format from copilot
latest = pump_log.iloc[-1]
latest_features = pd.DataFrame([[
    latest['moisture_1'],
    latest['moisture_2'],
    latest['moisture_3'],
    latest['time_since_last_water'],
    latest['hour_of_day'],
    latest['day_of_week']
]], columns=X.columns)

pred_seconds = rf.predict(latest_features)[0]
pred_time = latest['timestamp'] + pd.to_timedelta(pred_seconds, unit='s')

if pred_seconds < 3600:
    print(f"Predicted next watering in {int(pred_seconds//60)}m {int(pred_seconds%60)}s")
else:
    print(f"Predicted next watering in {pred_seconds/3600:.1f}h")

print("Predicted next watering timestamp:", pred_time)


importances = rf.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance:.4f}")
