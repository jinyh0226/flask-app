# 2단계: Keras 모델 개발 및 저장

# 2-1. 라이브러리 불러오기
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

# 2-2. 시드 설정
np.random.seed(42)
tf.random.set_seed(42)

# 2-3. 데이터 불러오기
fires_prepared = joblib.load("fires_prepared.pkl")
fires_labels = joblib.load("fires_labels.pkl")
strat_test_set = joblib.load("strat_test_set.pkl")

# 2-4. 훈련/검증/테스트셋 분리
# 먼저 전체 데이터를 훈련+검증, 테스트셋으로 나누기 (strat_test_set과 같은 수만큼 test로 분리)
test_size = len(strat_test_set)
X_full_train, X_test = fires_prepared[:-test_size], fires_prepared[-test_size:]
y_full_train, y_test_log = fires_labels[:-test_size], fires_labels[-test_size:]

# 이제 훈련+검증셋을 다시 훈련/검증으로 분리
X_train, X_valid, y_train, y_valid = train_test_split(X_full_train, y_full_train, test_size=0.2, random_state=42)

# y_test 실제값도 준비 (로그 복원용)
y_test = strat_test_set["burned_area"].copy().to_numpy()

# 2-5. 모델 정의
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])

# 2-6. 컴파일
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=1e-3))

# 2-7. 콜백 설정
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 2-8. 학습
history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping]
)

# 2-9. 학습 곡선 시각화
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.show()

# 2-10. 테스트셋 평가
y_pred_log = model.predict(X_test).flatten()
rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
print(f"테스트 RMSE (로그 스케일): {rmse_log:.4f}")

# 로그 역변환
y_pred = np.expm1(y_pred_log)
rmse_actual = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"테스트 RMSE (실제 스케일): {rmse_actual:.4f}")

# 예시 예측 결과
X_new = X_test[:3]
y_pred_sample = np.expm1(model.predict(X_new).flatten())
print("\n샘플 예측 결과 (헥타르 단위):", np.round(y_pred_sample, 2))

# 2-11. 모델 저장
model.save("best_fire_model.keras")
print("모델이 'best_fire_model.keras'로 저장되었습니다.")