import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from algorithms import hybrid_woa_gwo

# بارگذاری دیتاست
iris = load_iris()
X, y = iris.data, iris.target

# تقسیم دیتاست به تست و آموزش
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# استانداردسازی ویژگی‌ها
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# تابع ساخت مدل شبکه عصبی
def build_mlp(input_dim, layers=[10, 10], activation='relu', optimizer='adam'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    for layer_size in layers:
        model.add(tf.keras.layers.Dense(layer_size, activation=activation))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))  # تعداد کلاس‌ها برای iris
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# تابع ارزیابی برای محاسبه دقت
def fitness_function(params):
    model = build_mlp(input_dim=X_train.shape[1], **params)
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    return accuracy_score(y_test, y_pred)

# فضای جستجو برای بهینه‌سازی
search_space = {
    'layers': [(5, 10), (5, 10)],  # تعداد نورون‌ها در هر لایه
    'activation': ['relu', 'sigmoid'],
    'optimizer': ['adam', 'rmsprop'],
}

# اجرای الگوریتم ترکیبی برای یافتن بهترین پارامترها
best_params = hybrid_woa_gwo(search_space, fitness_function, num_individuals=10, iterations=5)

print("بهترین پارامترها: ", best_params)
