from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_mlp(input_size, params):
    """
    ساخت مدل شبکه عصبی بر اساس پارامترها.
    Args:
        input_size (int): تعداد ویژگی‌های ورودی.
        params (dict): پارامترهای مدل (لایه‌ها، نرون‌ها و ...).
    Returns:
        model: مدل شبکه عصبی.
    """
    model = Sequential()

    # اضافه کردن لایه‌های پنهان
    for _ in range(params["num_layers"]):
        model.add(Dense(params["num_neurons"], activation=params["activation_function"]))

    # اضافه کردن لایه خروجی
    model.add(Dense(1, activation="sigmoid"))

    # کامپایل مدل
    model.compile(
        optimizer=params.get("optimizer", "adam"),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
