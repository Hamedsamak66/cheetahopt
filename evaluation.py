from sklearn.metrics import accuracy_score

def evaluate_model(y_true, y_pred):
    """ارزیابی عملکرد مدل با دقت."""
    return accuracy_score(y_true, y_pred)
