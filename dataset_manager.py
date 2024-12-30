import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetManager:
    def __init__(self, file_path, target_column, test_split=0.2):
        """
        مدیریت دیتاست و تقسیم داده‌ها برای آموزش و تست.
        Args:
            file_path (str): مسیر فایل دیتاست.
            target_column (str): نام ستون لیبل (خروجی).
            test_split (float): درصد داده تست (پیش‌فرض: 20٪).
        """
        self.file_path = file_path
        self.target_column = target_column
        self.test_split = test_split

        self.X = None
        self.y = None
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_and_split()

    def _load_dataset(self):
        """بارگذاری فایل CSV یا Excel."""
        if str(self.file_path).endswith(".csv"):
            return pd.read_csv(self.file_path)
        elif str(self.file_path).endswith(".xlsx"):
            return pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")

    def _load_and_split(self):
        """بارگذاری و جداسازی ویژگی‌ها و لیبل‌ها + تقسیم‌بندی به Train و Test."""
        dataset = self._load_dataset()

        if self.target_column not in dataset.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")

        X = dataset.drop(columns=[self.target_column])
        y = dataset[self.target_column]

        return train_test_split(X, y, test_size=self.test_split, random_state=42)

    def get_splits(self):
        """بازگرداندن داده‌های تقسیم‌شده."""
        return self.X_train, self.X_test, self.y_train, self.y_test
