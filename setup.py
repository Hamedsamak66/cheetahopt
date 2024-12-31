from setuptools import setup, find_packages

setup(
    name='cheetahopt',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Optimization library for to find best MLP parameters using evolutionary algorithms',
    url='https://github.com/yourusername/cheetahopt',  # آدرس گیت‌هاب اگر دارید
    packages=find_packages(),
    install_requires=[
        'pandas',
        'tensorflow',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # یا نوع دیگر لایسنس
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
