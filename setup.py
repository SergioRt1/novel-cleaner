from setuptools import setup, find_packages

setup(
    name='ml_processor',  
    version='0.1',  
    packages=find_packages(),
    install_requires=[
        'datasets', 
        'scikit-learn',  
        'torch',
        'transformers',
    ],
    author='Sergio Rodriguez',
    description='A package to detect non-content sentences in novels using machine learning',
    url='https://github.com/SergioRt1/Webnovel-Scanner', ## TODO update path
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: MIT License',
    ],
    python_requires='>=3.8',
)
