from setuptools import setup, find_packages

setup(
    name='ml_processor',  
    version='1.0',  
    packages=find_packages(),
    install_requires=[
        'datasets==3.0.1', 
        'scikit-learn==1.5.2',  
        'transformers[torch]==4.45.1',
        'accelerate>=0.26.0', 
        'pandas==2.2.3',
        # torch==2.6.0.dev20241005+rocm6.2 this should be manually install with pip install --pre torch==2.6.0.dev20241005+rocm6.2 --index-url https://download.pytorch.org/whl/nightly/rocm6.2 --force-reinstall
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
