from setuptools import setup, find_packages
setup(
     name='evolutionary_features',
     version='0.1',
     author='Francisco',
     description='Use evolutionary algorithms to feature relevance selection.',
     url='https://github.com/UrbanoFonseca/evolutionary_features',
     classifiers=[ 
            'Development Status :: 1 - Planning',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.3'
            ],
     package_dir={'': '.'},
     packages=find_packages('.'),
     install_requires=[
         'numpy>=1.12.0',
         'scikit-learn>=0.18.0',
         'matplotlib>=2.0.0',
         'pandas>=0.19.2'
         ]    
    )
