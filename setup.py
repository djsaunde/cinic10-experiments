from setuptools import setup, find_packages

version = '0.1'

setup(
    name='cinic10',
    version=version,
    description='Loading and experimentation with the CINIC-10 natural images dataset.',
    license='MIT',
    url='http://github.com/djsaunde/cinic10',
    author='Daniel Saunders',
    author_email='djsaunde@cs.umass.edu',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'torch>=0.4.1', 'numpy', 'torchvision', 'fastai'
    ],
)
