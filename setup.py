#!/usr/bin/env python

from distutils.core import setup

setup(name='NeuroGLM',
	version='1.0',
	description='GLM fitting',
	author='David Fox',
	author_email='foxy1987@gmail.com',
	url='https://github.com/Foxy1987/neuroGLM',
	packages=['glmtools', 'utils'],
	install_requires=['numpy >= 1.19', 'scipy >= 1.5.1', 'matplotlib >= 3.3',
						'scikit-learn >= 0.23.1'],
	python_requires='>=3.8')










