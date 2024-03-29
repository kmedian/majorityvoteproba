from setuptools import setup


def read(fname):
    import os
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='majorityvoteproba',
      version='0.1.0',
      description='merge probability predictions for majority votes',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      url='http://github.com/kmedian/majorityvoteproba',
      author='Ulf Hamster',
      author_email='554c46@gmail.com',
      license='MIT',
      packages=['majorityvoteproba'],
      install_requires=[
          'setuptools>=40.0.0',
          'nose>=1.3.7',
          'numpy>=1.17.1'],
      python_requires='>=3.6',
      zip_safe=False)
