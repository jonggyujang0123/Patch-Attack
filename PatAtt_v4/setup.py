from setuptools import find_packages, setup

setup(name='PatchMIA',
        version='0.21',
        description='Unofficial implementation ',
        url='https://github.com/jonggyujang0123/Transformer-NLP',
        author='jonggyujang0123',
        author_email='jgjang0123@gmail.com',
        #  packages=find_packages(),
        packages=  ['models','utils', 'tools', 'datasets' ],
        python_requires = '>=3.7')
