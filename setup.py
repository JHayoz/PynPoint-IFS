from distutils.core import setup

setup(
    name='pynpoint-ifs',
    version='0.1.0',
    packages=['pynpoint-ifs'],
    url=''https://github.com/JHayoz/PynPoint-IFS',
    license='MIT',
    author='Jean Hayoz',
    author_email='jeanhayoz94@gmail.com',
    description='Package for IFS cube reduction',
    install_requires=[
                'numpy',
                'scipy',
                'matplotlib',
                'astropy',
                'PyAstronomy',
                'pandas',
                'sklearn',
                'seaborn',
                'spectres',
                'photutils',
                'skycalc_ipy'],
)
