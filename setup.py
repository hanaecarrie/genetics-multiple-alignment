from setuptools import setup

descr =  """
'genetics-multiple-alignment' is a package related to the multiple alignment 
of DNA sequences with the star alignment method.
This was developed in the context of a bio-informatics project at CentraleSupélec. 
"""

long_descr = """
'genetics-multiple-alignment' is a package related to the multiple alignment 
of DNA sequences with the star alignment method.
This was developed in the context of a bio-informatics course project at CentraleSupélec. 
This package mainly provides: utilitarian functions to insert gap into DNA sequences,
a function to align 2 sequences together with the Needleman and Wunsch method
and the multiple alignment function with the star method.
"""

setup(name='genetics-multiple-alignement',
      version='0.0.0',
      description=descr,
      long_description=long_descr,
      classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'],
      keywords='scientific studies',
      url='https://github.com/CherkaouiHamza/meri',
      packages=['meri'],
      install_requires=[
        'numpy>=1.14.5',
        'more-itertools>=4.3.0',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      zip_safe=False)
