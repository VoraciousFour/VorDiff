from distutils.core import setup
setup(
  name = 'VorDiff',         
  packages = ['VorDiff'],  
  version = '0.1',     
  license='MIT',        
  description = 'A package for forward and reverse automatic differentiation',  
  author = 'Erik Johnsson',                   
  author_email = 'ejohnsson@college.harvard.edu',     
  url = 'https://github.com/VoraciousFour/VorDiff',   
  download_url = 'https://github.com/VoraciousFour/VorDiff/archive/v_01.tar.gz',    
  keywords = ['automatic', 'differentiation'],   
  install_requires=[      
          'numpy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',     
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)