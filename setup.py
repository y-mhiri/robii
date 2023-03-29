import setuptools 



setuptools.setup(  
    name='robii',
    version='0.1',
    description='robii : ROBust Interferometric Imaging',
    url='y-mhiri.github.io',
    author='y-mhiri',
    install_requires=['numpy', 
                     'scipy', 
                     'scikit-image',
                     'ducc0',
                     'zarr',
                     'click',
                     'astropy',
                     'ducc0',
                     'torch',
                     'torchvision',
                     'matplotlib',
                     'numcodecs',
                     'python-casacore',
                     'flask-caching',
                     'omegaconf',
                     'dash',
                     'opencv-python==4.7.0.72'],
    entry_points={
            'console_scripts': [
                'robii = robii.commands.make_image:robii',
                'robiinet = robii.commands.make_image:robiinet' ,
                'generate_dataset = robii.commands.generate_dataset:generate_dataset',
                'save_telescope = robii.commands.generate_dataset:save_telescope',
                'train_model = robii.commands.deep:train_model',
                'test_model = robii.commands.deep:test_model',
            ]
    },
    author_email='yassine.mhiri@outlook.fr',
    packages=setuptools.find_packages(),
    zip_safe=False
        )

