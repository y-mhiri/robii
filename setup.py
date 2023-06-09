import setuptools 



setuptools.setup(  
    name='robii',
    version='0.1',
    description='robii : ROBust Interferometric Imaging',
    url='y-mhiri.github.io',
    author='y-mhiri',
    install_requires=['numpy==1.23.4', 
                     'scipy==1.9.3', 
                     'scikit-image',
                     'ducc0==0.27.0',
                     'zarr==2.14.2',
                     'click==8.0.0',
                     'astropy==5.1',
                     'torch==1.13.1',
                     'torchvision==0.14.1',
                     'matplotlib==3.6.2',
                     'numcodecs==0.11.0',
                     'python-casacore==3.5.1',
                     'codex-africanus==0.3.4',
                    #  'nifty-gridder==0.0.1'
                     'omegaconf==2.3.0',
                     'pandas==1.5.3',
                     'reproject==0.10.0',
                     'opencv-python==4.7.0.72'],
    entry_points={
            'console_scripts': [
                'robii = robii.commands.make_image:robii',
                'robiinet = robii.commands.make_image:robiinet' ,
                'generate_dataset = robii.commands.generate_dataset:generate_dataset',
                'plot_images = robii.commands.generate_dataset:plot_images',
                'save_telescope = robii.commands.generate_dataset:save_telescope',
                'train_model = robii.commands.deep:train_model',
                'test_model = robii.commands.deep:test_model',
            ]
    },
    author_email='yassine.mhiri@outlook.fr',
    packages=setuptools.find_packages(),
    zip_safe=False
        )

