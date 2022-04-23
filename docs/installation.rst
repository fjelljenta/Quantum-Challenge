Installation
============

Our code requires the latest Python (3.10) and a virtual environment to decouple it from other environments or installations. Please install python from here `Python <https://www.python.org/>`_. Make sure, that Python is listed in your PATH environment and then proceed creating a virtual environment. Open a command line and type
::

    python -m venv quantimize
    quantimize\Scripts\activate.bat

to create a quantimize environment and to activate the environment.
The package can be cloned from github using 
::

    git clone https://github.com/fjelljenta/Quantum-Challenge
    cd Quantum-Challenge

After that, one needs to install all required packages using
::

    pip install -r requirements.txt

Now, everything should be ready and set up to further run this notebook. Type
::

    jupyter notebook

and open the Quantimize.ipynb.