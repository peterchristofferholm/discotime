Installation
============

Discotime supports Python 3.10 or higher.

We recommend to install via pip:

.. code-block:: bash

    $ pip install discotime

You can also install the development version of discotime 
from the master branch of Git repository:

.. code-block:: bash

    $ pip install git+https://github.com/peterchristofferholm/discotime.git


Dependencies
------------

Dependencies are usually handled automatically by ``pip``, 
but see the ``setup.cfg`` on github for details.
The following packages should cover the essentials:

- torch>=1.13.0,<1.14
- numpy>=1.21.0,<1.26
- lightning>=2.0,<2.1
- scikit-learn>=1.2.0,<2.0
- einops>=0.6.0,<1.0.0
