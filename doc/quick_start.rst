###########################
Installation of the package
###########################

The package can be installed via `pip` or directly from the
repository.

Install using pip
-----------------

::

    $ pip install gmd


Install from the repository
===========================

::

    $ git clone https://github.com/scikit-learn-contrib/project-template.git
    $ cd gmd
    $ pip install .


Usage
#####

After the installation the library can be used like every scikit-learn
compatible estimator::

    from gmd import GMD

    gmd = GMD()
    gmd.fit(data)
    print(gmd.subspaces_)
