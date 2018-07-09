#!/bin/bash
# (sudo ./uninstall.sh)
# (cd fem && f2py -m fortran_module --overwrite-signature -h fortran_module.pyf fortran_module.f90)
# (sudo pip install -e .)
# (python increment_version_number.py)
# (python setup.py sdist)
# (twine upload dist/*) # joepatmckenna M@ybe1day
(cd ./fem && sphinx-apidoc -o ../doc -f .)
(cd ./doc && make html)
(git add -A . && cat version | xargs git commit -m  && git push origin master) # joepatmckenna maybe1day
(git subtree push --prefix doc/build/html origin gh-pages) # joepatmckenna maybe1day
# (rsync -azv doc/build/html/ mckennajp@lbm.niddk.nih.gov:/var/www/html/mckennajp/fem/) # Eat@peach
