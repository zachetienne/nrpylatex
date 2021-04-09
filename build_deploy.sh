# update version in setup.py and __init__.py
# run ./build_deploy.sh --test to deploy package to TestPyPI
# run ./build_deploy.sh to deploy package to PyPI

rm -r dist ;
python setup.py sdist bdist_wheel ;
if twine check dist/* ; then
    if [ "$1" = "--test" ] ; then
        twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    else
        twine upload dist/* ;
    fi
fi
