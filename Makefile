test:
	python setup.py test

publish:
	pip install twine
	python setup.py sdist bdist_wheel
	twine upload dist/*
	rm -rf build dist .egg suspect.egg-info
