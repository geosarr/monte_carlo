install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint: 
	pylint --disable=C monte_carlo tests utils
	# pylint --disable=R,C,E1101,W1309,E0611 monte_carlo tests utils

test: 
	pytest tests.py 

all: install format lint test
