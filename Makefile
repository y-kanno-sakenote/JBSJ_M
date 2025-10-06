run:
	streamlit run app.py

    install:
	pip install -r requirements.txt

    check:
	ruff check . && mypy .

    format:
	ruff format .
