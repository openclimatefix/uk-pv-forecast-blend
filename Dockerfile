FROM python:3.12-slim

ARG TESTING=0

# make sure it doesnt fail if the docker file doesnt know the git commit
ARG GIT_PYTHON_REFRESH=quiet

# copy files
COPY readme.md app/readme.md
COPY requirements.txt app/requirements.txt

# install requirements
RUN pip install -r app/requirements.txt

# copy library files
COPY forecast_blend/ app/forecast_blend/
COPY tests/ app/tests/

# change to app folder
WORKDIR /app

# install library
RUN export PYTHONPATH=${PYTHONPATH}:./forecast_blend

RUN if [ "$TESTING" = 1 ]; then pip install pytest pytest-cov coverage; fi

# run migrations using this file
CMD ["python", "-u", "forecast_blend/app.py"]
