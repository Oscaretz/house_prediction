# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /code

ENV FLASK_APP=main.py

ENV FLASK_RUN_HOST=0.0.0.0

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 5000

COPY . .
# Run flask when the container launches
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]