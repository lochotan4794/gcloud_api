FROM python:3.10-slim

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app

VOLUME /app/static_files

ENV PORT 5000

WORKDIR $APP_HOME

COPY . ./

RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install gunicorn

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app


