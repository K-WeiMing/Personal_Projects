FROM python:3.8

WORKDIR /app

COPY ./app app/app

RUN pip install -r app/app/requirements.txt

ENV PYTHONPATH=/app

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["app.app.main:app", "--host", "0.0.0.0", "--port", "80"]
