FROM python:3.10.9

WORKDIR /app

COPY . .

RUN pip --no-cache-dir install -r requirements.txt

ENV PYTHONPATH="$PYTHONPATH:/app"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]