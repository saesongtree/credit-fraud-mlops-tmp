FROM python:3.11-slim

# Create app directory
WORKDIR /app

# Install app dependencies
COPY docker_mlflow/requirements.txt .

RUN pip install -r requirements.txt

# Bundle app source
COPY . .

EXPOSE 5000

CMD [ "mlflow", "server","--host","0.0.0.0","--port","5000"]