# Flask API 

## Flask API Framework

## Core Stack

- Python3.6
- Flask
- MySQL


## Quick Start

Clone project and install dependencies
```bash
# $ pip install -r requirements.txt
$ pip3 install -r requirements.txt
```

Start the server
```bash
$ python run.py
```

View the apidocs
```bash
http://localhost:5001/apidocs/
```

## Deployment
```bash
$ docker bulid -t alo-model

$ docker run --name my-alo-model -d -p 5000:80 alo-model
```

## View

localhost:8090

## Plugins
- **flask_restful** - Flask-RESTful provides the building blocks for creating a great REST API.
https://flask-restful.readthedocs.io/en/latest/
- **flasgger** - Easy Swagger UI for your Flask API
https://github.com/rochacbruno/flasgger
- **flask_sqlalchemy** - SQLAlchemy support to Flask
https://github.com/mitsuhiko/flask-sqlalchemy
- **pymysql** - Pure Python MySQL Client
https://github.com/PyMySQL/PyMySQL
- **marshmallow** - A lightweight library for converting complex objects to and from simple Python datatypes.
https://github.com/marshmallow-code/marshmallow
- **pika** - Pure Python RabbitMQ/AMQP 0-9-1 client library
https://github.com/pika/pika



flask-pika

supervisor 守护进程