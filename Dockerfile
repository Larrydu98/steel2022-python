# FROM 219.216.80.32:8000/python-3.7.2:requirementsAdd
FROM 202.118.21.236:5000/basic-python-image
# docker build -t pythontest .
WORKDIR /usr/src/app

COPY . .
# COPY ./config.txt /usr/src/app/config.txt

EXPOSE 5000

CMD ["gunicorn", "-c", "gunicorn.conf", "run:app"]
