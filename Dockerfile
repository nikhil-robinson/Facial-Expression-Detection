FROM python:3

ADD facialExpression.py /

RUN pip install -r requirements.txt

CMD [ "python", "./facialExpression.py" ]
