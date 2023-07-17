FROM python:3.7
RUN pip install musicnn
ADD musicnn_tags.py /
WORKDIR ./
EXPOSE 3333
ENTRYPOINT ["python", "./musicnn_tags.py"]