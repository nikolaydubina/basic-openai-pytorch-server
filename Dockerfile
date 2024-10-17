FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel AS builder

WORKDIR /code

COPY requirements.txt /code
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN opentelemetry-bootstrap -a install

COPY ./main.py /code/

CMD ["opentelemetry-instrument", "python", "/code/main.py"]
