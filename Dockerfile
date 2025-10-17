FROM python:3.10-slim

WORKDIR /workspace

COPY . /workspace

RUN pip install --upgrade pip && pip install -e .

RUN chmod +x acecoderv3_fine_grained_test_cases/step1.sh

CMD ["bash", "acecoderv3_fine_grained_test_cases/step1.sh"]