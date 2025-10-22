FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN pip install --upgrade pip && pip install -e .

RUN chmod +x acecoderv3_fine_grained_test_cases/step3.sh

CMD ["bash", "acecoderv3_fine_grained_test_cases/step3.sh"]