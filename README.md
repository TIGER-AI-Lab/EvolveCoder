# AceCoderV2


## Installation
```bash
uv sync
uv pip install -e .
```

## Evaluation
`
```bash
mkdir -p eval
cd eval
git clone -b reasoning https://github.com/jdf-prog/LiveCodeBench
git clone https://github.com/jdf-prog/AceReasonEvalKit.git
```


## Synthesizer
```bash
cd acecoderv2/synthesizer
bash run.sh
```