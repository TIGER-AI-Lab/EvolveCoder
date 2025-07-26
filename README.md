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
bash scripts/run.sh
```

- to do analysis of previous acecoder dataset
```bash
cd acecoderv2/synthesizer
python ../scripts/format_old_acecoderv2_data
bash scripts/run_old_acecoderv2.sh
```
