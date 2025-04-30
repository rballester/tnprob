# tnprob

This is a library implementing tensor network algorithms for probabilistic graphical models, with a focus on:
- Sensitivity analysis
- Distance and divergence metrics

`tnprob` makes heavy use of two awesome tensor network manipulation and contraction libraries: [quimb](https://github.com/jcmgray/quimb) and [cotengra](https://github.com/jcmgray/cotengra).

## Installation

```
git clone git@github.com:rballester/tnprob.git
cd tnprob
pip install -r requirements.txt
pip install .
```

## Testing

```
cd tests
pytest .
```
