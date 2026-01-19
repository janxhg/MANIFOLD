# Data Directory

Training datasets for MANIFOLD.

## Files

- `math_10k.txt` - 10K mathematical expressions (addition/subtraction/multiplication)
- `math_examples_8digits.txt` - Example problems with 8-digit operands

## Usage

```python
from gfn.math_dataset import MathDataset

dataset = MathDataset(file_path='data/math_10k.txt', max_digits=2)
```

## Format

Each line is a problem in the format:
```
operand1 operator operand2 = result
```

Example:
```
12+34=46
99-23=76
5*7=35
```
