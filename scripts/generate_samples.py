import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.math_dataset import MathDataset
import tqdm

def generate_samples(output_file, count=1000):
    print(f"Generating {count} samples with max_digits=8...")
    dataset = MathDataset(max_digits=8)
    
    with open(output_file, 'w') as f:
        for _ in tqdm.tqdm(range(count)):
            # Direct access to the generation logic for pure string output
            # This ensures we get exactly what the model trains on
            completion = dataset._generate_problem()
            f.write(completion + '\n')
            
    print(f"Saved {count} examples to {output_file}")

if __name__ == "__main__":
    generate_samples("math_examples_8digits.txt")
