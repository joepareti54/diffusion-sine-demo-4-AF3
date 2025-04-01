# diffusion-sine-demo-4-AF3

A minimal educational implementation demonstrating core concepts of conditional diffusion models, inspired by AlphaFold 3's approach.

## Overview
This project implements a simple conditional diffusion model that generates sine waves based on binary conditions. It serves as an educational tool to understand the basic principles of diffusion models and conditioning mechanisms.

## Key Features
- 1000-step diffusion process
- Binary conditioning (-1 or 1) for wave orientation
- Pure input data approach
- Visualization of generation process

## Installation
git clone https://github.com/joepareti54/diffusion-sine-demo-4-AF3.git
cd diffusion-sine-demo-4-AF3
pip install -r requirements.txt

## Usage
python src/new_code_learn_conditioning_0003.py

## Output
The program will:

1. Train the model (2000 epochs)
2. Display training progress
3. Generate visualization showing:
	- Input sine wave
	- Generated waves with different conditions
	- Target waves

## Documentation

https://docs.google.com/document/d/1d4EmjJ0d-jlzA7u5pJSucBmIHQVmo9qtPqGQ8y5slSM/edit?usp=sharing
## Relation to AlphaFold 3
While vastly simplified, this implementation demonstrates the core concept of conditional generation used in AlphaFold 3:

- Demo: Binary conditioning for wave orientation
- AF3: Complex conditioning using protein sequence information
