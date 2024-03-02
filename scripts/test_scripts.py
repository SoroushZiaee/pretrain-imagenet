import sys
import os

# Add the parent directory to the Python path
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
parent_dir = os.path.dirname(script_dir)  # Get the parent directory
sys.path.append(parent_dir)


print("Completed")
from lit_modules import LitLaMemDataModule, LitVGG19

print("Completed")

from lit_modules import LitLaMemDataModule, LitMobileNetV2

print("Completed")

from lit_modules import LitLaMemDataModule, LitInception

print("Completed")

from lit_modules import LitLaMemDataModule, LitGoogleNet

print("Completed")

from lit_modules import LitLaMemDataModule, LitConvNet

print("Completed")
