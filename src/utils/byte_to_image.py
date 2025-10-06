#!/usr/bin/env python3
"""
Utility script to convert base64 byte strings to images
"""

import base64
import sys
import os
from io import BytesIO
from PIL import Image

def convert_byte_string_to_image(byte_string, output_path):
    """
    Convert a base64 byte string to an image file
    
    Args:
        byte_string (str): Base64 encoded image data
        output_path (str): Path to save the image
    """
    try:
        # Remove any whitespace
        byte_string = byte_string.strip()
        
        # Decode base64
        image_data = base64.b64decode(byte_string)
        image = Image.open(BytesIO(image_data))
        
        # Save image
        image.save(output_path)
        print(f"Image successfully saved to {output_path}")
        
    except Exception as e:
        print(f"Error converting byte string to image: {str(e)}")
        sys.exit(1)

def convert_file_to_image(input_file, output_path):
    """
    Convert a file containing base64 byte string to an image file
    
    Args:
        input_file (str): Path to file containing base64 data
        output_path (str): Path to save the image
    """
    try:
        # Read byte string from file
        with open(input_file, 'r') as f:
            byte_string = f.read()
        
        # Convert to image
        convert_byte_string_to_image(byte_string, output_path)
        
    except Exception as e:
        print(f"Error reading file or converting to image: {str(e)}")
        sys.exit(1)

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python byte_to_image.py <byte_string> <output_path>")
        print("  python byte_to_image.py -f <input_file> <output_path>")
        print("\nExamples:")
        print("  python byte_to_image.py 'base64data...' output.jpg")
        print("  python byte_to_image.py -f byte_data.txt output.jpg")
        sys.exit(1)
    
    if sys.argv[1] == "-f":
        # Convert from file
        if len(sys.argv) != 4:
            print("Error: Missing arguments for file conversion")
            sys.exit(1)
        
        input_file = sys.argv[2]
        output_path = sys.argv[3]
        
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} not found")
            sys.exit(1)
        
        convert_file_to_image(input_file, output_path)
    else:
        # Convert from command line argument
        if len(sys.argv) != 3:
            print("Error: Missing output path")
            sys.exit(1)
        
        byte_string = sys.argv[1]
        output_path = sys.argv[2]
        
        convert_byte_string_to_image(byte_string, output_path)

if __name__ == "__main__":
    main()