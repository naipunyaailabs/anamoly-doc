#!/usr/bin/env python3
"""
Excel Exporter for Crowd Anomaly Detection
Exports anomaly data to Excel with time, anomaly, and screenshot columns
"""

import pandas as pd
import base64
from io import BytesIO
from PIL import Image
import os

def create_anomaly_excel(data, output_file="anomaly_report.xlsx"):
    """
    Create an Excel file with time, anomaly, and screenshot columns
    
    Args:
        data (list): List of dictionaries with 'time', 'anomalies', and 'screenshot_data' keys
        output_file (str): Name of the output Excel file
    
    Returns:
        str: Path to the created Excel file
    """
    # Prepare data for Excel
    time_data = []
    anomaly_data = []
    screenshot_data = []
    
    for item in data:
        # Time column (now includes video time)
        time_data.append(str(item.get('time', '')))
        
        # Anomaly column
        anomalies = item.get('anomalies', [])
        anomaly_desc = ', '.join(anomalies) if anomalies else 'None'
        anomaly_data.append(anomaly_desc)
        
        # Screenshot data (base64 string)
        screenshot_data.append(item.get('screenshot_data', ''))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time': time_data,
        'Anomaly': anomaly_data,
        'Screenshot_Bytecode': screenshot_data
    })
    
    # Create Excel file
    df.to_excel(output_file, index=False, sheet_name='Anomalies')
    
    return output_file

def extract_images_from_excel(excel_file, output_dir="extracted_images"):
    """
    Extract images from the Excel file's screenshot column
    
    Args:
        excel_file (str): Path to the Excel file
        output_dir (str): Directory to save extracted images
    
    Returns:
        list: List of paths to extracted images
    """
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Create directory for images
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_images = []
    
    # Convert each base64 string to image
    for index, row in df.iterrows():
        try:
            # Decode base64
            if row['Screenshot_Bytecode']:
                image_data = base64.b64decode(row['Screenshot_Bytecode'])
                image = Image.open(BytesIO(image_data))
                
                # Save image
                image_path = os.path.join(output_dir, f"image_{index}.jpg")
                image.save(image_path)
                extracted_images.append(image_path)
                print(f"Saved {image_path}")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    
    return extracted_images

def byte_string_to_image(byte_string, output_path=None):
    """
    Convert a base64 byte string to an image file
    
    Args:
        byte_string (str): Base64 encoded image data
        output_path (str): Path to save the image (optional)
    
    Returns:
        PIL.Image: The decoded image
    """
    try:
        # Remove any whitespace
        byte_string = byte_string.strip()
        
        # Decode base64
        image_data = base64.b64decode(byte_string)
        image = Image.open(BytesIO(image_data))
        
        # Save image if output path is provided
        if output_path:
            image.save(output_path)
            print(f"Image saved to {output_path}")
        
        return image
    except Exception as e:
        raise Exception(f"Error decoding image: {str(e)}")

def byte_strings_to_images(byte_strings, output_dir="converted_images"):
    """
    Convert multiple base64 byte strings to image files
    
    Args:
        byte_strings (list): List of base64 encoded image data strings
        output_dir (str): Directory to save the images
    
    Returns:
        list: List of paths to converted images
    """
    # Create directory for images
    os.makedirs(output_dir, exist_ok=True)
    
    converted_images = []
    
    # Convert each byte string to image
    for i, byte_string in enumerate(byte_strings):
        try:
            if byte_string:
                image = byte_string_to_image(byte_string)
                image_path = os.path.join(output_dir, f"converted_image_{i}.jpg")
                image.save(image_path)
                converted_images.append(image_path)
                print(f"Converted image saved to {image_path}")
        except Exception as e:
            print(f"Error converting byte string {i}: {e}")
    
    return converted_images

def main():
    """Example usage of the Excel exporter"""
    # Sample data
    sample_data = [
        {
            "time": "00:01:25",  # Video time format
            "anomalies": ["Standing", "Phone Usage"],
            "screenshot_data": ""
        },
        {
            "time": "00:02:10",  # Video time format
            "anomalies": ["Empty Chair"],
            "screenshot_data": ""
        }
    ]
    
    # Create Excel file
    excel_file = create_anomaly_excel(sample_data, "sample_anomaly_report.xlsx")
    print(f"Created Excel file: {excel_file}")
    
    # Extract images (if any)
    # extracted_images = extract_images_from_excel(excel_file)
    # print(f"Extracted {len(extracted_images)} images")

if __name__ == "__main__":
    main()