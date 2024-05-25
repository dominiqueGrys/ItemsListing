import os
import re
import shutil
from PIL import Image
import pytesseract
import pandas as pd

# Directory containing receipt images
image_dir = 'receipts/'

# CSV file to store the extracted data
csv_file = 'receipts_data.csv'

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def prompt_for_store_name(lines):
    print("Please select the receipt title:")
    for i, line in enumerate(lines[:5], 1):
        print(f"{i}. {line}")
    
    while True:
        try:
            choice = int(input("Enter the number of the line to use as the store name: "))
            if 1 <= choice <= len(lines[:5]):
                return lines[choice - 1]
            else:
                print("Invalid choice, please select a number from the list.")
        except ValueError:
            print("Invalid input, please enter a number.")

def prompt_for_missing_values(receipt_data):
    for key, value in receipt_data.items():
        if value == 'N/A':
            receipt_data[key] = input(f"{key} is missing. Please enter the {key}: ")
    return receipt_data

def parse_receipt(text):
    # Regular expressions to find the date, store name, total price, receipt number, and payment method
    date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
    total_pattern = r'\bTotal\b.*?(\d+\.\d{2})'
    receipt_no_pattern = r'\bReceipt\s*No[:\s]*(\d+)\b'
    payment_method_pattern = r'\b(mastercard|visa|cardholder|credit card|debit card|cash)\b'

    # Find the date
    date_match = re.search(date_pattern, text)
    date = date_match.group(0) if date_match else 'N/A'

    # Split text into lines and prompt user to select the store name
    lines = text.split('\n')
    store_name = prompt_for_store_name(lines) if lines else 'N/A'

    # Find the total price
    total_match = re.search(total_pattern, text, re.IGNORECASE)
    total = total_match.group(1) if total_match else 'N/A'

    # Find the receipt number
    receipt_no_match = re.search(receipt_no_pattern, text, re.IGNORECASE)
    receipt_no = receipt_no_match.group(1) if receipt_no_match else 'N/A'

    # Determine payment method
    payment_method_match = re.search(payment_method_pattern, text, re.IGNORECASE)
    payment_method = payment_method_match.group(0).capitalize() if payment_method_match else 'Unknown'
    if payment_method!='Cash' and payment_method!='Unknown':
        payment_method='Card'
        
    receipt_data = {
        'date': date,
        'store_name': store_name,
        'total': total,
        'receipt_no': receipt_no,
        'payment_method': payment_method
    }

    # Check for missing values and prompt user for manual entry
    if any(value == 'N/A' for value in receipt_data.values()):
        print("Some fields are missing in the extracted data:")
        for key, value in receipt_data.items():
            print(f"{key}: {value}")
        receipt_data = prompt_for_missing_values(receipt_data)

    return receipt_data

def process_receipts(image_dir):
    receipts_data = []
    image_paths = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(image_dir, filename)
            text = extract_text_from_image(image_path)
            receipt_data = parse_receipt(text)
            receipts_data.append(receipt_data)
            image_paths.append(image_path)
            print(f"Processed: {filename}")

    return receipts_data, image_paths

def move_processed_images(image_paths, done_dir):
    os.makedirs(done_dir, exist_ok=True)
    for image_path in image_paths:
        shutil.move(image_path, os.path.join(done_dir, os.path.basename(image_path)))
        print(f"Moved: {os.path.basename(image_path)} to {done_dir}")

def save_to_csv(data, csv_file):
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    receipts_data, image_paths = process_receipts(image_dir)
    if input(f'{receipts_data}\nOK? (y/n)') == 'y':
        save_to_csv(receipts_data, csv_file)
        move_processed_images(image_paths, os.path.join(image_dir, 'Done'))
        print(f"Data saved to {csv_file} and images moved to Done folder")
    else:
        print("Data not saved, images not moved.")

