### README

# Receipt and Barcode Processing Project

## Description

This project provides tools for processing images of receipts and items with barcodes. It includes functionalities for extracting and parsing text from images, classifying items using a pre-trained ImageNet model, and generating descriptions using a language model. The project also features a receipt processing script to extract relevant data and save it to a CSV file.

## Features

- **Barcode Lookup**: Extract and lookup product details using barcodes.
- **Image Classification**: Classify items using a pre-trained ImageNet model.
- **Language Model Descriptions**: Generate descriptions of images using a language model.
- **Receipt Processing**: Extract and save receipt data (store name, date, total price, etc.) to a CSV file.

## Installation

To run this project, you need to install the required libraries. You can do this using `pip`:

```bash
pip install pillow pytesseract pandas opencv-python pyzbar aiohttp requests python-dotenv torch torchvision transformers
```

Additionally, you need to have Tesseract OCR installed on your system. You can find installation instructions [here](https://github.com/tesseract-ocr/tesseract).

## Usage

### Barcode and Product Details

1. Extract text from images using `pytesseract`.
2. Parse the text to extract relevant information such as barcodes.
3. Lookup product details using various APIs.

### Image Classification

1. Use a pre-trained ResNet model from `torchvision` to classify items.
2. Load the model and process images to get classification results.

### Language Model Descriptions

1. Use the BLIP-2 model to generate descriptions of images.
2. Load the model and process images to get descriptive text.

### Receipt Processing

1. Extract text from receipt images using `pytesseract`.
2. Parse the extracted text to identify key information (date, store name, total price, etc.).
3. Save the extracted data to a CSV file.
4. Optionally, move processed images to a "Done" folder.

### Example Script

An example script for processing receipts is included in the project. Here’s a brief overview of how to run the script:

1. Place your receipt images in the `receipts/` directory.
2. Run the script:

```python
if __name__ == "__main__":
    receipts_data, image_paths = process_receipts(image_dir)
    if input(f'{receipts_data}\nOK? (y/n)') == 'y':
        save_to_csv(receipts_data, csv_file)
        move_processed_images(image_paths, os.path.join(image_dir, 'Done'))
        print(f"Data saved to {csv_file} and images moved to Done folder")
    else:
        print("Data not saved, images not moved.")
```

## Credits

This project leverages several third-party libraries and models. Special thanks to the following projects and contributors:

- **[barcode_lookup](https://github.com/barcodelookup)**
- **[google_books](https://developers.google.com/books)**
- **[open_food_facts](https://world.openfoodfacts.org/data)**
- **[upc](https://upcitemdb.com)**
- **[datakik](https://datakik.com)**
- **[ImageNet Labels](https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json)**
- **[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models by Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi](https://arxiv.org/abs/2301.12597)**



---

Feel free to reach out for any questions or contributions to the project.
