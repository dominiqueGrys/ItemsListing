import cv2
import os
from pyzbar.pyzbar import decode
import aiohttp
import asyncio
import requests
from dotenv import load_dotenv
import async_timeout
import time
import torch
import torchvision.transforms as transforms
from torchvision import models

# Load environment variables from .env file
load_dotenv()

# Directory containing images
image_dir = 'images'

# API keys
api_keys = {
    'barcode_lookup': os.getenv('BARCODELOOKUP_API_KEY'),
    'google_books': os.getenv('GOOGLE_API_KEY'),
    'upc': os.getenv('UPC_API_KEY')
}

# API endpoints
api_endpoints = {
    'barcode_lookup': 'https://api.barcodelookup.com/v3/products?barcode={barcode}&formatted=y&key={api_key}',
    'google_books': 'https://www.googleapis.com/books/v1/volumes?q=isbn:{barcode}&key={api_key}',
    'open_food_facts': 'https://world.openfoodfacts.org/api/v3/product/{barcode}.json',
    'upc': 'https://api.upcitemdb.com/prod/trial/lookup?upc={barcode}',
    'datakik': 'https://www.gtinsearch.org/api/items/{barcode}'
}

api_endpoints = {
    #'barcode_lookup': 'https://api.barcodelookup.com/v3/products?barcode={barcode}&formatted=y&key={api_key}',
    #'google_books': 'https://www.googleapis.com/books/v1/volumes?q=isbn:{barcode}&key={api_key}',
    #'open_food_facts': 'https://world.openfoodfacts.org/api/v3/product/{barcode}.json',
    #'upc': 'https://api.upcitemdb.com/prod/trial/lookup?upc={barcode}',
    #'datakik': 'https://www.gtinsearch.org/api/items/{barcode}'
}

# Load the pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# ImageNet class labels
imagenet_labels = requests.get("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json").json()

def classify_image(image_path):
    input_image = cv2.imread(image_path)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(input_image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
    _, predicted = outputs.max(1)
    return imagenet_labels[predicted.item()]



def read_barcodes(image):
    barcodes = decode(image)
    barcode_data = []
    for barcode in barcodes:
        barcode_info = barcode.data.decode('utf-8')
        barcode_data.append(barcode_info)
    return barcode_data

async def fetch(session, url, retries=3):
    for i in range(retries):
        try:
            async with async_timeout.timeout(30):
                async with session.get(url) as response:
                    if response.status == 200 and response.content_type == 'application/json':
                        return await response.json()
                    else:
                        text = await response.text()
                        print(f"Failed to fetch data from {url}. Status: {response.status}, Content: {text}")
                        return None
        except (aiohttp.ClientError, aiohttp.client_exceptions.ClientConnectorError) as e:
            print(f"Attempt {i+1} failed with error: {e}")
            if i < retries - 1:
                await asyncio.sleep(2 ** i)  # Exponential backoff
            else:
                print(f"Max retries reached for {url}")
                return None

async def fetch_product_details(barcodes):
    products = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for barcode in barcodes:
            for service, url in api_endpoints.items():
                api_key = api_keys.get(service)
                if api_key:
                    request_url = url.format(barcode=barcode, api_key=api_key)
                else:
                    request_url = url.format(barcode=barcode)
                tasks.append(fetch(session, request_url))
        
        responses = await asyncio.gather(*tasks)
        
        for i, barcode in enumerate(barcodes):
            product_details = None
            for j, service in enumerate(api_endpoints.keys()):
                product_info = responses[i * len(api_endpoints) + j]
                
                if product_info:
                    if service == 'barcode_lookup' and 'products' in product_info and len(product_info['products']) > 0:
                        product = product_info['products'][0]
                        product_details = {
                            'barcode': barcode,
                            'product_name': product.get('product_name', 'Unknown'),
                            'brand': product.get('brand', 'Unknown'),
                            'category': product.get('category', 'Unknown'),
                            'source': 'Barcode Lookup'
                        }
                    elif service == 'google_books' and 'items' in product_info and len(product_info['items']) > 0:
                        book = product_info['items'][0]['volumeInfo']
                        product_details = {
                            'barcode': barcode,
                            'product_name': book.get('title', 'Unknown'),
                            'brand': book.get('authors', ['Unknown'])[0],
                            'category': book.get('categories', ['Unknown'])[0],
                            'source': 'Google Books'
                        }
                    elif service == 'open_food_facts' and 'product' in product_info:
                        product = product_info['product']
                        product_details = {
                            'barcode': barcode,
                            'product_name': product.get('product_name', product.get('product_name_en', 'Unknown')),
                            'brand': product.get('brands', 'Unknown'),
                            'category': product.get('categories', 'Unknown'),
                            'source': 'Open Food Facts'
                        }
                    elif service == 'upc' and 'items' in product_info and len(product_info['items']) > 0:
                        product = product_info['items'][0]
                        product_details = {
                            'barcode': barcode,
                            'product_name': product.get('title', 'Unknown'),
                            'brand': product.get('brand', 'Unknown'),
                            'category': product.get('category', 'Unknown'),
                            'source': 'UPC'
                        }
                    elif service == 'datakik' and 'products' in product_info and len(product_info['products']) > 0:
                        product = product_info['products'][0]
                        product_details = {
                            'barcode': barcode,
                            'product_name': product.get('product_name', 'Unknown'),
                            'brand': product.get('brand', 'Unknown'),
                            'category': product.get('category', 'Unknown'),
                            'source': 'Datakik'
                        }
                
                if product_details:
                    break
            
            if not product_details:
                product_details = {
                    'barcode': barcode,
                    'product_name': 'Unknown',
                    'brand': 'Unknown',
                    'category': 'Unknown',
                    'source': 'None'
                }
            
            products.append(product_details)
    
    return products

def process_images(image_dir):
    barcodes = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            detected_barcodes = read_barcodes(image)
            barcodes.extend(detected_barcodes)
            
            # Classify the image using ImageNet
            image_class = classify_image(image_path)
            print(f"Image: {filename}, Classified as: {image_class}")
    
    if barcodes:
        loop = asyncio.get_event_loop()
        product_details = loop.run_until_complete(fetch_product_details(barcodes))
        for product in product_details:
            print(f"Barcode: {product['barcode']}, Product Name: {product['product_name']}, Brand: {product['brand']}, Category: {product['category']}, Source: {product['source']}")
    else:
        print("No barcodes found.")

if __name__ == "__main__":
    process_images(image_dir)
