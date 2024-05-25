import cv2
import os
from pyzbar.pyzbar import decode
import requests
from dotenv import load_dotenv

load_dotenv()

# Directory containing images
image_dir = 'images'

# Your Barcode Lookup API key
barcode_lookup_api_key = os.getenv('BARCODELOOKUP_API_KEY')
google_books_api_key = os.getenv('GOOGLE_API_KEY')
upc_api_key = os.getenv('UPC_API_KEY')

datakik_url = 'https://www.gtinsearch.org/api/items/{barcode}'


def read_barcodes(image):
    barcodes = decode(image)
    barcode_data = []
    for barcode in barcodes:
        barcode_info = barcode.data.decode('utf-8')
        barcode_data.append(barcode_info)
    return barcode_data

def fetch_from_barcode_lookup(barcode):
    response = requests.get(f'https://api.barcodelookup.com/v3/products?barcode={barcode}&formatted=y&key={barcode_lookup_api_key}')
    if response.status_code == 200:
        product_info = response.json()
        if 'products' in product_info and len(product_info['products']) > 0:
            product = product_info['products'][0]
            return {
                'barcode': barcode,
                'product_name': product.get('product_name', 'Unknown'),
                'brand': product.get('brand', 'Unknown'),
                'category': product.get('category', 'Unknown'),
                'source': 'Barcode Lookup'
            }
    return None

def fetch_from_google_books(barcode):
    response = requests.get(f'https://www.googleapis.com/books/v1/volumes?q=isbn:{barcode}&key={google_books_api_key}')
    if response.status_code == 200:
        book_info = response.json()
        if 'items' in book_info and len(book_info['items']) > 0:
            book = book_info['items'][0]['volumeInfo']
            return {
                'barcode': barcode,
                'product_name': book.get('title', 'Unknown'),
                'brand': book.get('authors', ['Unknown'])[0],
                'category': book.get('categories', ['Unknown'])[0],
                'source': 'Google Books'
            }
    return None

def fetch_from_open_food_facts(barcode):
    response = requests.get(f'https://world.openfoodfacts.org/api/v0/product/{barcode}.json')
    if response.status_code == 200:
        product_info = response.json()
        if 'product' in product_info:
            product = product_info['product']
            return {
                'barcode': barcode,
                'product_name': product.get('product_name', 'Unknown'),
                'brand': product.get('brands', 'Unknown'),
                'category': product.get('categories', 'Unknown'),
                'source': 'Open Food Facts'
            }
    return None

def fetch_product_details(barcodes):
    products = []
    for barcode in barcodes:
        product_details = fetch_from_barcode_lookup(barcode)
        if not product_details:
            product_details = fetch_from_google_books(barcode)
        if not product_details:
            product_details = fetch_from_open_food_facts(barcode)
        
        if product_details:
            products.append(product_details)
        else:
            products.append({
                'barcode': barcode,
                'product_name': 'Unknown',
                'brand': 'Unknown',
                'category': 'Unknown',
                'source': 'None'
            })
    return products

def process_images(image_dir):
    barcodes = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            detected_barcodes = read_barcodes(image)
            barcodes.extend(detected_barcodes)
    
    if barcodes:
        product_details = fetch_product_details(barcodes)
        for product in product_details:
            print(f"Barcode: {product['barcode']}, Product Name: {product['product_name']}, Brand: {product['brand']}, Category: {product['category']}, Source: {product['source']}")
    else:
        print("No barcodes found.")

if __name__ == "__main__":
    process_images(image_dir)

