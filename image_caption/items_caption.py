from PIL import Image
import os
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch



# Directory containing images
image_dir = 'images/'

def resize_image(image_path, max_size=(500, 500)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.LANCZOS)
        return img

def process_images(image_dir):
    resized_images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(image_dir, filename)
            resized_image = resize_image(image_path)
            resized_images.append(resized_image)
            print(f"Resized and added to list: {filename}")
    return resized_images
    

if __name__ == "__main__":

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    

    resized_images = process_images(image_dir)
    # Example usage: Display the first resized image
    if resized_images:
        image = resized_images[0]
        
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        

        generated_ids = model.generate(**inputs)

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)
