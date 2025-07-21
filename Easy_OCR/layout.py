from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch
import json

# Load the LayoutLMv3 processor and model
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# Load the image (your invoice image)
image_path = 'testocr3.jpg'  # Update this with your image path
image = Image.open(image_path)

# Encode the image and prepare it for the model
print("Image encoding started...")
encoding = processor(image, return_tensors="pt")
print("Image encoding completed.")

# Perform inference
print("Starting model inference...")
with torch.no_grad():
    outputs = model(**encoding)
print("Model inference completed.")

# Get the predicted labels for the tokens
predictions = outputs.logits.argmax(-1).squeeze().tolist()

# Extract the predicted tokens and corresponding bounding boxes
tokens = processor.tokenizer.convert_ids_to_tokens(encoding.input_ids.squeeze().tolist())
bboxes = encoding.bbox.squeeze().tolist()

# Process the result to find the invoice number and other fields
fields = {
    "invoice_number": "",
    "invoice_date": "",
    "total_amount": ""
}

# Example of extracting information by searching for specific keywords or token labels
for token, prediction, bbox in zip(tokens, predictions, bboxes):
    if "INV" in token.upper():
        fields["invoice_number"] = token  # Add logic to get full invoice number
    elif "DATE" in token.upper():
        fields["invoice_date"] = token
    elif "TOTAL" in token.upper():
        fields["total_amount"] = token

# Output in JSON format
output_data = {
    "invoice_number": fields["invoice_number"],
    "invoice_date": fields["invoice_date"],
    "total_amount": fields["total_amount"],
    "bounding_boxes": bboxes
}

# Print the extracted fields as JSON
print(json.dumps(output_data, indent=4))
