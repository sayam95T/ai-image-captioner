from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gradio as gr

# Load model + processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Gradio UI
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="AI Image Caption Generator",
    description="Upload an image and get an AI-generated caption."
)

if __name__ == "__main__":
    demo.launch()
