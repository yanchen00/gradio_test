from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import gradio as gr
import requests
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
ver = torch.backends.cudnn.version()
back = torch.backends.cudnn.enabled
print(device, back)

# from google.colab import drive
# drive.mount('/content/gdrive')

processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50", revision="no_timm")


def gradio_prediction_procress(image):
    image = Image.fromarray(image)
    target_sizes = torch.tensor([image.size[::-1]])
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    predictions = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.75)[0]
    readable_labels = [category_map[int(category_id)]
                       for category_id in predictions['labels']]
    draw = ImageDraw.Draw(image)

    for box in predictions['boxes']:
        box = box.detach().numpy()  # Convert to NumPy array
        draw.rectangle(((box[0], box[1]), (box[2], box[3])),
                       outline="red", width=0.5)

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    return readable_labels, image


demo = gr.Interface(
    fn=gradio_prediction_procress,
    inputs="image",
    outputs=["text", "image"]
)

demo.launch(server_name="0.0.0.0")
