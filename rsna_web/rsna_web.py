# %%
import gradio as gr
import PIL.Image as Image

from ultralytics import  YOLO
import pandas as pd

# %%
model = YOLO('weights/best-correctLabel.pt')
model.to("cuda:1")

# %%
def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    predictions = {}
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        # get table of prediction
        #r.box
        for pred_class, conf in zip(r.boxes.cls, r.boxes.conf):
            pred_class = pred_class.item()
            conf = conf.item()
            _class = model.names[pred_class]
            predictions[_class] = conf

   # pT = pd.DataFrame(predictions)
   # pT.index.name = "Position"
    return im, pd.DataFrame.from_dict(predictions, orient="index", columns=["Values"])\
        .rename_axis("Name").reset_index()

# %%
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=[gr.Image(type="pil", label="Result"),gr.Dataframe() ],
    title="RSNA 척축",
    description="이미지 업로드.",
    examples=[
        [ "assets/2383459912_969865975_10.jpg", 0.25, 0.45],
        [ "assets/732899790_4144762857_11.jpg", 0.25, 0.45],
        [ "assets/74294498_507422603_9.jpg", 0.25, 0.45],
    ],
)

# %%
iface.launch(allowed_paths=['assets'])  # server_port=1201
