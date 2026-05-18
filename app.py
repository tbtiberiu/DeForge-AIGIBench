import gradio as gr
from detector_codes import DEVICE, detector_classes, weight_mapping

# Model cache to avoid reloading
model_cache = {'name': None, 'instance': None}


def predict(model_name, input_image):
    if input_image is None:
        return 'Please upload an image.'

    global model_cache

    # Load model if not in cache or if model changed
    if model_cache['name'] != model_name:
        print(f'Loading model: {model_name}...')
        try:
            detector_class = detector_classes[model_name]
            weights = weight_mapping[model_name]
            model_cache['instance'] = detector_class(weights)
            model_cache['name'] = model_name
        except Exception as e:
            return f'Error loading model {model_name}: {str(e)}'

    detector = model_cache['instance']

    # Preprocess image
    try:
        img_tensor = detector.transform(input_image).unsqueeze(0).to(DEVICE)

        # Inference
        p_fake = detector.detect(img_tensor).item()
        p_real = 1.0 - p_fake

        return {'Real Image': p_real, 'AI Generated': p_fake}
    except Exception as e:
        return f'Error during inference: {str(e)}'


# Define the Gradio interface
with gr.Blocks(title='AIGI Detector Bench') as demo:
    gr.Markdown('# AIGI Detector Benchmark')
    gr.Markdown(
        "Select a model and upload an image to check if it's AI-generated or real."
    )

    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=sorted(list(detector_classes.keys())),
                value='DeForge-AI',
                label='Select Detection Model',
            )
            input_img = gr.Image(type='pil', label='Upload Image')
            btn = gr.Button('Detect', variant='primary')

        with gr.Column():
            output_label = gr.Label(num_top_classes=2, label='Prediction')

    btn.click(fn=predict, inputs=[model_dropdown, input_img], outputs=output_label)

    gr.Markdown("""
    ### About
    This tool is a **modified version** of the official [AIGIBench](https://github.com/HorizonTEL/AIGIBench) repository, featuring state-of-the-art AI-Generated Image (AIGI) detectors. 
    
    In this version, I have integrated the original baselines along with my own proposed solutions: **DeForge-AI** and **C2P-DINOv2**.

    - **Project Page**: [NeurIPS 2025] Is Artificial Intelligence Generated Image Detection a Solved Problem?
    - **Original Repository**: [HorizonTEL/AIGIBench](https://github.com/HorizonTEL/AIGIBench)

    #### Featured Models:
    - **DeForge-AI**: My proposed multi-modal forensic detector (optimized for diverse generators).
    - **C2P-DINOv2**: My solution leveraging DINOv2 features (intermediary solution).
    - **RIGID, AIDE, SAFE, Effort, NPR, LaDeDa, etc.**: Original SOTA baselines.

    Each model has different strengths. DeForge-AI generally provides the best performance across diverse generators.
    """)

if __name__ == '__main__':
    demo.launch()
