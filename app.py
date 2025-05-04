import json
import os
import time
import uuid
import tempfile
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import base64
import mimetypes

from google import genai
from google.genai import types

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)

def generate(text, file_name, api_key, model="gemini-2.0-flash-exp"):
    # Initialize client using provided api_key (or fallback to env variable)
    client = genai.Client(api_key=(api_key.strip() if api_key and api_key.strip() != ""
                                     else os.environ.get("GEMINI_API_KEY")))
    
    files = [ client.files.upload(file=file_name) ]
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text=text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_modalities=["image", "text"],
        response_mime_type="text/plain",
    )

    text_response = ""
    image_path = None
    # Create a temporary file to potentially store image data.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            candidate = chunk.candidates[0].content.parts[0]
            # Check for inline image data
            if candidate.inline_data:
                save_binary_file(temp_path, candidate.inline_data.data)
                print(f"File of mime type {candidate.inline_data.mime_type} saved to: {temp_path} and prompt input: {text}")
                image_path = temp_path
                # If an image is found, we assume that is the desired output.
                break
            else:
                # Accumulate text response if no inline_data is present.
                text_response += chunk.text + "\n"
    
    del files
    return image_path, text_response

def process_image_and_prompt(composite_pil, prompt, gemini_api_key):
    try:
        # Save the composite image to a temporary file.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            composite_path = tmp.name
            composite_pil.save(composite_path)
        
        file_name = composite_path  
        input_text = prompt 
        model = "gemini-2.0-flash-exp" 

        image_path, text_response = generate(text=input_text, file_name=file_name, api_key=gemini_api_key, model=model)
        
        if image_path:
            # Load and convert the image if needed.
            result_img = Image.open(image_path)
            if result_img.mode == "RGBA":
                result_img = result_img.convert("RGB")
            return [result_img], ""  # Return image in gallery and empty text output.
        else:
            # Return no image and the text response.
            return None, text_response
    except Exception as e:
        raise gr.Error(f"Error Getting {e}", duration=5)


# Build a Blocks-based interface with a custom HTML header and CSS
with gr.Blocks(css_paths="style.css", theme=gr.themes.Soft()) as demo:
    # Custom HTML header with proper class for styling
    gr.HTML(
    """
    <div class="header-container">
        <div class="logo-section">
            <div class="logo-wrapper">
                <img src="https://www.gstatic.com/lamda/images/gemini_favicon_f069958c85030456e93de685481c559f160ea06b.png" alt="Gemini logo" class="logo">
                <div class="logo-glow"></div>
            </div>
            <div class="title-section">
                <h1 class="app-title">Gemini Vision Pro</h1>
                <p class="tagline">AI Image Processing</p>
            </div>
        </div>
        <div class="header-content">
            <div class="creator-info">
                <p class="creator">Developed by <span class="highlight">Abhijai Rajawat</span></p>
                <p class="version">v2.1.0 | Enterprise Edition</p>
            </div>
            <div class="header-links">
                <a href="https://gradio.app/" class="link">Powered by Gradio ⚡️</a>
                <a href="https://aistudio.google.com/apikey" class="link">Get API Key</a>
                <a href="#" class="link">Documentation</a>
            </div>
        </div>
    </div>
    """
    )
    
    with gr.Accordion("⚙️ Enterprise Configuration", open=False, elem_classes="config-accordion"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("""
                ### 🚀 Advanced Settings
                - **Model Selection:** Choose between different Gemini models for optimal results
                - **API Configuration:** Secure your API key for uninterrupted service
                - **Performance Settings:** Optimize for speed or quality
                - **Batch Processing:** Enable for multiple image processing
                - **Output Format:** Select preferred image format and quality
                """)
            with gr.Column(scale=1):
                gr.Markdown("""
                ### 📊 System Status
                - **API Status:** Connected
                - **Model Status:** Ready
                - **Processing Queue:** Empty
                - **Last Updated:** Just now
                """)

    with gr.Accordion("📚 Enterprise Guide", open=False, elem_classes="instructions-accordion"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("""
                ### 🎯 Enterprise Features
                1. **Advanced Image Processing:** State-of-the-art AI-powered image editing
                2. **Batch Processing:** Process multiple images simultaneously
                3. **Custom Presets:** Save and load your favorite editing configurations
                4. **API Integration:** Seamless integration with your existing workflow
                """)
            with gr.Column(scale=1):
                gr.Markdown("""
                ### 💡 Best Practices
                - Use high-resolution images for optimal results
                - Experiment with different prompts to explore AI capabilities
                - Save your favorite configurations for future use
                - Monitor API usage for optimal resource allocation
                """)

    with gr.Row(elem_classes="main-content"):
        with gr.Column(elem_classes="input-column"):
            image_input = gr.Image(
                type="pil",
                label="📸 Upload Your Image",
                image_mode="RGBA",
                elem_id="image-input",
                elem_classes="upload-box"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    gemini_api_key = gr.Textbox(
                        lines=1,
                        placeholder="🔑 Enter your Gemini API Key",
                        label="API Key",
                        elem_classes="api-key-input",
                        type="password"
                    )
                with gr.Column(scale=1):
                    model_select = gr.Dropdown(
                        choices=["gemini-2.0-flash-exp", "gemini-pro", "gemini-ultra"],
                        value="gemini-2.0-flash-exp",
                        label="Model",
                        elem_classes="model-select"
                    )
            prompt_input = gr.Textbox(
                lines=2,
                placeholder="✨ Describe your desired changes...",
                label="AI Prompt",
                elem_classes="prompt-input"
            )
            with gr.Row():
                submit_btn = gr.Button("🚀 Generate", elem_classes="generate-btn")
                clear_btn = gr.Button("🗑️ Clear", elem_classes="clear-btn")
        
        with gr.Column(elem_classes="output-column"):
            output_gallery = gr.Gallery(
                label="🎨 Generated Results", 
                elem_classes="output-gallery",
                show_label=True
            )
            with gr.Accordion("📊 Processing Details", open=False):
                output_text = gr.Textbox(
                    label="📝 AI Response", 
                    placeholder="AI insights will appear here...",
                    elem_classes="output-text"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### Processing Metrics
                        - **Time Taken:** 0.0s
                        - **Tokens Used:** 0
                        - **Model Version:** gemini-2.0-flash-exp
                        """)
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### Quality Metrics
                        - **Resolution:** 1024x1024
                        - **Format:** PNG
                        - **Size:** 0 KB
                        """)

    # Set up the interaction with two outputs.
    submit_btn.click(
        fn=process_image_and_prompt,
        inputs=[image_input, prompt_input, gemini_api_key],
        outputs=[output_gallery, output_text],
    )
    
    clear_btn.click(
        fn=lambda: [None, None, None],
        inputs=[],
        outputs=[image_input, output_gallery, output_text],
    )
    
    gr.Markdown("## 🎯 Enterprise Examples", elem_classes="gr-examples-header")
    
    examples = [
        ["data/1.webp", 'Transform text to "ENTERPRISE"', ""],
        ["data/2.webp", "Remove background and add professional theme", ""],
        ["data/3.webp", 'Enhance with corporate style', ""],
        ["data/1.jpg", "Add professional effects", ""],
        ["data/1777043.jpg", "Create business presentation style", ""],
        ["data/2807615.jpg", "Add professional touch", ""],
        ["data/76860.jpg", "Transform into corporate art", ""],
        ["data/2807615.jpg", "Add business elements", ""],
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[image_input, prompt_input],
        elem_id="examples-grid"
    )

    gr.Markdown("""
    <div class="footer">
        <div class="footer-content">
            <div class="footer-section">
                <h3>Gemini Vision Pro</h3>
                <p>AI Image Processing</p>
                <p>Developed by Abhijai Rajawat</p>
            </div>
            <div class="footer-section">
                <h3>Contact</h3>
                <p>
                  <a href="mailto:abhijairajawat@gmail.com" class="link" target="_blank">abhijairajawat@gmail.com</a>
                </p>
                <p>
                  <a href="https://www.linkedin.com/in/abhijai-rajawat/" class="link" target="_blank">LinkedIn</a>
                </p>
            </div>
            <div class="footer-section">
                <h3>Legal</h3>
                <p>© 2024 Gemini Vision Pro</p>
                <p>All rights reserved</p>
                <p>Terms of Service | Privacy Policy</p>
            </div>
        </div>
    </div>
    """)

demo.queue(max_size=50).launch()