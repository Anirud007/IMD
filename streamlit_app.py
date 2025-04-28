import sys
sys.path.append('./')
from PIL import Image
import streamlit as st
import torch
import os
import numpy as np
from typing import List
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from io import BytesIO
import cv2
from segment_anything import sam_model_registry, SamPredictor
import nest_asyncio
import asyncio
nest_asyncio.apply()
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from utils_mask import get_mask_location, refine_mask
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false" 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True:
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

def load_models():
    base_path = 'yisol/IDM-VTON'
    
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    unet.requires_grad_(False)
    
    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
    
    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained(
        base_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
    )
    
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    
    UNet_Encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipe.unet_encoder = UNet_Encoder
    
    return pipe, parsing_model, openpose_model, tensor_transform

def start_tryon(human_img, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
    pipe, parsing_model, openpose_model, tensor_transform = st.session_state.models
    
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)
    
    garm_img = garm_img.convert("RGB").resize((768,1024))
    human_img_orig = human_img.convert("RGB")
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))
    
    if is_checked:
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        mask = pil_to_binary_mask(human_img.resize((768, 1024)))
    
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)
    
    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    
    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:,:,::-1]
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                
                prompt = "a photo of " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                if not isinstance(prompt, List):
                    prompt = [prompt] * 1
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * 1
                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )
                
                pose_img = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
                garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
                generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                images = pipe(
                    prompt_embeds=prompt_embeds.to(device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img.to(device, torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                    cloth=garm_tensor.to(device, torch.float16),
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img.resize((768,1024)),
                    guidance_scale=2.0,
                )[0]
    
    if is_checked_crop:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray


def load_parsing_and_openpose_models():
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    return parsing_model, openpose_model

def display_masking_technique(human_img, model_type, category, min_area):
    parsing_model, openpose_model = load_parsing_and_openpose_models()
    openpose_model.preprocessor.body_estimation.model.to(device)
    
    human_img = human_img.convert("RGB")
    keypoints = openpose_model(human_img.resize((384,512)))
    model_parse, _ = parsing_model(human_img.resize((384,512)))
    
    # Get mask using the selected model type and category
    mask, mask_gray = get_mask_location(model_type, category, model_parse, keypoints)
    mask = mask.resize(human_img.size)
    
    # Refine the mask
    refined_mask = refine_mask(mask)
    
    mask_gray = (1-transforms.ToTensor()(refined_mask)) * transforms.ToTensor()(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)
    
    # Extract the masked area from the original image
    mask_np = np.array(refined_mask) / 255
    human_img_np = np.array(human_img)
    extracted_area = Image.fromarray((human_img_np * mask_np[..., None]).astype(np.uint8))
    
    return mask_gray, extracted_area

def refine_mask(mask, min_area=500):
    # Convert mask to numpy array 
    mask_np = np.array(mask)

    # Find contours
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new mask to draw filtered contours
    refined_mask_np = np.zeros_like(mask_np)

    # Filter contours by area
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(refined_mask_np, [contour], -1, (255), thickness=cv2.FILLED)

    # Apply Gaussian blur to smooth the mask
    refined_mask_np = cv2.GaussianBlur(refined_mask_np, (5, 5), 0)

    # Convert back to PIL Image
    refined_mask = Image.fromarray(refined_mask_np)
    return refined_mask

def load_sam_model():
    """Load the SAM model for interactive segmentation"""
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    if not os.path.exists(sam_checkpoint):
        st.warning("Downloading SAM model checkpoint... This may take a few minutes.")
        import urllib.request
        import zipfile
        import tempfile
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def download_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            progress_bar.progress(percent)
            status_text.text(f"Downloading: {percent}%")
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download the checkpoint
            checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            checkpoint_path = os.path.join(tmpdir, sam_checkpoint)
            
            try:
                urllib.request.urlretrieve(checkpoint_url, checkpoint_path, reporthook=download_progress)
                # Move the downloaded file to the current directory
                import shutil
                shutil.move(checkpoint_path, sam_checkpoint)
                progress_bar.progress(100)
                status_text.text("Download complete!")
                st.success("SAM model checkpoint downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download SAM model checkpoint: {str(e)}")
                st.error("Please download it manually from https://github.com/facebookresearch/segment-anything#model-checkpoints")
                return None
    
    # Show loading indicator for model initialization
    with st.spinner("Initializing SAM model..."):
        try:
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            predictor = SamPredictor(sam)
            st.success("SAM model loaded successfully!")
            return predictor
        except Exception as e:
            st.error(f"Failed to load SAM model: {str(e)}")
            return None

def main():
    st.set_page_config(page_title="VTON Virtual Try-on", layout="wide")
    
    st.title("VTON 👕👔👚")
    st.markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    
    if 'models' not in st.session_state:
        with st.spinner('Loading models...'):
            st.session_state.models = load_models()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Virtual Try-on", "Masking Process", "Interactive Segmentation"])
    
    with tab1:
        st.subheader("Virtual Try-on")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.subheader("Human Image")
            human_image_placeholder = st.empty()
            human_img = st.file_uploader("Upload human image", type=['png', 'jpg', 'jpeg'])
            if human_img is not None:
                human_image_placeholder.image(human_img, use_column_width=True)
            is_checked = st.checkbox("Use auto-generated mask (Takes 5 seconds)", value=True)
            is_checked_crop = st.checkbox("Use auto-crop & resizing", value=False)
        
        with col2:
            st.subheader("Garment Image")
            garm_image_placeholder = st.empty()
            garm_img = st.file_uploader("Upload garment image", type=['png', 'jpg', 'jpeg'])
            garment_des = st.text_input("Description of garment", placeholder="e.g., Short Sleeve Round Neck T-shirts")
            if garm_img is not None:
                garm_image_placeholder.image(garm_img, use_column_width=True)
        
        with col3:
            st.subheader("Masked Image")
            masked_img_placeholder = st.empty()
        
        with col4:
            st.subheader("Output")
            output_placeholder = st.empty()
        
        with st.sidebar:
            st.subheader("Advanced Settings")
            denoise_steps = st.slider("Denoising Steps", min_value=20, max_value=40, value=30, step=1)
            seed = st.number_input("Seed", min_value=-1, max_value=2147483647, value=42, step=1)
        
        if st.button("Try-on"):
            if human_img is None or garm_img is None:
                st.error("Please upload both human and garment images")
                return
            
            with st.spinner('Generating virtual try-on...'):
                human_img = Image.open(human_img)
                garm_img = Image.open(garm_img)
                result_img, masked_img = start_tryon(
                    human_img, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed
                )
                
                masked_img_placeholder.image(masked_img, use_column_width=True)
                output_placeholder.image(result_img, use_column_width=True)
    
    with tab2:
        st.subheader("Masking Process")
        human_img_masking = st.file_uploader("Upload human image for masking", type=['png', 'jpg', 'jpeg'])
        model_type = st.selectbox("Model Type", options=["hd", "dc"], index=0)
        category = st.selectbox("Category", options=["dresses", "upper_body", "lower_body"], index=1)
        min_area = st.slider("Minimum Area for Contour", min_value=100, max_value=5000, value=500, step=100)
        
        # Initialize session state for parameters
        if 'mask_params' not in st.session_state:
            st.session_state.mask_params = {
                'model_type': model_type,
                'category': category,
                'min_area': min_area
            }
        
        # Check if any parameter has changed
        param_changed = (
            st.session_state.mask_params['model_type'] != model_type or
            st.session_state.mask_params['category'] != category or
            st.session_state.mask_params['min_area'] != min_area
        )
        
        # Update session state with new parameters
        st.session_state.mask_params.update({
            'model_type': model_type,
            'category': category,
            'min_area': min_area
        })
        
        # Add a remask button
        if st.button("Remask") or param_changed:
            if human_img_masking is not None:
                human_img_masking = Image.open(human_img_masking)
                mask_gray, extracted_area = display_masking_technique(human_img_masking, model_type, category, min_area)
                
                # Display images in a single row without resizing
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(human_img_masking, caption="Original Image")
                with col2:
                    st.image(mask_gray, caption="Image with Grey Mask")
                with col3:
                    st.image(extracted_area, caption="Extracted Mask Area")
                
                # Provide download link for the extracted area
                buffered = BytesIO()
                extracted_area.save(buffered, format="PNG")
                st.download_button(
                    label="Download Extracted Area as PNG",
                    data=buffered.getvalue(),
                    file_name="extracted_area.png",
                    mime="image/png"
                )

    with tab3:
        st.subheader("Interactive Segmentation with SAM")
        st.markdown("Upload an image and click on it to segment objects. The model will automatically segment the object around your click point.")
        
        # File uploader for the image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Initialize SAM model if not already loaded
            if 'sam_predictor' not in st.session_state:
                with st.spinner('Loading SAM model...'):
                    st.session_state.sam_predictor = load_sam_model()
            
            if st.session_state.sam_predictor is not None:
                # Convert image to numpy array for SAM
                image_np = np.array(image)
                
                # Set image in SAM predictor
                st.session_state.sam_predictor.set_image(image_np)
                
                # Get click coordinates using Streamlit's click_data
                click_data = st.empty()
                click_data.markdown("Click on the image above to segment an object")
                
                # Create a container for displaying the segmented result
                result_container = st.empty()
                
                # Get click coordinates from the user
                click_x = st.number_input("X coordinate (0 to image width)", min_value=0, max_value=image.width, value=image.width//2)
                click_y = st.number_input("Y coordinate (0 to image height)", min_value=0, max_value=image.height, value=image.height//2)
                
                if st.button("Segment"):
                    # Create input point and label for SAM
                    input_point = np.array([[click_x, click_y]])
                    input_label = np.array([1])  # 1 indicates foreground point
                    
                    # Get mask from SAM
                    masks, scores, logits = st.session_state.sam_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )
                    
                    # Display the best mask
                    mask = masks[0]  # Get the first (best) mask
                    
                    # Create a visualization of the mask
                    mask_visualization = np.zeros_like(image_np)
                    mask_visualization[mask] = image_np[mask]
                    
                    # Display the segmented result
                    result_container.image(mask_visualization, caption="Segmented Object", use_column_width=True)
                    
                    # Add download button for the mask
                    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                    buffered = BytesIO()
                    mask_pil.save(buffered, format="PNG")
                    st.download_button(
                        label="Download Mask",
                        data=buffered.getvalue(),
                        file_name="segmentation_mask.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main() 