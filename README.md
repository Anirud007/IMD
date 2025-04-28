# IMD

**IMD** is a computer vision project that leverages [Detectron2](https://github.com/facebookresearch/detectron2) and DensePose for human pose estimation and segmentation. It provides a web-based interface for users to upload images and visualize DensePose outputs.

## Features

- **DensePose Integration**: Maps all human pixels of an RGB image to the 3D surface of the human body.
- **Detectron2 Backend**: Utilizes Facebook AI Research's Detectron2 for object detection and segmentation.
- **Web Interface**: Offers a user-friendly interface built with Streamlit for easy interaction.
- **Modular Codebase**: Organized structure with separate modules for utilities, application logic, and examples.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Anirud007/IMD.git
   cd IMD
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To launch the Streamlit web application:

```bash
streamlit run streamlit_app.py
```

This will start the app in your default web browser. You can upload images and view the DensePose outputs directly.

## Project Structure

```
IMD/
├── app.py                 # Main application script
├── streamlit_app.py       # Streamlit web interface
├── apply_net.py           # Applies the DensePose model to input images
├── utils_mask.py          # Utility functions for mask processing
├── requirements.txt       # Python dependencies
├── densepose/             # DensePose-related modules
├── detectron2/            # Detectron2 modules
├── example/               # Example images and outputs
└── __pycache__/           # Cached Python files
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Detectron2](https://github.com/facebookresearch/detectron2)
- [DensePose](https://github.com/facebookresearch/DensePose)
- [Streamlit](https://streamlit.io/)
