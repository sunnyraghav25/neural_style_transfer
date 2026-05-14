Sunny, e

Neural Style Transfer - Internship Task 3
This project implements an Artistic Neural Style Transfer model as part of my internship assignments. It uses Deep Learning to take a "Content Image" and a "Style Image" to create a third image that maintains the content of the former but adopts the artistic style of the latter.

🚀 Features
Deep Learning Framework: Built using TensorFlow 2.x and TensorFlow Hub.

Model: Utilizes the pre-trained Magenta Arbitrary Image Stylization model.

Fast Execution: Optimized to run efficiently on both CPU and GPU (via Google Colab).

🛠️ Installation & Setup
To run this project locally, ensure you have Python 3.11 or 3.12 installed (Note: TensorFlow is currently optimizing support for Python 3.13).

Clone the repository:

Bash
git clone https://github.com/sunnyraghav25/neural_style_transfer.git
Install dependencies:

Bash
pip install tensorflow tensorflow-hub matplotlib Pillow
Add your images:
Place your images in the root folder and name them content.jpg and style.jpg.

💻 Usage
Run the following command to execute the stylization:

Bash
python style_transfer.py
📊 Results
The model generates a high-quality artistic version of the input image.

Input: Content Photo + Artistic Painting.

Output: stylized_output.png (saved automatically).
