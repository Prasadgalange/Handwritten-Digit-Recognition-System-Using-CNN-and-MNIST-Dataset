A deep learning project that recognizes handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN). The model achieves an impressive **99.1% test accuracy**.
Project Highlights
Dataset:Trained a CNN model on 70,000 MNIST images (60,000 for training, 10,000 for testing).
Optimization techniques: Applied data preprocessing, normalization (scaling pixel values to 0-1), and dropout regularization to minimize overfitting.
Analysis: Visualized training and validation curves (accuracy and loss) using Matplotlib to diagnose model performance and guide tuning decisions.
Tech Stack
Python
TensorFlow & Keras
NumPy
Matplotlib
Installation & Setup
1. Clone this repository to your local machine:
   ```bash
   git clone <your-repository-url>
   cd <your-repository-directory>
   ```
2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install tensorflow keras matplotlib numpy
   ```
Usage
To train the model and view the evaluation results, execute the main script:
```bash
python project_1.py
```
Running the script will:
1. Download the MNIST dataset.
2. Train the Convolutional Neural Network for 5 epochs.
3. Output the final test accuracy in the terminal.
4. Predict a sample digit from the test set and pop up a window displaying the actual image.
5. Generate and display graphical plots comparing training/validation accuracy and training/validation loss.
Results
Final Test Accuracy: ~99.1%
Loss/Accuracy Curves: The generated graphs provide a clear visual indication of how the model learns over time and demonstrate the effectiveness of dropout in preventing severe overfitting.
