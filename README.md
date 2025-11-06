# Deep-learning---egg-detection

This project uses deep learning techniques to detect and classify eggs (e.g., damaged vs. non-damaged) based on image data.

---

## Folder Structure

### `Model/`
Contains the **main model** used to run the project.  
This is the core trained model applied for running the project.

### `user_models/`
Stores all **models created by our team members**.  
Each subfolder may contain individual experiments or personalized model versions.

### `results/`
Contains all **training and evaluation outputs**, including:
- Accuracy and loss curves  
- Classification metrics reports  
- Confusion matrix visualizations  

---

## Technologies Used
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib
- OpenCV

---

## How to Run the Project


### Install Required Libraries
Install the necessary Python packages:

```bash
pip install tensorflow matplotlib numpy opencv-python
```

### Download the Repository


```bash
git clone https://github.com/tuandung1625/Deep-learning---egg-detection
cd egg-damage-detection
```


### Add Your Image
Place the image you want to test inside the project folder.


Example:
```
egg-damage-detection/
├─ predict.py
├─ model/
├─ your_image.jpg <--- add here
```


### Run the Prediction Script


```bash
python predict.py image_file_name.jpg
```


### Output
The program will display:
- Predicted class → Damaged or Not Damaged
- Model confidence (%)


Example:
```
Predicted: Damaged (92.5% confidence)
