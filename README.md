## README: Pothole Segmentation for Road Damage

### Project Overview
This project aims to detect and segment potholes from road images and videos using a pre-trained YOLOv8 segmentation model. The process involves extracting a dataset, fine-tuning the YOLOv8 model, and evaluating its performance through a series of metrics such as loss curves, precision-recall curves, and confusion matrices. Additionally, the trained model can perform real-time inferences on new images and videos to detect and highlight road damage.

### Steps to Run the Project

1. **Dataset Extraction**  
   The project begins by extracting a zip file containing the dataset. Make sure the zip file is in the correct directory. If the file exists, it will be extracted to a folder called `extracted_data/`.

2. **Installing Dependencies**  
   Install necessary packages, including the `ultralytics` package, which contains the YOLOv8 model:
   ```
   !pip install ultralytics
   ```

3. **Load YOLOv8 Model**  
   The project uses the YOLOv8 segmentation model (`yolov8n-seg.pt`), which is a lightweight model pre-trained for object detection and segmentation tasks.

4. **Dataset Preparation**  
   After extracting the dataset, the code checks the number of training and validation images, ensuring all images have the correct size (640x640 pixels). The dataset is stored in directories like `train/images` and `valid/images`.

5. **Fine-Tuning the Model**  
   The YOLOv8 model is fine-tuned using the provided dataset, with 150 training epochs. Parameters such as learning rate, batch size, and early stopping are set to optimize the model's performance.

6. **Evaluating the Model**  
   Once training is complete, several performance evaluation techniques are used:
   - **Learning Curves**: Track the model's loss for bounding boxes, classification, and segmentation.
   - **Precision-Recall Curves**: Analyze the model's ability to detect potholes accurately at various confidence levels.
   - **Confusion Matrix**: Visualize the confusion matrix and its normalized version to assess model performance.

7. **Model Inference**  
   After training, the model can perform inference on new images and videos. The code generates inferences on both the validation dataset and an additional sample video.

8. **Real-Time Road Damage Detection**  
   The model is also capable of real-time detection of road damage, segmenting potholes and calculating areas of damage.

### Key Features
- **Segmentation Model**: Uses YOLOv8's segmentation capabilities to detect potholes and segment the damaged areas.
- **Custom Dataset**: The model is fine-tuned on a custom dataset specifically for pothole detection.
- **Evaluation Metrics**: Includes learning curves, confusion matrices, precision-recall curves, and segmentation mask visualizations.

### How to Use
1. Extract the dataset.
2. Fine-tune the YOLOv8 model using the provided dataset.
3. Evaluate the model's performance through the various plots and metrics.
4. Use the model for inference on new images or videos for real-time road damage detection.

### Files Included
- **Pothole Segmentation Dataset**: Contains images for training and validation.
- **YOLOv8 Model Files**: Pre-trained and fine-tuned models.
- **Training Logs**: Includes results for loss curves, precision-recall curves, and more.

### Future Work
- Expand the dataset for more general road damage scenarios.
- Improve model accuracy with advanced hyperparameter tuning.
- Deploy the model in real-world road monitoring systems.
