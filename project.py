import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import cv2
from PIL import Image, ImageDraw
import os

class HandwrittenDigitRecognizer:
    def __init__(self):
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the MNIST dataset"""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape data to add channel dimension
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def build_model(self):
        """Build a CNN model for digit recognition"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
        """Train the model"""
        print("Training the model...")
        
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
        ]
        
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, x_test, y_test):
        """Evaluate the model performance"""
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        return test_loss, test_acc
    
    def preprocess_image(self, image_path_or_array):
        """Preprocess an image for prediction"""
        if isinstance(image_path_or_array, str):
            # Load image from path
            image = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        else:
            # Use provided array
            image = image_path_or_array
            
        # Resize to 28x28
        image = cv2.resize(image, (28, 28))
        
        # Invert if needed (MNIST digits are white on black)
        if np.mean(image) > 127:
            image = 255 - image
            
        # Normalize
        image = image.astype('float32') / 255.0
        
        # Reshape for model input
        image = image.reshape(1, 28, 28, 1)
        
        return image
    
    def predict_digit(self, image_path_or_array):
        """Predict a single digit"""
        processed_image = self.preprocess_image(image_path_or_array)
        prediction = self.model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_digit, confidence
    
    def segment_digits(self, image_path):
        """Segment an image containing multiple digits"""
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by x-coordinate (left to right)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        digit_images = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out small contours (noise)
            if w > 10 and h > 10:
                # Extract digit region
                digit = thresh[y:y+h, x:x+w]
                
                # Add padding to make it square
                max_dim = max(w, h)
                padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
                
                # Center the digit
                start_x = (max_dim - w) // 2
                start_y = (max_dim - h) // 2
                padded[start_y:start_y+h, start_x:start_x+w] = digit
                
                digit_images.append(padded)
        
        return digit_images
    
    def predict_sequence(self, image_path):
        """Predict a sequence of digits from an image"""
        digit_images = self.segment_digits(image_path)
        predictions = []
        
        for digit_img in digit_images:
            predicted_digit, confidence = self.predict_digit(digit_img)
            predictions.append((predicted_digit, confidence))
        
        return predictions
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def create_test_image(self, digits_sequence, filename):
        """Create a test image with a sequence of digits"""
        # Create a white image
        img_width = len(digits_sequence) * 40 + 20
        img_height = 60
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw digits
        for i, digit in enumerate(digits_sequence):
            x = 30 + i * 40
            y = 20
            draw.text((x, y), str(digit), fill='black')
        
        img.save(filename)
        print(f"Test image saved as {filename}")

class DigitRecognitionUI:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        
    def interactive_menu(self):
        """Main interactive menu for testing"""
        while True:
            print("\n" + "="*60)
            print("🎯 HANDWRITTEN DIGIT RECOGNITION - TEST INTERFACE")
            print("="*60)
            print("1. 🖼️  Test single image file")
            print("2. 📷  Capture from webcam")
            print("3. 🎨  Draw digit with mouse")
            print("4. 📝  Test sequence of digits")
            print("5. 📁  Batch test multiple images")
            print("6. 🧪  Test with MNIST samples")
            print("7. 📊  Show model performance")
            print("8. 💾  Save/Load model")
            print("9. ❌  Exit")
            print("-"*60)
            
            choice = input("Enter your choice (1-9): ").strip()
            
            if choice == '1':
                self.test_single_image()
            elif choice == '2':
                self.capture_from_webcam()
            elif choice == '3':
                self.draw_digit_interface()
            elif choice == '4':
                self.test_digit_sequence()
            elif choice == '5':
                self.batch_test_images()
            elif choice == '6':
                self.test_mnist_samples()
            elif choice == '7':
                self.show_model_performance()
            elif choice == '8':
                self.save_load_model()
            elif choice == '9':
                print("👋 Thanks for using the digit recognizer!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def test_single_image(self):
        """Test a single image file"""
        print("\n📁 Testing Single Image")
        print("-" * 30)
        image_path = input("Enter image path (or 'back' to return): ").strip()
        
        if image_path.lower() == 'back':
            return
        
        if not os.path.exists(image_path):
            print("❌ File not found!")
            return
        
        try:
            predicted_digit, confidence = self.recognizer.predict_digit(image_path)
            print(f"✅ Predicted digit: {predicted_digit}")
            print(f"📊 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            
            # Show the image if matplotlib is available
            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                plt.figure(figsize=(6, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(img, cmap='gray')
                plt.title('Original Image')
                plt.axis('off')
                
                # Show preprocessed image
                processed = self.recognizer.preprocess_image(image_path)
                plt.subplot(1, 2, 2)
                plt.imshow(processed.reshape(28, 28), cmap='gray')
                plt.title(f'Predicted: {predicted_digit} ({confidence:.3f})')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            except:
                print("📊 Image display not available")
                
        except Exception as e:
            print(f"❌ Error processing image: {e}")
    
    def capture_from_webcam(self):
        """Capture digit from webcam"""
        print("\n📷 Webcam Capture")
        print("-" * 20)
        print("Instructions:")
        print("- Press SPACE to capture")
        print("- Press ESC to exit")
        
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Cannot open webcam")
                return
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Add instructions on frame
                cv2.putText(frame, "Press SPACE to capture, ESC to exit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Draw digit on paper and show to camera", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow('Webcam - Digit Recognition', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == 32:  # SPACE
                    # Capture and predict
                    predicted_digit, confidence = self.recognizer.predict_digit(gray)
                    print(f"✅ Predicted: {predicted_digit} (Confidence: {confidence:.3f})")
                    
                    # Show result on frame
                    result_text = f"Predicted: {predicted_digit} ({confidence:.3f})"
                    cv2.putText(frame, result_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Webcam - Digit Recognition', frame)
                    cv2.waitKey(2000)  # Show result for 2 seconds
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"❌ Webcam error: {e}")
    
    def draw_digit_interface(self):
        """Simple drawing interface using matplotlib"""
        print("\n🎨 Draw Digit Interface")
        print("-" * 25)
        print("Draw a digit in the window that opens")
        print("Close the window when done")
        
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(0, 28)
            ax.set_ylim(0, 28)
            ax.set_aspect('equal')
            ax.invert_yaxis()  # Invert y-axis to match image coordinates
            ax.set_title('Draw a digit (close window when done)', fontsize=14)
            
            # Create empty 28x28 canvas
            canvas = np.zeros((28, 28))
            im = ax.imshow(canvas, cmap='gray_r', vmin=0, vmax=1)
            
            # Drawing variables
            drawing = {'is_drawing': False, 'points': []}
            
            def on_press(event):
                if event.inaxes != ax:
                    return
                drawing['is_drawing'] = True
                drawing['points'] = [(int(event.xdata), int(event.ydata))]
            
            def on_motion(event):
                if not drawing['is_drawing'] or event.inaxes != ax:
                    return
                x, y = int(event.xdata), int(event.ydata)
                if 0 <= x < 28 and 0 <= y < 28:
                    # Draw a small circle
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < 28 and 0 <= ny < 28:
                                canvas[ny, nx] = 1.0
                    im.set_array(canvas)
                    fig.canvas.draw()
            
            def on_release(event):
                drawing['is_drawing'] = False
            
            fig.canvas.mpl_connect('button_press_event', on_press)
            fig.canvas.mpl_connect('motion_notify_event', on_motion)
            fig.canvas.mpl_connect('button_release_event', on_release)
            
            plt.show()
            
            # Predict the drawn digit
            if np.any(canvas):
                # Reshape for prediction
                canvas_reshaped = canvas.reshape(1, 28, 28, 1)
                prediction = self.recognizer.model.predict(canvas_reshaped, verbose=0)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)
                
                print(f"✅ Predicted digit: {predicted_digit}")
                print(f"📊 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                
                # Show all probabilities
                print("\n📊 All probabilities:")
                for i, prob in enumerate(prediction[0]):
                    print(f"  {i}: {prob:.3f} ({prob*100:.1f}%)")
            else:
                print("❌ Nothing was drawn!")
                
        except Exception as e:
            print(f"❌ Drawing interface error: {e}")
    
    def test_digit_sequence(self):
        """Test sequence of digits"""
        print("\n📝 Testing Digit Sequence")
        print("-" * 30)
        image_path = input("Enter path to sequence image (or 'back' to return): ").strip()
        
        if image_path.lower() == 'back':
            return
        
        if not os.path.exists(image_path):
            print("❌ File not found!")
            return
        
        try:
            predictions = self.recognizer.predict_sequence(image_path)
            
            if predictions:
                sequence = [str(pred[0]) for pred in predictions]
                print(f"✅ Recognized sequence: {''.join(sequence)}")
                print("\nDetailed results:")
                for i, (digit, confidence) in enumerate(predictions):
                    print(f"  Position {i+1}: {digit} (confidence: {confidence:.3f})")
            else:
                print("❌ No digits detected in the image")
                
        except Exception as e:
            print(f"❌ Error processing sequence: {e}")
    
    def batch_test_images(self):
        """Test multiple images in a folder"""
        print("\n📁 Batch Testing")
        print("-" * 20)
        folder_path = input("Enter folder path containing images: ").strip()
        
        if not os.path.exists(folder_path):
            print("❌ Folder not found!")
            return
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))
        
        if not image_files:
            print("❌ No image files found!")
            return
        
        print(f"Found {len(image_files)} images. Processing...")
        
        results = []
        for i, image_path in enumerate(image_files):
            try:
                predicted_digit, confidence = self.recognizer.predict_digit(image_path)
                results.append({
                    'file': os.path.basename(image_path),
                    'prediction': predicted_digit,
                    'confidence': confidence
                })
                print(f"  {i+1}/{len(image_files)}: {os.path.basename(image_path)} -> {predicted_digit} ({confidence:.3f})")
            except Exception as e:
                print(f"  ❌ Error with {os.path.basename(image_path)}: {e}")
        
        # Summary
        print(f"\n📊 Batch Test Summary:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Successfully processed: {len(results)}")
        if results:
            avg_confidence = np.mean([r['confidence'] for r in results])
            print(f"  Average confidence: {avg_confidence:.3f}")
    
    def test_mnist_samples(self):
        """Test with random MNIST samples"""
        print("\n🧪 Testing MNIST Samples")
        print("-" * 30)
        
        try:
            # Load test data
            (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            
            num_samples = int(input("How many samples to test? (default 10): ").strip() or "10")
            
            # Random samples
            indices = np.random.choice(len(x_test), num_samples, replace=False)
            
            correct = 0
            for i, idx in enumerate(indices):
                test_image = x_test[idx]
                actual_digit = y_test[idx]
                
                predicted_digit, confidence = self.recognizer.predict_digit(test_image)
                
                is_correct = predicted_digit == actual_digit
                if is_correct:
                    correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"  Sample {i+1}: {status} Predicted={predicted_digit}, Actual={actual_digit}, Confidence={confidence:.3f}")
            
            accuracy = correct / num_samples
            print(f"\n📊 Accuracy on {num_samples} samples: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error testing MNIST samples: {e}")
    
    def show_model_performance(self):
        """Show model performance metrics"""
        print("\n📊 Model Performance")
        print("-" * 25)
        
        try:
            # Load test data
            (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_test = x_test.astype('float32') / 255.0
            x_test = x_test.reshape(-1, 28, 28, 1)
            y_test = tf.keras.utils.to_categorical(y_test, 10)
            
            # Evaluate model
            test_loss, test_acc = self.recognizer.model.evaluate(x_test, y_test, verbose=0)
            
            print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
            print(f"Test Loss: {test_loss:.4f}")
            
            # Show training history if available
            if self.recognizer.history:
                self.recognizer.plot_training_history()
            
        except Exception as e:
            print(f"❌ Error showing performance: {e}")
    
    def save_load_model(self):
        """Save or load model"""
        print("\n💾 Save/Load Model")
        print("-" * 22)
        print("1. Save current model")
        print("2. Load existing model")
        
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == '1':
            filename = input("Enter filename (default: my_digit_model.h5): ").strip() or "my_digit_model.h5"
            try:
                self.recognizer.save_model(filename)
            except Exception as e:
                print(f"❌ Error saving model: {e}")
        
        elif choice == '2':
            filename = input("Enter filename to load: ").strip()
            if os.path.exists(filename):
                try:
                    self.recognizer.load_model(filename)
                    print("✅ Model loaded successfully!")
                except Exception as e:
                    print(f"❌ Error loading model: {e}")
            else:
                print("❌ File not found!")

def main():
    print("🚀 Starting Handwritten Digit Recognition System...")
    
    # Initialize the recognizer
    recognizer = HandwrittenDigitRecognizer()
    
    # Check if pre-trained model exists
    model_file = 'digit_recognition_model.h5'
    if os.path.exists(model_file):
        print(f"📁 Found existing model: {model_file}")
        load_existing = input("Load existing model? (y/n): ").strip().lower()
        
        if load_existing == 'y':
            recognizer.load_model(model_file)
        else:
            # Train new model
            print("🔄 Training new model...")
            (x_train, y_train), (x_test, y_test) = recognizer.load_and_preprocess_data()
            model = recognizer.build_model()
            recognizer.train_model(x_train, y_train, x_test, y_test, epochs=10)
            recognizer.save_model(model_file)
    else:
        # Train new model
        print("🔄 Training new model...")
        (x_train, y_train), (x_test, y_test) = recognizer.load_and_preprocess_data()
        model = recognizer.build_model()
        recognizer.train_model(x_train, y_train, x_test, y_test, epochs=10)
        recognizer.save_model(model_file)
    
    # Start interactive UI
    ui = DigitRecognitionUI(recognizer)
    ui.interactive_menu()

if __name__ == "__main__":
    main()