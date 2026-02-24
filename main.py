"""
Main application for facial recognition meeting assistant.
Integrates camera, detection, embedding, database, and recognition.
"""

import sys
import logging
import config
from camera import Camera
from detector import FaceDetector
from embedder import FaceEmbedder
from database import FaceDatabase
from recognizer import FaceRecognizer

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)


class FacialRecognitionSystem:
    """Main facial recognition system."""
    
    def __init__(self):
        """Initialize all components."""
        logger.info("Initializing Facial Recognition System...")
        
        self.camera = None
        self.detector = None
        self.embedder = None
        self.database = None
        self.recognizer = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize camera
            logger.info("Initializing camera...")
            self.camera = Camera()
            
            # Initialize face detector
            logger.info("Initializing face detector...")
            self.detector = FaceDetector()
            
            # Initialize face embedder
            logger.info("Initializing face embedder...")
            self.embedder = FaceEmbedder()
            
            # Initialize database
            logger.info("Initializing database...")
            self.database = FaceDatabase()
            
            if not self.database.connect():
                raise Exception("Failed to connect to database")
            
            # Initialize recognizer
            logger.info("Initializing recognizer...")
            self.recognizer = FaceRecognizer(self.database)
            
            logger.info("✓ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up resources...")
        
        if self.camera:
            self.camera.close()
        
        if self.database:
            self.database.disconnect()
        
        logger.info("✓ Cleanup complete")
    
    def process_face(self, image_path: str) -> tuple:
        """
        Process a captured face image through the entire pipeline.
        
        Args:
            image_path: Path to captured image
            
        Returns:
            tuple: (success, name, is_new, confidence, details)
        """
        try:
            logger.info("=" * 70)
            logger.info("PROCESSING FACE")
            logger.info("=" * 70)
            
            # Step 1: Detect face
            logger.info("Step 1/3: Detecting face...")
            success, face, detection_info = self.detector.detect_and_extract_largest_face(
                image_path
            )
            
            if not success:
                logger.error("Failed to detect face")
                return False, None, False, 0.0, "No face detected"
            
            logger.info(f"✓ Face detected (confidence: {detection_info['confidence']:.3f})")
            
            # Step 2: Generate embedding
            logger.info("Step 2/3: Generating face embedding...")
            success, embedding = self.embedder.embed_face(face)
            
            if not success:
                logger.error("Failed to generate embedding")
                return False, None, False, 0.0, "Failed to generate embedding"
            
            logger.info("✓ Embedding generated successfully")
            
            # Step 3: Recognize or register
            logger.info("Step 3/3: Recognizing person...")
            name, is_new, confidence = self.recognizer.recognize_or_register(
                embedding,
                ask_for_name_callback=self._ask_for_name
            )
            
            if name is None:
                logger.error("Failed to recognize or register person")
                return False, None, False, 0.0, "Failed to recognize or register"
            
            if is_new:
                logger.info(f"✓ NEW person registered: '{name}'")
                details = f"Registered as new person"
            else:
                logger.info(f"✓ Person RECOGNIZED: '{name}' (confidence: {confidence:.3f})")
                details = f"Recognized with {confidence:.1%} confidence"
            
            logger.info("=" * 70)
            
            return True, name, is_new, confidence, details
            
        except Exception as e:
            logger.error(f"Error processing face: {e}")
            return False, None, False, 0.0, str(e)
    
    def _ask_for_name(self) -> str:
        """
        Ask user for name of new person.
        
        Returns:
            str: Person's name
        """
        print("\n" + "=" * 70)
        print("NEW PERSON DETECTED")
        print("=" * 70)
        
        while True:
            name = input("Enter person's name: ").strip()
            
            if not name:
                print("Name cannot be empty. Please try again.")
                continue
            
            if len(name) < config.MIN_NAME_LENGTH:
                print(f"Name too short (minimum {config.MIN_NAME_LENGTH} characters)")
                continue
            
            if len(name) > config.MAX_NAME_LENGTH:
                print(f"Name too long (maximum {config.MAX_NAME_LENGTH} characters)")
                continue
            
            # Validate characters
            invalid_chars = [c for c in name if c not in config.ALLOWED_NAME_CHARS]
            if invalid_chars:
                print(f"Invalid characters: {invalid_chars}")
                print(f"Allowed: letters, numbers, spaces, hyphens, underscores, periods")
                continue
            
            # Confirm
            confirm = input(f"Confirm name '{name}'? (y/n): ").strip().lower()
            if confirm == 'y':
                return name
            else:
                print("Let's try again...")
    
    def run_interactive(self):
        """Run interactive mode with live camera feed."""
        try:
            logger.info("\n" + "=" * 70)
            logger.info("FACIAL RECOGNITION SYSTEM - INTERACTIVE MODE")
            logger.info("=" * 70)
            
            # Show registered people
            people = self.recognizer.get_all_registered_people()
            logger.info(f"Currently registered people: {len(people)}")
            if people:
                logger.info(f"Names: {', '.join(people)}")
            
            print("\n" + "=" * 70)
            print("INSTRUCTIONS:")
            print(f"  - Press '{config.CAPTURE_KEY}' to capture face")
            print(f"  - Press '{config.QUIT_KEY}' to quit")
            print("=" * 70 + "\n")
            
            # Run capture loop
            image_path = self.camera.run_capture_loop()
            
            if image_path is None:
                logger.info("No image captured - exiting")
                return
            
            # Process captured face
            success, name, is_new, confidence, details = self.process_face(image_path)
            
            # Display results
            print("\n" + "=" * 70)
            print("RESULT:")
            print("=" * 70)
            
            if success:
                if is_new:
                    print(f"✓ NEW person registered: {name}")
                    print(f"  Status: First time interaction")
                else:
                    print(f"✓ Person RECOGNIZED: {name}")
                    print(f"  Confidence: {confidence:.1%}")
                    print(f"  Status: Returning person")
            else:
                print(f"✗ Failed: {details}")
            
            print("=" * 70 + "\n")
            
            # Cleanup temp image
            if config.DELETE_TEMP_IMAGES:
                self.camera.delete_temp_image()
            
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
        finally:
            # Final cleanup
            if self.camera:
                self.camera.close(cleanup_temp=True)
            if self.database:
                self.database.disconnect()
    
    def run_batch(self, image_path: str):
        """
        Run recognition on a single image file.
        
        Args:
            image_path: Path to image file
        """
        try:
            logger.info("\n" + "=" * 70)
            logger.info("FACIAL RECOGNITION SYSTEM - BATCH MODE")
            logger.info("=" * 70)
            logger.info(f"Processing image: {image_path}")
            
            # Process face
            success, name, is_new, confidence, details = self.process_face(image_path)
            
            # Display results
            print("\n" + "=" * 70)
            print("RESULT:")
            print("=" * 70)
            
            if success:
                if is_new:
                    print(f"✓ NEW person registered: {name}")
                else:
                    print(f"✓ Person RECOGNIZED: {name}")
                    print(f"  Confidence: {confidence:.1%}")
            else:
                print(f"✗ Failed: {details}")
            
            print("=" * 70 + "\n")
            
        except Exception as e:
            logger.error(f"Error in batch mode: {e}")
        finally:
            self.cleanup()
    
    def list_registered_people(self):
        """List all registered people."""
        try:
            people = self.recognizer.get_all_registered_people()
            count = len(people)
            
            print("\n" + "=" * 70)
            print(f"REGISTERED PEOPLE: {count}")
            print("=" * 70)
            
            if people:
                for i, name in enumerate(people, 1):
                    print(f"{i}. {name}")
            else:
                print("No people registered yet")
            
            print("=" * 70 + "\n")
            
        except Exception as e:
            logger.error(f"Error listing people: {e}")
        finally:
            self.cleanup()


def print_usage():
    """Print usage instructions."""
    print("\n" + "=" * 70)
    print("FACIAL RECOGNITION SYSTEM - USAGE")
    print("=" * 70)
    print("\nModes:")
    print("  1. Interactive mode (live camera):")
    print("     python main.py")
    print()
    print("  2. Batch mode (process image file):")
    print("     python main.py <image_path>")
    print()
    print("  3. List registered people:")
    print("     python main.py --list")
    print()
    print("Examples:")
    print("  python main.py")
    print("  python main.py photo.jpg")
    print("  python main.py --list")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    try:
        # Parse arguments
        if len(sys.argv) > 1:
            arg = sys.argv[1]
            
            if arg in ['--help', '-h']:
                print_usage()
                return
            
            elif arg in ['--list', '-l']:
                # List mode
                system = FacialRecognitionSystem()
                system.list_registered_people()
            
            else:
                # Batch mode
                image_path = arg
                system = FacialRecognitionSystem()
                system.run_batch(image_path)
        
        else:
            # Interactive mode
            system = FacialRecognitionSystem()
            system.run_interactive()
    
    except KeyboardInterrupt:
        logger.info("\nExiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
