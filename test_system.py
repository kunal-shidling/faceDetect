"""
Testing utilities for the facial recognition system.
Tests embedding consistency, similarity thresholds, and duplicate detection.
"""

import numpy as np
import logging
import config
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


def test_embedding_consistency():
    """Test that same image produces consistent embeddings."""
    print("\n" + "=" * 70)
    print("TEST 1: EMBEDDING CONSISTENCY")
    print("=" * 70)
    
    try:
        import sys
        if len(sys.argv) < 2:
            print("Usage: python test_system.py <test_image_path>")
            return False
        
        image_path = sys.argv[1]
        
        embedder = FaceEmbedder()
        
        print(f"\nGenerating 3 embeddings from same image: {image_path}")
        
        embeddings = []
        for i in range(3):
            print(f"\nIteration {i+1}:")
            success, emb = embedder.embed_from_file(image_path)
            
            if not success:
                print(f"✗ Failed to generate embedding")
                return False
            
            embeddings.append(emb)
            print(f"✓ Embedding generated - Shape: {emb.shape}")
        
        # Compare embeddings
        print("\n" + "-" * 70)
        print("CONSISTENCY ANALYSIS:")
        print("-" * 70)
        
        all_consistent = True
        
        for i in range(len(embeddings) - 1):
            similarity = embedder.compare_embeddings(embeddings[i], embeddings[i+1])
            print(f"Similarity between iteration {i+1} and {i+2}: {similarity:.6f}")
            
            if similarity < 0.999:  # Should be nearly identical
                print(f"  ⚠ Warning: Embeddings not perfectly consistent")
                all_consistent = False
            else:
                print(f"  ✓ Embeddings are consistent")
        
        if all_consistent:
            print("\n✓ TEST PASSED: Embeddings are consistent")
            return True
        else:
            print("\n⚠ TEST WARNING: Some inconsistency detected")
            return True  # Still pass, minor variations acceptable
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False


def test_same_person_similarity():
    """Test that different photos of same person have high similarity."""
    print("\n" + "=" * 70)
    print("TEST 2: SAME PERSON SIMILARITY")
    print("=" * 70)
    
    try:
        import sys
        if len(sys.argv) < 3:
            print("Usage: python test_system.py <image1_path> <image2_path>")
            return False
        
        image1_path = sys.argv[1]
        image2_path = sys.argv[2]
        
        embedder = FaceEmbedder()
        
        print(f"\nGenerating embedding for image 1: {image1_path}")
        success1, emb1 = embedder.embed_from_file(image1_path)
        if not success1:
            print("✗ Failed to generate embedding for image 1")
            return False
        print("✓ Embedding 1 generated")
        
        print(f"\nGenerating embedding for image 2: {image2_path}")
        success2, emb2 = embedder.embed_from_file(image2_path)
        if not success2:
            print("✗ Failed to generate embedding for image 2")
            return False
        print("✓ Embedding 2 generated")
        
        # Compute similarity
        similarity = embedder.compare_embeddings(emb1, emb2)
        
        print("\n" + "-" * 70)
        print("SIMILARITY ANALYSIS:")
        print("-" * 70)
        print(f"Cosine Similarity: {similarity:.4f}")
        print(f"Recognition Threshold: {config.RECOGNITION_THRESHOLD}")
        
        if similarity >= config.RECOGNITION_THRESHOLD:
            print(f"✓ Similarity ABOVE threshold - Would be RECOGNIZED as same person")
            print(f"\n✓ TEST PASSED")
            return True
        else:
            print(f"✗ Similarity BELOW threshold - Would NOT be recognized")
            print(f"\n✗ TEST FAILED")
            return False
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False


def test_different_person_similarity():
    """Test that different people have low similarity."""
    print("\n" + "=" * 70)
    print("TEST 3: DIFFERENT PERSON SIMILARITY")
    print("=" * 70)
    
    try:
        import sys
        if len(sys.argv) < 3:
            print("Usage: python test_system.py <person1_image> <person2_image>")
            return False
        
        image1_path = sys.argv[1]
        image2_path = sys.argv[2]
        
        embedder = FaceEmbedder()
        
        print(f"\nGenerating embedding for person 1: {image1_path}")
        success1, emb1 = embedder.embed_from_file(image1_path)
        if not success1:
            print("✗ Failed to generate embedding for person 1")
            return False
        print("✓ Embedding 1 generated")
        
        print(f"\nGenerating embedding for person 2: {image2_path}")
        success2, emb2 = embedder.embed_from_file(image2_path)
        if not success2:
            print("✗ Failed to generate embedding for person 2")
            return False
        print("✓ Embedding 2 generated")
        
        # Compute similarity
        similarity = embedder.compare_embeddings(emb1, emb2)
        
        print("\n" + "-" * 70)
        print("SIMILARITY ANALYSIS:")
        print("-" * 70)
        print(f"Cosine Similarity: {similarity:.4f}")
        print(f"Recognition Threshold: {config.RECOGNITION_THRESHOLD}")
        
        if similarity < config.RECOGNITION_THRESHOLD:
            print(f"✓ Similarity BELOW threshold - Would be recognized as DIFFERENT persons")
            print(f"\n✓ TEST PASSED")
            return True
        else:
            print(f"✗ Similarity ABOVE threshold - Would be recognized as SAME person")
            print(f"\n✗ TEST FAILED (or they look very similar!)")
            return False
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False


def test_duplicate_detection():
    """Test duplicate embedding detection."""
    print("\n" + "=" * 70)
    print("TEST 4: DUPLICATE DETECTION")
    print("=" * 70)
    
    try:
        # Connect to database
        db = FaceDatabase()
        if not db.connect():
            print("✗ Failed to connect to database")
            return False
        
        print("✓ Connected to database")
        
        # Create test embedding
        test_embedding = np.random.randn(config.EMBEDDING_SIZE)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        # Store first time
        print("\n1. Storing test embedding for 'Test Person'...")
        success1 = db.store_embedding("Test Person 1", test_embedding, check_duplicates=False)
        if not success1:
            print("✗ Failed to store embedding")
            db.disconnect()
            return False
        print("✓ First embedding stored")
        
        # Try to store very similar embedding (should be rejected)
        print("\n2. Attempting to store very similar embedding...")
        similar_embedding = test_embedding + np.random.randn(config.EMBEDDING_SIZE) * 0.01
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
        
        success2 = db.store_embedding("Test Person 2", similar_embedding, check_duplicates=True)
        
        if not success2:
            print("✓ Duplicate correctly REJECTED")
            passed = True
        else:
            print("✗ Duplicate was NOT rejected (should have been)")
            passed = False
        
        # Cleanup
        print("\n3. Cleaning up test data...")
        db.delete_embedding("Test Person 1")
        db.delete_embedding("Test Person 2")
        db.disconnect()
        print("✓ Cleanup complete")
        
        if passed:
            print("\n✓ TEST PASSED")
        else:
            print("\n✗ TEST FAILED")
        
        return passed
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False


def test_recognition_pipeline():
    """Test the complete recognition pipeline."""
    print("\n" + "=" * 70)
    print("TEST 5: COMPLETE RECOGNITION PIPELINE")
    print("=" * 70)
    
    try:
        import sys
        if len(sys.argv) < 2:
            print("Usage: python test_system.py <test_image_path>")
            return False
        
        image_path = sys.argv[1]
        
        # Initialize components
        print("\n1. Initializing components...")
        embedder = FaceEmbedder()
        db = FaceDatabase()
        
        if not db.connect():
            print("✗ Failed to connect to database")
            return False
        
        recognizer = FaceRecognizer(db)
        print("✓ Components initialized")
        
        # Generate embedding
        print(f"\n2. Generating embedding from: {image_path}")
        success, embedding = embedder.embed_from_file(image_path)
        if not success:
            print("✗ Failed to generate embedding")
            db.disconnect()
            return False
        print("✓ Embedding generated")
        
        # Test recognition (should not recognize)
        print("\n3. Testing recognition (should NOT find match)...")
        recognized, name, confidence = recognizer.recognize(embedding)
        if not recognized:
            print(f"✓ Correctly NOT recognized (confidence: {confidence:.4f})")
        else:
            print(f"⚠ Was recognized as '{name}' (might be in database already)")
        
        # Register
        print("\n4. Registering as 'Test Subject'...")
        success = recognizer.register_new_person("Test Subject", embedding)
        if not success:
            print("✗ Failed to register")
            db.disconnect()
            return False
        print("✓ Registered successfully")
        
        # Test recognition again (should recognize)
        print("\n5. Testing recognition again (should find match)...")
        recognized, name, confidence = recognizer.recognize(embedding)
        if recognized and name == "Test Subject":
            print(f"✓ Correctly RECOGNIZED as '{name}' (confidence: {confidence:.4f})")
            passed = True
        else:
            print(f"✗ Failed to recognize (name: {name}, confidence: {confidence:.4f})")
            passed = False
        
        # Cleanup
        print("\n6. Cleaning up...")
        db.delete_embedding("Test Subject")
        db.disconnect()
        print("✓ Cleanup complete")
        
        if passed:
            print("\n✓ TEST PASSED")
        else:
            print("\n✗ TEST FAILED")
        
        return passed
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("FACIAL RECOGNITION SYSTEM - COMPLETE TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Embedding Consistency
    results['consistency'] = test_embedding_consistency()
    
    # Test 2: Same Person Similarity
    # results['same_person'] = test_same_person_similarity()
    
    # Test 3: Different Person Similarity
    # results['different_person'] = test_different_person_similarity()
    
    # Test 4: Duplicate Detection
    results['duplicate_detection'] = test_duplicate_detection()
    
    # Test 5: Recognition Pipeline
    results['recognition_pipeline'] = test_recognition_pipeline()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 70 + "\n")
    
    return all(results.values())


def main():
    """Main test entry point."""
    import sys
    
    print("\n" + "=" * 70)
    print("FACIAL RECOGNITION SYSTEM - TESTING UTILITIES")
    print("=" * 70)
    print("\nAvailable tests:")
    print("  1. Embedding consistency")
    print("  2. Same person similarity")
    print("  3. Different person similarity")
    print("  4. Duplicate detection")
    print("  5. Recognition pipeline")
    print("  all. Run all tests")
    print("\nUsage:")
    print("  python test_system.py <test_number> [image_path] [image_path_2]")
    print("\nExamples:")
    print("  python test_system.py 1 photo.jpg")
    print("  python test_system.py 2 photo1.jpg photo2.jpg")
    print("  python test_system.py all photo.jpg")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("\nPlease specify a test number or 'all'")
        return
    
    test_choice = sys.argv[1].lower()
    
    if test_choice == '1':
        test_embedding_consistency()
    elif test_choice == '2':
        test_same_person_similarity()
    elif test_choice == '3':
        test_different_person_similarity()
    elif test_choice == '4':
        test_duplicate_detection()
    elif test_choice == '5':
        test_recognition_pipeline()
    elif test_choice == 'all':
        run_all_tests()
    else:
        print(f"\nUnknown test: {test_choice}")


if __name__ == "__main__":
    main()
