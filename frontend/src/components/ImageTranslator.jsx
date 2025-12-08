import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FiUpload } from "react-icons/fi";
import { trackActivity } from "../utils/activityTracker";
import "./ImageTranslator.css";

function ImageTranslator() {
  const [image, setImage] = useState(null);
  const [translatedText, setTranslatedText] = useState(
    "Your translated text will appear here"
  );
  const [isProcessing, setIsProcessing] = useState(false);

  // Track activity when component mounts
  useEffect(() => {
    trackActivity("image");
  }, []);

  // Handle image upload
  const handleImageUpload = async (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage(URL.createObjectURL(file));
      setIsProcessing(true);
      setTranslatedText("Processing image...");
      
      // Here you can send image to backend for processing
      try {
        const formData = new FormData();
        formData.append("image", file);

        const response = await fetch("http://localhost:5000/predict", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        setTranslatedText(data.text || "Translation complete!");
      } catch (error) {
        console.error("Error processing image:", error);
        setTranslatedText("Error processing image. Please try again.");
      } finally {
        setIsProcessing(false);
      }
    }
  };

  return (
    <motion.div
      className="image-section-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
    >
      <h2>📷 Image → Text</h2>

      <div className="upload-area">
        <label htmlFor="file-upload" className="upload-label">
          <FiUpload size={40} />
          <p className="upload-text">Click to upload or drag and drop</p>
          <p className="upload-subtext">PNG, JPG, JPEG up to 10MB</p>
        </label>
        <input 
          id="file-upload"
          type="file" 
          accept="image/*" 
          onChange={handleImageUpload}
          style={{ display: 'none' }}
        />
      </div>

      {image && (
        <div className="image-preview">
          <img src={image} alt="Uploaded" />
          {isProcessing && (
            <div className="processing-overlay">
              <div className="spinner-large"></div>
              <p>Processing...</p>
            </div>
          )}
        </div>
      )}

      <div className="text-output">
        <h3>📝 Translated Text</h3>
        <div className="text-display">
          <p>{translatedText}</p>
        </div>
      </div>
    </motion.div>
  );
}

export default ImageTranslator;
