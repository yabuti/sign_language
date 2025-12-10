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
  const [selectedModel, setSelectedModel] = useState("asl_alphabet");

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
      
      try {
        const formData = new FormData();
        formData.append("image", file);
        formData.append("model", selectedModel);

        const response = await fetch("http://localhost:5000/predict", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        
        if (data.success) {
          let resultText = `✅ ${data.text} (${data.confidence}% confidence)\n`;
          resultText += `📊 Model: ${data.model_used}\n\n`;
          
          if (data.top_3 && data.top_3.length > 0) {
            resultText += "Top 3 Predictions:\n";
            data.top_3.forEach((pred, idx) => {
              resultText += `${idx + 1}. ${pred.label} - ${pred.confidence}%\n`;
            });
          }
          setTranslatedText(resultText);
        } else {
          setTranslatedText(data.text || "Error processing image");
        }
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

      {/* Language Selection */}
      <div className="language-selector">
        <button
          className={`lang-btn ${selectedModel === "asl_alphabet" ? "active" : ""}`}
          onClick={() => setSelectedModel("asl_alphabet")}
        >
          🇺🇸 ASL Alphabet (A-Z)
        </button>
        <button
          className={`lang-btn ${selectedModel === "ethiopian" ? "active" : ""}`}
          onClick={() => setSelectedModel("ethiopian")}
        >
          🇪🇹 Ethiopian Signs
        </button>
      </div>

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
          <p style={{ whiteSpace: 'pre-line' }}>{translatedText}</p>
        </div>
      </div>
    </motion.div>
  );
}

export default ImageTranslator;
