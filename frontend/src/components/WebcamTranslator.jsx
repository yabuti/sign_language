import React, { useRef, useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FiVideo, FiVideoOff } from "react-icons/fi";
import Webcam from "react-webcam";
import { trackActivity } from "../utils/activityTracker";
import "./WebcamTranslator.css";

function WebcamTranslator() {
  const webcamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [translatedText, setTranslatedText] = useState("Your text will appear here");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [debugFrameUrl, setDebugFrameUrl] = useState(null);
  const [frameInfo, setFrameInfo] = useState("");
  const recordedChunksRef = useRef([]);

  // Ethiopian landmark model (uses MediaPipe hand detection)
  const selectedModel = "eth_landmark";

  // Track activity when component mounts
  useEffect(() => {
    trackActivity("webcam");
  }, []);

  // Start recording video
  const startRecording = () => {
    if (!webcamRef.current || !webcamRef.current.stream) return;

    recordedChunksRef.current = [];
    const mediaRecorder = new MediaRecorder(webcamRef.current.stream, {
      mimeType: "video/webm"
    });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = sendVideoToBackend;

    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.start();
    setIsRecording(true);
    setTranslatedText("🎥 Recording 4 seconds... Hold your sign steady!");

    // Auto-stop after 4 seconds (more frames = more stable prediction)
    setTimeout(() => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
        stopRecording();
      }
    }, 4000);
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setTranslatedText("⏳ Processing video...");
    }
  };

  // Send recorded video to backend
  const sendVideoToBackend = async () => {
    if (recordedChunksRef.current.length === 0) {
      setTranslatedText("❌ No video recorded");
      return;
    }

    try {
      setIsProcessing(true);
      const blob = new Blob(recordedChunksRef.current, { type: "video/webm" });
      
      const formData = new FormData();
      formData.append("video", blob, "recording.webm");
      formData.append("model", selectedModel);

      const response = await fetch("http://localhost:5000/predict-webcam", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      
      if (data.success) {
        // Check if prediction is stable
        const isStable = data.is_stable !== false;
        const stabilityIcon = isStable ? "✅" : "⚠️";
        const stabilityNote = isStable ? "" : " (Unstable - try again)";
        
        let resultText = `${stabilityIcon} ${data.text} (${data.confidence}% confidence)${stabilityNote}\n`;
        resultText += `📊 Agreement: ${data.agreement || "N/A"}\n\n`;
        
        if (data.top_3 && data.top_3.length > 0) {
          resultText += "Top 3 Predictions:\n";
          data.top_3.forEach((pred, idx) => {
            resultText += `${idx + 1}. ${pred.label} - ${pred.confidence}%\n`;
          });
        }
        
        setTranslatedText(resultText);
        
        // Load debug frame with timestamp to prevent caching
        setDebugFrameUrl(`http://localhost:5000/debug-frame?t=${Date.now()}`);
        setFrameInfo(data.frame_info || "");
      } else {
        setTranslatedText(data.error || "Error processing video");
        setDebugFrameUrl(null);
        setFrameInfo("");
      }
    } catch (error) {
      console.error("Error sending video:", error);
      setTranslatedText("❌ Error processing video. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <motion.div
      className="section-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
    >
      <h2>🎥 Webcam → Text (Live)</h2>

      {/* Language Selection */}
      <div className="language-selector">
        <h3>🇪🇹 Ethiopian Sign Language</h3>
      </div>

      {/* Main Left-Right Containers */}
      <div className="main-container">
        {/* Left container: Webcam + Buttons */}
        <div className="left-container">
          {cameraOn ? (
            <div className="webcam-wrapper">
              {!cameraReady && (
                <div className="webcam-loading">
                  <div className="spinner-large"></div>
                  <p>Connecting to camera...</p>
                </div>
              )}
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="webcam"
                onUserMedia={() => setCameraReady(true)}
                onUserMediaError={() => {
                  setCameraReady(false);
                  setCameraOn(false);
                  alert("Failed to access camera. Please check permissions.");
                }}
                style={{ display: cameraReady ? 'block' : 'none' }}
              />
              {isProcessing && (
                <div className="processing-indicator">
                  <div className="spinner"></div>
                  <span>Processing...</span>
                </div>
              )}
            </div>
          ) : (
            <div className="webcam-off">
              <FiVideoOff size={60} color="#a78bfa" />
              <p className="camera-off-text">Camera is Off</p>
              <p className="camera-off-subtitle">Click below to start</p>
            </div>
          )}

          {/* Camera Controls under webcam */}
          <div className="camera-controls">
            {!cameraOn ? (
              <button onClick={() => { setCameraOn(true); setCameraReady(false); }}>
                <FiVideo size={20} />
                <span>Turn Camera On</span>
              </button>
            ) : (
              <>
                <button 
                  onClick={startRecording} 
                  disabled={isRecording || isProcessing}
                  className="capture-btn"
                >
                  <FiVideo size={20} />
                  <span>{isRecording ? "Recording... Hold steady!" : "Capture Sign (4s)"}</span>
                </button>
                <button onClick={() => setCameraOn(false)} className="off-btn">
                  <FiVideoOff size={20} />
                  <span>Turn Camera Off</span>
                </button>
              </>
            )}
          </div>
        </div>

        {/* Right container: Translated Text */}
        <div className="right-container">
          <h3>📝 Translated Text</h3>
          <div className="text-display">
            <p>{translatedText}</p>
          </div>
          
          {/* Debug Frame - shows the exact frame used for prediction */}
          {debugFrameUrl && (
            <div className="debug-frame-container">
              <h4>🖼️ Frame Used for Detection</h4>
              <p className="frame-info">{frameInfo}</p>
              <img 
                src={debugFrameUrl} 
                alt="Debug frame" 
                className="debug-frame"
                onError={() => setDebugFrameUrl(null)}
              />
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

export default WebcamTranslator;
