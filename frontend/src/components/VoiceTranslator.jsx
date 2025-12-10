import React, { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { FiMic, FiSquare } from "react-icons/fi";
import { trackActivity } from "../utils/activityTracker";
import "./VoiceTranslator.css";

function VoiceTranslator() {
  const [recording, setRecording] = useState(false);
  const [language, setLanguage] = useState("en");
  const [originalText, setOriginalText] = useState("Your speech will appear here");

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Track activity when component mounts
  useEffect(() => {
    trackActivity("voice");
  }, []);

  const languageMap = {
    en: "en-US",
    am: "am-ET",
  };

  const startRecording = () => {
    // Check if browser supports speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      setOriginalText("❌ Speech recognition not supported. Please use Chrome or Edge browser.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = languageMap[language];
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    let finalTranscript = "";

    recognition.onstart = () => {
      setRecording(true);
      setOriginalText("🎤 Listening... Please speak now!");
      console.log("Speech recognition started");
    };

    recognition.onresult = (event) => {
      let interimTranscript = "";
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript + " ";
        } else {
          interimTranscript += transcript;
        }
      }
      
      const displayText = finalTranscript + interimTranscript;
      setOriginalText(displayText || "🎤 Listening... Please speak now!");
      console.log("Transcript:", displayText);
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      setRecording(false);
      
      let errorMessage = "❌ Error: ";
      switch(event.error) {
        case 'no-speech':
          errorMessage += "No speech detected. Please speak louder or check your microphone.";
          break;
        case 'audio-capture':
          errorMessage += "Microphone not found. Please connect a microphone.";
          break;
        case 'not-allowed':
          errorMessage += "Microphone permission denied. Please allow microphone access.";
          break;
        case 'network':
          errorMessage += "Network error. Please check your internet connection.";
          break;
        default:
          errorMessage += event.error;
      }
      setOriginalText(errorMessage);
    };

    recognition.onend = () => {
      console.log("Speech recognition ended");
      setRecording(false);
      
      if (finalTranscript.trim() === "") {
        setOriginalText("⚠️ No speech detected. Tips:\n• Speak clearly and loudly\n• Check microphone permissions\n• Make sure microphone is working");
      } else {
        setOriginalText(finalTranscript.trim());
      }
    };

    mediaRecorderRef.current = recognition;
    
    try {
      recognition.start();
      console.log("Starting speech recognition for language:", languageMap[language]);
    } catch (error) {
      console.error("Failed to start recognition:", error);
      setOriginalText("❌ Failed to start recording. Please try again.");
      setRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
  };

  return (
    <motion.div
      className="voice-translator-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
    >
      <h2>🎤 Voice → Text</h2>

      <select
        value={language}
        onChange={(e) => {
          setLanguage(e.target.value);
          setOriginalText("Your speech will appear here");
        }}
        className="language-select"
      >
        <option value="en">🇺🇸 English</option>
        <option value="am">🇪🇹 Amharic</option>
      </select>

      <div className={`mic-icon ${recording ? 'recording' : ''}`}>
        <FiMic size={60} color={recording ? "#ef4444" : "#8B5CF6"} />
      </div>

      <div className="buttons">
        <button onClick={startRecording} disabled={recording} className="start-btn">
          <FiMic size={20} />
          <span>Start Recording</span>
        </button>
        <button onClick={stopRecording} disabled={!recording} className="stop-btn">
          <FiSquare size={20} />
          <span>Stop Recording</span>
        </button>
      </div>

      <div className="text-output">
        <h3>
          📝 {language === "en" ? "English" : "Amharic"} Text
        </h3>
        <div className="text-display">
          <p>{originalText}</p>
        </div>
      </div>
    </motion.div>
  );
}

export default VoiceTranslator;
