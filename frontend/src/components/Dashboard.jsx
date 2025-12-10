import React, { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Navbar from "./Navbar";
import WebcamTranslator from "./WebcamTranslator";
import ImageTranslator from "./ImageTranslator";
import VoiceTranslator from "./VoiceTranslator";
import AlphabetLearning from "./AlphabetLearning";
import "../index.css";

function Dashboard() {
  const [section, setSection] = useState("webcam");

  return (
    <div className="app-container">
      <Navbar onSelect={setSection} />
      <AnimatePresence mode="wait">
        {section === "webcam" && (
          <motion.div
            key="webcam"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
          >
            <WebcamTranslator />
          </motion.div>
        )}
        {section === "image" && (
          <motion.div
            key="image"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
          >
            <ImageTranslator />
          </motion.div>
        )}
        {section === "voice" && (
          <motion.div
            key="voice"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
          >
            <VoiceTranslator />
          </motion.div>
        )}
        {section === "learn" && (
          <motion.div
            key="learn"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
          >
            <AlphabetLearning />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default Dashboard;
