import React, { useState } from "react";
import "./WordLearning.css";

// Ethiopian Sign Language words - matching ETH_LABELS from the model
const amharicWords = [
  { id: 0, word: "ሂድ", latinName: "Go" },
  { id: 1, word: "ህመም", latinName: "Pain/Sick" },
  { id: 2, word: "መንደር", latinName: "Village" },
  { id: 3, word: "ምግብ", latinName: "Food" },
  { id: 4, word: "ሰላም", latinName: "Hello/Peace" },
  { id: 5, word: "ቀለም", latinName: "Color/Pen" },
  { id: 6, word: "አመሰግናለሁ", latinName: "Thank You" },
  { id: 7, word: "አቁም", latinName: "Stop" },
  { id: 8, word: "አዎን", latinName: "Yes" },
  { id: 9, word: "እባክህ", latinName: "Please" },
  { id: 10, word: "እንደገና", latinName: "Again" },
  { id: 11, word: "እገዛ", latinName: "Help" },
  { id: 12, word: "እግር", latinName: "Foot/Leg" },
  { id: 13, word: "ውሃ", latinName: "Water" },
  { id: 14, word: "ይቅርታ", latinName: "Sorry" },
  { id: 15, word: "ድምፅ", latinName: "Sound/Voice" },
  { id: 16, word: "ድንጋይ", latinName: "Stone" },
  { id: 17, word: "ግራ", latinName: "Left" },
  { id: 18, word: "ጥሩ", latinName: "Good" },
  { id: 19, word: "ጨምር", latinName: "Add" },
];

function WordLearning() {
  const [selectedWord, setSelectedWord] = useState(null);

  const handleWordClick = (word) => {
    setSelectedWord(word);
  };

  const closeModal = () => {
    setSelectedWord(null);
  };

  const getVideoPath = (word) => {
    return `/words/ethiopian/${word.id}.mp4`;
  };

  return (
    <div className="word-learning">
      <h2 className="word-title">የአማርኛ ቃላት - Amharic Words</h2>
      <p className="word-subtitle">ቃላትን ጠቅ ያድርጉ ቪዲዮ ለማየት</p>

      <div className="word-grid">
        {amharicWords.map((word) => (
          <div
            key={word.id}
            className="word-box"
            onClick={() => handleWordClick(word)}
          >
            <div className="word-char">{word.word}</div>
            <div className="word-name">{word.latinName}</div>
            <div className="word-icons">
              <span className="media-icon">🎬</span>
            </div>
          </div>
        ))}
      </div>

      {/* Word Modal - Video Only */}
      {selectedWord && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content word-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeModal}>×</button>
            
            <div className="modal-word-display">
              <h1 className="modal-word">{selectedWord.word}</h1>
              <p className="modal-word-latin">{selectedWord.latinName}</p>
            </div>

            {/* Video Display */}
            <div className="modal-media">
              <div className="modal-video-container">
                <video
                  key={selectedWord.id}
                  controls
                  autoPlay
                  loop
                  muted
                  className="modal-video"
                >
                  <source src={getVideoPath(selectedWord)} type="video/mp4" />
                  Your browser does not support video.
                </video>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default WordLearning;
