import React, { useState } from "react";
import "./AlphabetLearning.css";
import WordLearning from "./WordLearning";

// Ethiopian Amharic alphabet - 33 base characters (ፊደል) + አመሰራረት + ምሳሌ
const ethiopianAlphabet = [
  { id: 1, char: "ሀ", name: "ሀ (Ha)", latinName: "Ha" },
  { id: 2, char: "ለ", name: "ለ (Le)", latinName: "Le" },
  { id: 3, char: "ሐ", name: "ሐ (Ḥa)", latinName: "Hha" },
  { id: 4, char: "መ", name: "መ (Me)", latinName: "Me" },
  { id: 5, char: "ሠ", name: "ሠ (Se)", latinName: "Sse" },
  { id: 6, char: "ረ", name: "ረ (Re)", latinName: "Re" },
  { id: 7, char: "ሰ", name: "ሰ (Se)", latinName: "Se" },
  { id: 8, char: "ሸ", name: "ሸ (She)", latinName: "She" },
  { id: 9, char: "ቀ", name: "ቀ (Qe)", latinName: "Qe" },
  { id: 10, char: "በ", name: "በ (Be)", latinName: "Be" },
  { id: 11, char: "ተ", name: "ተ (Te)", latinName: "Te" },
  { id: 12, char: "ቸ", name: "ቸ (Che)", latinName: "Che" },
  { id: 13, char: "ኀ", name: "ኀ (Ḫa)", latinName: "Xha" },
  { id: 14, char: "ነ", name: "ነ (Ne)", latinName: "Ne" },
  { id: 15, char: "ኘ", name: "ኘ (Ñe)", latinName: "Nye" },
  { id: 16, char: "አ", name: "አ (A)", latinName: "A" },
  { id: 17, char: "ከ", name: "ከ (Ke)", latinName: "Ke" },
  { id: 18, char: "ኸ", name: "ኸ (Ḫe)", latinName: "Khe" },
  { id: 19, char: "ወ", name: "ወ (We)", latinName: "We" },
  { id: 20, char: "ዐ", name: "ዐ (ʿA)", latinName: "Aa" },
  { id: 21, char: "ዘ", name: "ዘ (Ze)", latinName: "Ze" },
  { id: 22, char: "ዠ", name: "ዠ (Zhe)", latinName: "Zhe" },
  { id: 23, char: "የ", name: "የ (Ye)", latinName: "Ye" },
  { id: 24, char: "ደ", name: "ደ (De)", latinName: "De" },
  { id: 25, char: "ጀ", name: "ጀ (Je)", latinName: "Je" },
  { id: 26, char: "ገ", name: "ገ (Ge)", latinName: "Ge" },
  { id: 27, char: "ጠ", name: "ጠ (Ṭe)", latinName: "Tte" },
  { id: 28, char: "ጨ", name: "ጨ (Č̣e)", latinName: "Cche" },
  { id: 29, char: "ጰ", name: "ጰ (P̣e)", latinName: "Ppe" },
  { id: 30, char: "ጸ", name: "ጸ (Ṣe)", latinName: "Tse" },
  { id: 31, char: "ፀ", name: "ፀ (Ṣ́e)", latinName: "Tsse" },
  { id: 32, char: "ፈ", name: "ፈ (Fe)", latinName: "Fe" },
  { id: 33, char: "ፐ", name: "ፐ (Pe)", latinName: "Pe" },
  { id: 34, char: "📋", name: "አመሰራረት", latinName: "Direction", isSpecial: true },
  { id: 35, char: "🎬", name: "ምሳሌ", latinName: "Examples", isExample: true },
];

// Video examples
const videoExamples = [
  { id: "ha", char: "ሀ", name: "ሀ ቤተሰብ (Ha Family)", videoFile: "ha.mp4" },
  { id: "le", char: "ለ", name: "ለ ቤተሰብ (Le Family)", videoFile: "le.mp4" },
];

// American ASL alphabet - A to Z
const americanAlphabet = [
  { id: 1, char: "A", name: "A" },
  { id: 2, char: "B", name: "B" },
  { id: 3, char: "C", name: "C" },
  { id: 4, char: "D", name: "D" },
  { id: 5, char: "E", name: "E" },
  { id: 6, char: "F", name: "F" },
  { id: 7, char: "G", name: "G" },
  { id: 8, char: "H", name: "H" },
  { id: 9, char: "I", name: "I" },
  { id: 10, char: "J", name: "J" },
  { id: 11, char: "K", name: "K" },
  { id: 12, char: "L", name: "L" },
  { id: 13, char: "M", name: "M" },
  { id: 14, char: "N", name: "N" },
  { id: 15, char: "O", name: "O" },
  { id: 16, char: "P", name: "P" },
  { id: 17, char: "Q", name: "Q" },
  { id: 18, char: "R", name: "R" },
  { id: 19, char: "S", name: "S" },
  { id: 20, char: "T", name: "T" },
  { id: 21, char: "U", name: "U" },
  { id: 22, char: "V", name: "V" },
  { id: 23, char: "W", name: "W" },
  { id: 24, char: "X", name: "X" },
  { id: 25, char: "Y", name: "Y" },
  { id: 26, char: "Z", name: "Z" },
];

function AlphabetLearning() {
  const [selectedLanguage, setSelectedLanguage] = useState("ethiopian");
  const [selectedLetter, setSelectedLetter] = useState(null);
  const [showExamples, setShowExamples] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [learningMode, setLearningMode] = useState("letters"); // "letters" or "words"

  const alphabet = selectedLanguage === "ethiopian" ? ethiopianAlphabet : americanAlphabet;

  const handleLetterClick = (letter) => {
    if (letter.isExample) {
      setShowExamples(true);
    } else {
      setSelectedLetter(letter);
    }
  };

  const closeModal = () => {
    setSelectedLetter(null);
    setShowExamples(false);
    setSelectedVideo(null);
  };

  const handleVideoSelect = (video) => {
    setSelectedVideo(video);
  };

  const getImagePath = (letter) => {
    if (selectedLanguage === "ethiopian") {
      // Use .png for አመሰራረት (id 34)
      const ext = letter.id === 34 ? 'png' : 'jpg';
      return `/alphabets/ethiopian/${letter.id}.${ext}`;
    }
    return `/alphabets/american/${letter.char.toLowerCase()}.jpg`;
  };

  return (
    <div className="alphabet-learning">
      <div className="language-selector">
        <button
          className={`lang-btn ${selectedLanguage === "ethiopian" ? "active" : ""}`}
          onClick={() => setSelectedLanguage("ethiopian")}
        >
          🇪🇹 Ethiopian (አማርኛ)
        </button>
        <button
          className={`lang-btn ${selectedLanguage === "american" ? "active" : ""}`}
          onClick={() => setSelectedLanguage("american")}
        >
          🇺🇸 American (ASL)
        </button>
      </div>

      {/* Learning Mode Tabs - Only show for Ethiopian */}
      {selectedLanguage === "ethiopian" && (
        <div className="learning-mode-tabs">
          <button
            className={`mode-tab ${learningMode === "letters" ? "active" : ""}`}
            onClick={() => setLearningMode("letters")}
          >
            📝 ፊደላት (Letters)
          </button>
          <button
            className={`mode-tab ${learningMode === "words" ? "active" : ""}`}
            onClick={() => setLearningMode("words")}
          >
            📖 ቃላት (Words)
          </button>
        </div>
      )}

      {/* Show Words component if in words mode */}
      {selectedLanguage === "ethiopian" && learningMode === "words" ? (
        <WordLearning />
      ) : (
        <>
          <h2 className="alphabet-title">
            {selectedLanguage === "ethiopian" ? "የአማርኛ ፊደላት - Amharic Alphabet" : "ASL Alphabet"}
          </h2>

      <div className="alphabet-grid">
        {alphabet.map((letter) => (
          <div
            key={letter.id}
            className={`letter-box ${letter.isSpecial ? 'special-box' : ''} ${letter.isExample ? 'example-box-style' : ''}`}
            onClick={() => handleLetterClick(letter)}
          >
            <div className="letter-char">{letter.char}</div>
            <div className="letter-name">
              {selectedLanguage === "ethiopian" ? letter.latinName : letter.name}
            </div>

          </div>
        ))}
      </div>

      {/* Letter Modal */}
      {selectedLetter && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className={`modal-content ${selectedLetter.isSpecial ? 'fullscreen-image' : ''}`} onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeModal}>×</button>
            <div className="modal-letter-display">
              <h1 className="modal-char">{selectedLetter.char}</h1>
              <h2 className="modal-name">{selectedLetter.name}</h2>
              {selectedLanguage === "ethiopian" && !selectedLetter.isSpecial && (
                <p className="modal-latin">({selectedLetter.latinName})</p>
              )}
              {selectedLetter.isSpecial && (
                <p className="modal-description">የፊደል አመሰራረት አቅጣጫ</p>
              )}
            </div>
            
            <div className="modal-media-container">
              <div className="modal-image-container">
                <img
                  src={getImagePath(selectedLetter)}
                  alt={selectedLetter.name}
                  className="modal-image"
                  onError={(e) => {
                    e.target.style.display = 'none';
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Examples Modal */}
      {showExamples && !selectedVideo && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content examples-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeModal}>×</button>
            <div className="modal-letter-display">
              <h1 className="modal-char">🎬</h1>
              <h2 className="modal-name">ምሳሌ (Examples)</h2>
              <p className="modal-description">የፊደል ቤተሰብ ምሳሌዎች</p>
            </div>
            
            <div className="examples-grid">
              {videoExamples.map((video) => (
                <div 
                  key={video.id} 
                  className="example-box"
                  onClick={() => handleVideoSelect(video)}
                >
                  <div className="example-char">{video.char}</div>
                  <div className="example-name">{video.name}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Video Player Modal */}
      {selectedVideo && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content video-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeModal}>×</button>
            <div className="modal-letter-display">
              <h1 className="modal-char">{selectedVideo.char}</h1>
              <h2 className="modal-name">{selectedVideo.name}</h2>
            </div>
            
            <div className="modal-video-container">
              <video
                controls
                autoPlay
                className="modal-video"
              >
                <source src={`/alphabets/ethiopian/${selectedVideo.videoFile}`} type="video/mp4" />
                Your browser does not support video.
              </video>
            </div>
            
            <button className="back-btn" onClick={() => setSelectedVideo(null)}>
              ← ወደ ምሳሌዎች ተመለስ
            </button>
          </div>
        </div>
      )}
        </>
      )}
    </div>
  );
}

export default AlphabetLearning;
