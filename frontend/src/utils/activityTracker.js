// Activity Tracker Utility

export const trackActivity = async (feature) => {
  const token = localStorage.getItem("token");
  
  if (!token) return;

  try {
    await fetch("http://localhost:5000/track-activity", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      },
      body: JSON.stringify({ feature }),
    });
  } catch (error) {
    console.error("Activity tracking error:", error);
  }
};
