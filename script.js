document.getElementById("home-btn").addEventListener("click", function(e) {
  e.preventDefault();

  // Show the exoplanet popup
  const popup = document.getElementById("exoplanet-popup");
  popup.classList.remove("hidden");
  popup.classList.add("show");
  
  // Prevent body scroll
  document.body.classList.add("popup-open");
});

// Close popup functionality removed - only back button remains

// Close popup when clicking outside
document.getElementById("exoplanet-popup").addEventListener("click", function(e) {
  if (e.target === this) {
    this.classList.remove("show");
    this.classList.add("hidden");
    
    // Restore body scroll
    document.body.classList.remove("popup-open");
  }
});

// Close popup with Escape key
document.addEventListener("keydown", function(e) {
  if (e.key === "Escape") {
    const popup = document.getElementById("exoplanet-popup");
    if (popup.classList.contains("show")) {
      popup.classList.remove("show");
      popup.classList.add("hidden");
      
      // Restore body scroll
      document.body.classList.remove("popup-open");
    }
  }
});

// Back button functionality
document.getElementById("back-btn").addEventListener("click", function() {
  const popup = document.getElementById("exoplanet-popup");
  popup.classList.remove("show");
  popup.classList.add("hidden");
  
  // Restore body scroll
  document.body.classList.remove("popup-open");
});