import "./App.css"
import { useState } from "react"
import { FETCH_URL } from "./constants.js"

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [imageUrl, setImageUrl] = useState(null)
  const [loading, setLoading] = useState(false)

  async function handleUpload(e) {
    e.preventDefault()
    if (!selectedFile) {
      alert("Please select a file first")
      return
    }

    setLoading(true)
    try {
      const formData = new FormData()
      formData.append("file", selectedFile)

      console.log("Uploading file:", selectedFile.name, selectedFile.type, selectedFile.size)

      const response = await fetch(`${FETCH_URL}api/v1/file_upload/`, {
        method: "POST",
        body: formData,
      })

      console.log("Response status:", response.status, response.ok)

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Upload failed: ${response.status} - ${errorText}`)
      }

      // Получаем blob данных
      const blob = await response.blob()
      console.log("Received blob:", blob.size, blob.type)
      
      if (blob.size === 0) {
        throw new Error("Received empty blob from server")
      }

      // Создаем URL для отображения
      const url = URL.createObjectURL(blob)
      console.log("Created blob URL:", url)
      
      setImageUrl(url)
      setSelectedFile(null)
      
    } catch (error) {
      console.error("Upload error:", error)
      alert("Upload failed: " + error.message)
    } finally {
      setLoading(false)
    }
  }

  function handleFileChange(e) {
    const file = e.target.files[0]
    if (file) {
      console.log("File selected:", file.name, file.type, file.size)
      setSelectedFile(file)
    }
  }

  function clearImage() {
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl)
      setImageUrl(null)
    }
  }

  return (
    <div style={{ padding: "20px" }}>
      {imageUrl ? (
        <div style={{ marginBottom: "20px" }}>
          <h2>Uploaded Image:</h2>
          <img 
            src={imageUrl} 
            alt="Uploaded preview" 
            style={{ 
              maxWidth: "100%", 
              maxHeight: "500px",
              border: "2px solid #007bff",
              borderRadius: "8px",
              display: "block"
            }} 
            onLoad={() => console.log("Image loaded successfully")}
            onError={() => console.log("Error loading image")}
          />
          <button 
            onClick={clearImage}
            style={{ 
              marginTop: "10px", 
              padding: "10px 20px",
              backgroundColor: "#dc3545",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer"
            }}
          >
            Clear Image
          </button>
        </div>
      ) : (
        <p>No image displayed</p>
      )}

      <form onSubmit={handleUpload} style={{ marginTop: "20px" }}>
        <div style={{ marginBottom: "10px" }}>
          <input 
            type="file" 
            accept="image/*"
            onChange={handleFileChange}
            disabled={loading}
            style={{ padding: "10px" }}
          />
        </div>
        <button 
          type="submit" 
          disabled={!selectedFile || loading}
          style={{ 
            padding: "10px 20px",
            backgroundColor: selectedFile && !loading ? "#007bff" : "#6c757d",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: selectedFile && !loading ? "pointer" : "not-allowed"
          }}
        >
          {loading ? "Uploading..." : "Upload Image"}
        </button>
      </form>

      {selectedFile && !loading && (
        <p style={{ marginTop: "10px" }}>
          Selected file: <strong>{selectedFile.name}</strong> ({Math.round(selectedFile.size / 1024)} KB)
        </p>
      )}

      {loading && (
        <p style={{ marginTop: "10px", color: "#007bff" }}>Uploading file...</p>
      )}
    </div>
  )
}