from fastapi import FastAPI, File, UploadFile
import cv2  # Import OpenCV
import numpy as np  # Import NumPy
import inference
import postprocessing

app = FastAPI()

@app.post("/meterimage")
async def upload_image(image: UploadFile = File(...)):
    # Read the uploaded image data
    content_bytes = await image.read()
    # Decode the bytes to a NumPy array
    image_array = cv2.imdecode(np.frombuffer(content_bytes, np.uint8), cv2.IMREAD_COLOR)
    number_list = inference.readmeterimage(image_array)
    meter_reading = postprocessing.format_output(number_list)
    # You can now process the image here using OpenCV functions (e.g., convert to grayscale, detect objects)
    # For this example, let's just get image dimensions
    # height, width, channels = image_array.shape
    
    # # Return a JSON response with image details
    return {
        "reading_value": meter_reading
    }
    # return 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)