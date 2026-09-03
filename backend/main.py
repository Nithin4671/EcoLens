from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.gemini_service import analyze_waste


app = FastAPI(title="EcoLens API")


# Allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "EcoLens API is running 🌱"
    }


async def process_image(image: UploadFile):
    """Read an uploaded image and send it to Gemini."""

    # Make sure the uploaded file is actually an image
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    # Read image data
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty image file."
        )

    try:
        # Send image to Gemini
        result = analyze_waste(
            image_bytes,
            image.content_type
        )

        return {
            "result": result
        }

    except ValueError as e:
        raise HTTPException(
            status_code=502,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {e}"
        )


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    return await process_image(image)


@app.post("/classify")
async def classify(image: UploadFile = File(...)):
    return await process_image(image)