from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware


from app.model import process_image_and_preview


app = FastAPI(
    title="Sonoma",
    version="0.1.0",
    contact={
        "name": "Roshchin Philipp",
        "url": "https://github.com/PHILIPP111007",
        "email": "r.phil@yandex.ru",
    },
    license_info={
        "name": "MIT",
        "identifier": "MIT",
    },
    openapi_url="/docs/openapi.json",
)

app.openapi_version = "3.0.0"


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.post("/api/v1/file_upload/")
# async def upload_file(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()

#         return Response(content=contents, media_type=file.content_type or "image/jpeg")

#     except Exception as e:
#         print(e)


@app.post("/api/v1/file_upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        images = process_image_and_preview(image=content)

        return Response(content=images, media_type=file.content_type or "image/jpeg")

    except Exception as e:
        print(e)
