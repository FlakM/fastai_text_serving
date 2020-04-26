import asyncio
import logging

import aiohttp
import uvicorn
from fastai.vision import *
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# put your url here here
model_file_url = 'https://www.dropbox.com/s/...?raw=1'
model_file_name = 'model'
path = Path(__file__).parent

logging.basicConfig(format="%(levelname)s:     %(message)s", level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])


def hashsum(path, hex=True, hash_type=hashlib.md5):
    hashinst = hash_type()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(hashinst.block_size * 128), b''):
            hashinst.update(chunk)
    return hashinst.hexdigest() if hex else hashinst.digest()


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


async def setup_learner():
    model_file = path.parent / 'models' / f'{model_file_name}.pkl'
    if not model_file.exists():
        logging.info("Will download file %s from %s", model_file, model_file_url)
        await download_file(model_file_url, model_file)
        logging.info("Downloaded file md5sum: %s", hashsum(model_file))
    else:
        logging.info("File %s already exists will reuse md5sum: %s", model_file,  hashsum(model_file))

    # Loading the saved model using fastai's load_learner method
    model = load_learner(model_file.parent, f'{model_file_name}.pkl')
    classes = model.data.classes
    return model, classes


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
model, classes = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


def sortByProb(val):
    return val["prob"]


@app.route('/predict', methods=['POST'])
async def analyze(request):
    data = await request.form()
    text = data['text']
    predict_class, predict_idx, predict_values = model.predict(text)

    results = []

    for idx, val in enumerate(predict_values):
        prob = val.item()
        if prob > 0.01:
            record = {"value": classes[idx], "prob": prob}
            results.append(record)

    results.sort(key=sortByProb, reverse=True)
    return JSONResponse(results[:5])


if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0' port=4000)
